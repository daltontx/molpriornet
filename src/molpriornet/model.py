import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class AttMPN(nn.Module):
    """
    Attention-based  Molecular Prior Network.
    """

    class Block(nn.Module):
        def __init__(self, dim: int, vdim: int | None = None) -> None:
            """
            Initialize the Block.
            Args:
                dim: The hidden dimension.
                vdim: The dimension of the values for attention.
                    If None, it is set to the same as the hidden dimension.
            Returns:
                None.
            """
            super().__init__()
            self.proj = nn.Linear(dim, 4 * dim)
            self.attn = nn.MultiheadAttention(dim, num_heads=1, vdim=vdim)
            self.proj_out = nn.Linear(4 * dim, dim)

        def forward(
            self, h: torch.Tensor, v: torch.Tensor, temperature: nn.Parameter
        ) -> torch.Tensor:
            """
            Forward pass.
            Args:
                h: The hidden state.
                v: The values for attention.
                temperature: The softmax temperature.
            Returns:
                The updated hidden state.
            """
            h_n = F.layer_norm(h, h.shape[1:])
            h = h + self.attn(h_n / temperature, h_n[: len(v)], v)[0]
            h_n = F.layer_norm(h, h.shape[1:])
            h_p = self.proj(h_n)
            h_p = F.relu(h_p).square()
            h = h + self.proj_out(h_p)
            return h

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 128,
        n_layers: int = 1,
        predict_variance: bool = False,
    ) -> None:
        """
        Initialize the AttMPN.
        Args:
            n_features: The number of features of the input molecules.
            n_hidden: The number of hidden units.
            n_layers: The number of layers.
            predict_variance: If True, the output head predicts both the mean
                and the log-variance of a Gaussian over the label, enabling
                heteroscedastic uncertainty estimation. If False (default), the
                model is a plain point regressor.
        Returns:
            None.
        """
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.predict_variance = predict_variance
        self.n_label_dim = int(n_hidden / 4)

        self.label_in = nn.Linear(1, self.n_label_dim, bias=False)
        self.v_in = nn.Linear(n_features + self.n_label_dim, n_hidden)
        self.h_in = nn.Linear(n_features, n_hidden)

        self.blocks = nn.ModuleList(
            [
                AttMPN.Block(n_hidden, vdim=self.n_hidden + self.n_label_dim)
                for _ in range(n_layers)
            ]
        )

        self.out = nn.Linear(self.n_hidden, 2 if predict_variance else 1)

        self.temperature = nn.Parameter(torch.tensor(0.2))

    def forward(
        self, x_support: torch.Tensor, y_support: torch.Tensor, x_query: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_support: The features of the support set.
            y_support: The labels of the support set.
            x_query: The features of the query set.
        Returns:
            The predicted query labels, shape (n_query, 1). If predict_variance
            is True, shape (n_query, 2) where column 0 is the predicted mean and
            column 1 is the predicted log-variance.
        """

        h = torch.cat([x_support, x_query], dim=0)
        h = self.h_in(h)

        y_support = self.label_in(y_support)
        v = torch.cat([h[: len(x_support)], y_support], dim=1)

        for block in self.blocks:
            h = block(h, v, self.temperature)
            v = torch.cat([h[: len(x_support)], y_support], dim=1)

        out = self.out(h[-len(x_query) :, :])
        return out

    def save(self, model_path: str) -> None:
        """
        Save the model.
        Args:
            model_path: The path to save the model.
        Returns:
            None.
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "n_features": self.n_features,
                "hidden_size": self.n_hidden,
                "n_layers": self.n_layers,
                "predict_variance": self.predict_variance,
            },
            model_path,
        )

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        SK-learn interface. Saves the support set features and labels.
        Args:
            train_x: The features of the training set.
            train_y: The labels of the training set.
        Returns:
            None.
        """
        self.x_support = torch.tensor(train_x, dtype=torch.float32)
        self.y_support = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """
        SK-learn interface. Predicts the labels of the query set using the support set.
        Args:
            test_x: The features of the query set.
        Returns:
            The predicted query labels.
        """
        x_query = torch.tensor(test_x, dtype=torch.float32)
        out = self.forward(self.x_support, self.y_support, x_query).detach()
        # Column 0 is always the mean; ignore any variance column for the
        # point-prediction interface.
        return out[:, 0].numpy()

    def predict_with_uncertainty(
        self, test_x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SK-learn-style interface returning predictions and their standard
        deviation. Only valid when the model was built with predict_variance.
        Args:
            test_x: The features of the query set.
        Returns:
            A tuple (mean, std), each of shape (n_query,).
        """
        if not self.predict_variance:
            raise ValueError(
                "predict_with_uncertainty requires a model trained with "
                "predict_variance=True."
            )
        x_query = torch.tensor(test_x, dtype=torch.float32)
        out = self.forward(self.x_support, self.y_support, x_query).detach()
        mean = out[:, 0]
        std = (0.5 * out[:, 1]).exp()  # std = exp(log_var / 2)
        return mean.numpy(), std.numpy()

    def predict_with_distances(
        self, test_x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts query labels and returns cosine similarity metrics against the support set.
        Args:
            test_x: The features of the query set.
        Returns:
            A tuple of five arrays, each of shape (n_query,):
            - predictions: the predicted labels.
            - fp_nearest_sim: cosine similarity to the nearest support fingerprint.
            - fp_mean_sim: mean cosine similarity across the full support fingerprints.
            - hidden_nearest_sim: cosine similarity to the nearest support final hidden state.
            - hidden_mean_sim: mean cosine similarity across all support final hidden states.
        """
        x_query = torch.tensor(test_x, dtype=torch.float32)

        with torch.no_grad():
            # Fingerprint-space cosine similarities.
            fp_q = F.normalize(x_query, p=2, dim=1)
            fp_s = F.normalize(self.x_support, p=2, dim=1)
            fp_sims = torch.matmul(fp_q, fp_s.T)  # (n_query, n_support)
            fp_nearest_sim = fp_sims.max(dim=1).values
            fp_mean_sim = fp_sims.mean(dim=1)

            # Re-run forward pass, mirroring self.forward, to capture final hidden states.
            h = torch.cat([self.x_support, x_query], dim=0)
            h = self.h_in(h)
            y_support = self.label_in(self.y_support)
            v = torch.cat([h[: len(self.x_support)], y_support], dim=1)
            for block in self.blocks:
                h = block(h, v, self.temperature)
                v = torch.cat([h[: len(self.x_support)], y_support], dim=1)

            h_support = h[: len(self.x_support)]
            h_query = h[-len(x_query) :]

            # Column 0 is the mean (a variance column may follow it).
            predictions = self.out(h_query)[:, 0].numpy()

            # Hidden-state cosine similarities.
            hs_q = F.normalize(h_query, p=2, dim=1)
            hs_s = F.normalize(h_support, p=2, dim=1)
            hs_sims = torch.matmul(hs_q, hs_s.T)  # (n_query, n_support)
            hidden_nearest_sim = hs_sims.max(dim=1).values
            hidden_mean_sim = hs_sims.mean(dim=1)

        return (
            predictions,
            fp_nearest_sim.numpy(),
            fp_mean_sim.numpy(),
            hidden_nearest_sim.numpy(),
            hidden_mean_sim.numpy(),
        )

    @staticmethod
    def load_model(model_path: str) -> "AttMPN":
        """
        Load the model.
        Args:
            model_path: The path to the model file.
        Returns:
            The AttMPN model.
        """
        checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
        model = AttMPN(
            n_features=checkpoint["n_features"],
            n_hidden=checkpoint["hidden_size"],
            n_layers=checkpoint["n_layers"],
            # Default to False for checkpoints saved before this flag existed.
            predict_variance=checkpoint.get("predict_variance", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


class KernelMPN(nn.Module):
    """
    The Soft Nearest Neighbors model.
    """

    def __init__(self, dim: int) -> None:
        """
        Initialize the KernelMPN.
        Args:
            dim: The dimension of the input features.
        Returns:
            None.
        """
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, x_support: torch.Tensor, y_support: torch.Tensor, x_query: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x_support: The features of the support set.
            y_support: The labels of the support set.
            x_query: The features of the query set.
        Returns:
            The predicted query labels.
        """
        x_query = self.proj(x_query)
        x_support = self.proj(x_support)
        query_norm = F.normalize(x_query, p=2, dim=1)
        support_norm = F.normalize(x_support, p=2, dim=1)
        weights = torch.matmul(query_norm, support_norm.T) / self.temperature
        weights = F.softmax(weights, dim=1)

        return torch.matmul(weights, y_support)

    def save(self, model_path: str) -> None:
        """
        Save the model.
        Args:
            model_path: The path to the save file.
        Returns:
            None.
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
            },
            model_path,
        )

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        SK-learn interface. Saves the support set features and labels.
        Args:
            train_x: The features of the training set.
            train_y: The labels of the training set.
        Returns:
            None.
        """
        self.x_support = torch.tensor(train_x, dtype=torch.float32)
        self.y_support = torch.tensor(train_y, dtype=torch.float32)

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """
        SK-learn interface. Predicts the labels of the query set using the support set.
        Args:
            test_x: The features of the query set.
        Returns:
            The predicted query labels.
        """
        x_query = torch.tensor(test_x, dtype=torch.float32)
        return self.forward(self.x_support, self.y_support, x_query).detach().numpy()

    @staticmethod
    def load_model(model_path: str) -> "KernelMPN":
        """
        Load the model.
        Args:
            model_path: The path to the model file.
        Returns:
            The KernelMPN model.
        """
        checkpoint = torch.load(model_path, weights_only=True, map_location="cpu")
        model = KernelMPN(checkpoint["model_state_dict"]["proj.weight"].shape[1])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

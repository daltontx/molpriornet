import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import yaml
from rdkit import Chem  # noqa: F401
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from molpriornet.fsmol_loader import FSMolLoader
from molpriornet.model import AttMPN, KernelMPN

DATASET_SIZE_LIMIT = 500
SUPPORT_SIZES = [16, 32, 64, 128, 256]

with pathlib.Path(sys.argv[1]).open("r") as f:
    config = yaml.safe_load(f)

train_dir = pathlib.Path(config["train_dir"])
val_dir = pathlib.Path(config["val_dir"])
experiment_name = config.get("experiment_name", None)
tracking_uri = config.get("tracking_uri", os.environ.get("MLFLOW_TRACKING_URI"))
fingerprint_size = config.get("fingerprint_size", 500)
hidden_size = config.get("hidden_size", 128)
n_layers = config.get("n_layers", 1)
learning_rate = config.get("learning_rate", 0.001)
batch_size = config.get("batch_size", 1)
checkpoint_path = config.get("checkpoint_path", None)
model_name = config.get("model", "AttMPN")
partition_file = config.get("partition_file", None)
train_partition = config.get("train_partition", None)
val_partition = config.get("val_partition", None)
output_path = config.get("output_path", "best_model.pth")
predict_variance = config.get("predict_variance", False)

# Resolve the output path up front, so a bad destination fails before training
# rather than at the first checkpoint. Cloud URIs (s3://, gs://, az://) need
# cloudpathlib, which is optional: install it with the backend you use, e.g.
# `pip install cloudpathlib[s3]`.
try:
    import cloudpathlib

    output = cloudpathlib.AnyPath(output_path)
except ImportError:
    if "://" in output_path:
        sys.exit(
            f"Writing to '{output_path}' requires cloudpathlib; install it for "
            "your backend (e.g. 'pip install cloudpathlib[s3]'), or set "
            "output_path to a local path."
        )
    output = pathlib.Path(output_path)

# MLflow tracking is optional: without both an experiment name and a tracking
# URI, metrics are only printed to stdout.
use_mlflow = bool(experiment_name and tracking_uri)
if experiment_name and not tracking_uri:
    print(
        "Warning: 'experiment_name' is set but no tracking URI was found; "
        "set 'tracking_uri' in the config or MLFLOW_TRACKING_URI in the "
        "environment to enable MLflow logging."
    )
if use_mlflow:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_params(config)

train_set_loader = FSMolLoader(
    train_dir,
    FSMolLoader.TargetType.LOG_REGRESSION_PROPERTY,
    shuffle=True,
    fingerprint_size=fingerprint_size,
    cache_dir="loader_cache",
    partition_file=partition_file,
    partition=train_partition,
)

val_set_loader = FSMolLoader(
    val_dir,
    FSMolLoader.TargetType.LOG_REGRESSION_PROPERTY,
    shuffle=False,
    fingerprint_size=fingerprint_size,
    cache_dir="loader_cache",
    partition_file=partition_file,
    partition=val_partition,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if checkpoint_path:
    if model_name == "AttMPN":
        model = AttMPN.load_model(checkpoint_path)
    elif model_name == "KernelMPN":
        model = KernelMPN.load_model(checkpoint_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")
else:
    if model_name == "AttMPN":
        model = AttMPN(
            n_features=fingerprint_size,
            n_hidden=hidden_size,
            n_layers=n_layers,
            predict_variance=predict_variance,
        )
    elif model_name == "KernelMPN":
        model = KernelMPN(dim=fingerprint_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if predict_variance:
    # Gaussian negative log-likelihood: L = 0.5 * [(y - mu)^2 / var + log var].
    # The model head emits log-variance, so we exponentiate to get var.
    gaussian_nll = torch.nn.GaussianNLLLoss(full=True)

    def loss_fn(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL over a mean / log-variance prediction pair."""
        mean = preds[:, 0:1]
        log_var = preds[:, 1:2]
        return gaussian_nll(mean, target, log_var.exp())
else:
    loss_fn = torch.nn.MSELoss()

min_val_mae = float("inf")
train_step_counter = 0
for epoch in range(config["epochs"]):
    epoch_start = time.time()
    train_losses = []
    train_maes = []
    for i, (path, mols, labels) in enumerate(train_set_loader):
        if len(mols) > DATASET_SIZE_LIMIT:
            indices = random.sample(range(len(mols)), DATASET_SIZE_LIMIT)
            mols = mols[indices]
            labels = labels[indices]

        try:
            task_mean = np.mean(labels)
            task_std = np.std(labels)
            if task_std < 1e-8:
                continue
            labels = [(y - task_mean) / task_std for y in labels]

            if SUPPORT_SIZES:
                train_size = min(int(len(mols) * 0.8), random.choice(SUPPORT_SIZES))
                support_x, query_x, support_y, query_y = train_test_split(
                    mols, labels, train_size=train_size
                )
            else:
                support_x, query_x, support_y, query_y = train_test_split(
                    mols, labels, test_size=0.2
                )
            support_x = torch.tensor(support_x, dtype=torch.float32, device=device)
            support_y = torch.tensor(
                support_y, dtype=torch.float32, device=device
            ).reshape(-1, 1)
            query_x = torch.tensor(query_x, dtype=torch.float32, device=device)
            query_y = torch.tensor(query_y, dtype=torch.float32, device=device).reshape(
                -1, 1
            )
        except Exception as e:  # noqa: BLE001 - skip any task we cannot batch
            print(f"Error getting sets for {path}: {e}")
            print(f"Skipping {path}...")
            continue

        preds = model(support_x, support_y, query_x)
        loss = loss_fn(preds, query_y)
        loss.backward()
        # optimizer.step()
        if (i > 0) and (i % batch_size == 0):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=100.0
            )  # float("inf")
            optimizer.step()
            optimizer.zero_grad()

        print(f"{i} {path}: {loss.item()}")
        train_losses.append(loss.item())
        if predict_variance:
            # The loss is NLL, so also track point accuracy of the mean column.
            # Scaled units, matching the (also scaled) training loss above.
            train_maes.append((preds[:, 0:1].detach() - query_y).abs().mean().item())

        if i > 0 and i % 100 == 0:
            avg_loss = np.mean(np.array(train_losses))
            if use_mlflow:
                mlflow.log_metric("train_loss", avg_loss, step=train_step_counter)
                mlflow.log_metric(
                    "grad_norm", grad_norm.item(), step=train_step_counter
                )
            print(f"Training loss: {avg_loss}, Grad norm: {grad_norm.item()}")
            if predict_variance:
                avg_mae = np.mean(np.array(train_maes))
                if use_mlflow:
                    mlflow.log_metric("train_mae", avg_mae, step=train_step_counter)
                print(f"Training MAE: {avg_mae}")
                train_maes = []
            train_losses = []
            train_step_counter += 1

    with torch.no_grad():
        fs_rng = random.Random(0)
        val_maes = []
        val_nlls = []
        for path, mols, labels in val_set_loader:
            if len(mols) > DATASET_SIZE_LIMIT:
                rng = random.Random(0)
                indices = rng.sample(range(len(mols)), DATASET_SIZE_LIMIT)
                mols = mols[indices]
                labels = labels[indices]

            try:
                if SUPPORT_SIZES:
                    train_size = min(int(len(mols) * 0.8), fs_rng.choice(SUPPORT_SIZES))
                    support_x, query_x, support_y, query_y = train_test_split(
                        mols, labels, train_size=train_size, random_state=0
                    )
                else:
                    support_x, query_x, support_y, query_y = train_test_split(
                        mols, labels, test_size=0.2, random_state=0
                    )

                task_mean = np.mean(support_y)
                task_std = np.std(support_y)
                if task_std < 1e-8:
                    continue
                support_y_scaled = [(y - task_mean) / task_std for y in support_y]

                support_x = torch.tensor(support_x, dtype=torch.float32, device=device)
                support_y = torch.tensor(
                    support_y_scaled, dtype=torch.float32, device=device
                ).reshape(-1, 1)
                query_x = torch.tensor(query_x, dtype=torch.float32, device=device)
                query_y = torch.tensor(
                    query_y, dtype=torch.float32, device=device
                ).reshape(-1, 1)
            except Exception as e:  # noqa: BLE001 - skip any task we cannot batch
                print(f"Error getting sets for {path}: {e}")
                print(f"Skipping {path}...")
                continue
            out_scaled = model(support_x, support_y, query_x)
            # Column 0 is the (scaled) mean; undo the support-set standardisation.
            preds = out_scaled[:, 0:1] * task_std + task_mean
            mae = mean_absolute_error(query_y.cpu(), preds.cpu())
            val_maes.append(mae)

            if predict_variance:
                # Undo standardisation in log-variance space: a factor of
                # task_std on the labels scales the variance by task_std**2.
                log_var = out_scaled[:, 1:2] + 2 * np.log(task_std)
                nll = gaussian_nll(preds, query_y, log_var.exp())
                val_nlls.append(nll.item())
    val_mae = np.mean(np.array(val_maes))
    if use_mlflow:
        mlflow.log_metric("Validation MAE", val_mae, step=epoch)
    print(f"Validation MAE: {val_mae}")
    if predict_variance:
        val_nll = np.mean(np.array(val_nlls))
        if use_mlflow:
            mlflow.log_metric("Validation NLL", val_nll, step=epoch)
        print(f"Validation NLL: {val_nll}")
    if val_mae < min_val_mae:
        min_val_mae = val_mae
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            model.save(f)

    epoch_duration = time.time() - epoch_start
    print(f"Epoch {epoch} time: {epoch_duration:.2f} seconds")

    if use_mlflow:
        mlflow.log_metric("Epoch duration", epoch_duration, step=epoch)

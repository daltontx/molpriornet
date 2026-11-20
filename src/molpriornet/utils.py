"""Shared helpers for featurizing molecules and preparing support/query sets."""

import random

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split

from molpriornet.model import AttMPN, KernelMPN

MODEL_CLASSES = {"AttMPN": AttMPN, "KernelMPN": KernelMPN}


def get_fingerprint_size(model: AttMPN | KernelMPN) -> int:
    """
    Get the input fingerprint size a model expects.
    Args:
        model: The loaded model.
    Returns:
        The fingerprint size.
    """
    if isinstance(model, KernelMPN):
        # KernelMPN checkpoints don't store the size; read it off the projection.
        return model.proj.in_features
    return model.n_features


def featurize_smiles(
    smiles: list[str], fingerprint_size: int
) -> tuple[np.ndarray, list[int]]:
    """
    Encode SMILES strings as Morgan fingerprints.
    Args:
        smiles: The SMILES strings.
        fingerprint_size: The fingerprint size the model expects.
    Returns:
        A tuple (features, kept), where ``kept`` holds the positions in
        ``smiles`` of the molecules that could be encoded. Molecules RDKit
        cannot parse are skipped, so ``features`` may be shorter than the input.
    """
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=fingerprint_size
    )

    x = []
    kept = []
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(mol)
            fps = morgan_generator.GetFingerprint(mol)
        except Exception as e:  # noqa: BLE001 - RDKit raises a variety of errors
            print(f"Error generating fingerprint for {smi}: {e}")
            continue
        x.append(fps)
        kept.append(i)

    return np.array(x), kept


def featurize(
    df: pd.DataFrame,
    smiles_column: str,
    target_column: str,
    fingerprint_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop rows with a missing label and encode the molecules as fingerprints.
    Args:
        df: The dataframe to featurize.
        smiles_column: The column holding the SMILES strings.
        target_column: The column holding the labels.
        fingerprint_size: The fingerprint size the model expects.
    Returns:
        A tuple (features, labels). Molecules RDKit cannot parse are skipped.
    """
    df = df[[smiles_column, target_column]].dropna()
    x, kept = featurize_smiles(df[smiles_column].tolist(), fingerprint_size)
    y = df[target_column].to_numpy(dtype=np.float32)[kept]
    return x, y


def support_statistics(support_y: np.ndarray) -> tuple[float, float]:
    """
    Get the support set statistics used to standardise the labels.
    Args:
        support_y: The support set labels.
    Returns:
        A tuple (mean, std). The std falls back to 1.0 when the support labels
        are (nearly) constant, which would otherwise divide by zero.
    """
    mean = float(np.mean(support_y))
    std = float(np.std(support_y))
    if std < 1e-8:
        print("Warning: the support labels have near-zero spread; using std = 1.")
        std = 1.0
    return mean, std


def get_support_query_sets(
    smiles: list[str],
    labels: list[float],
    fingerprint_size: int,
    dataset_size_limit: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the support and query sets.
    Args:
        smiles: The SMILES strings.
        labels: The labels.
        fingerprint_size: The fingerprint size.
        dataset_size_limit: The dataset size limit.
        random_state: The random state.
    Returns:
        A tuple containing the support and query sets.
    """
    if random_state is not None:
        rng = random.Random(random_state)
    else:
        rng = random.Random()

    if dataset_size_limit and len(smiles) > dataset_size_limit:
        smiles_labels = rng.sample(list(zip(smiles, labels)), dataset_size_limit)
    else:
        smiles_labels = list(zip(smiles, labels))

    sampled_smiles = [smi for smi, _ in smiles_labels]
    sampled_labels = [label for _, label in smiles_labels]

    x, kept = featurize_smiles(sampled_smiles, fingerprint_size)
    y = np.array([sampled_labels[i] for i in kept])

    support_x, query_x, support_y, query_y = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    return support_x, support_y, query_x, query_y

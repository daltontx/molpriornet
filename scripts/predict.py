"""Predict labels for unlabelled molecules with a trained AttMPN or KernelMPN.

Takes a labelled support CSV and a query CSV that only needs SMILES, and
writes the query rows back out with a ``prediction`` column. For an AttMPN
checkpoint trained with ``predict_variance``, a ``std`` column is added too.

Predictions are in the units of the support labels: if the model was trained on
log-scale labels (as with FS-Mol's LogRegressionProperty), the support labels
should be on that scale as well.

To score a model against known labels instead, use ``evaluate.py``.

Usage:
    python predict.py <model_path> <support_path> <query_path> \\
        --smiles-column SMILES --target-column <column>
        [--model {AttMPN,KernelMPN}] [--output predictions.csv]
        [--dataset-size-limit N] [--seed 0]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from molpriornet.model import AttMPN
from molpriornet.utils import (
    MODEL_CLASSES,
    featurize,
    featurize_smiles,
    get_fingerprint_size,
    support_statistics,
)


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict labels for unlabelled molecules with an AttMPN or "
        "KernelMPN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", help="Path to the trained model checkpoint")
    parser.add_argument(
        "support_path", help="Path to the CSV holding the labelled support data"
    )
    parser.add_argument(
        "query_path",
        help="Path to the CSV of molecules to predict. Only the SMILES column "
        "is required; any other columns are copied to the output.",
    )
    parser.add_argument(
        "--smiles-column",
        required=True,
        help="Column containing the SMILES strings, in both CSVs.",
    )
    parser.add_argument(
        "--target-column",
        required=True,
        help="Column containing the labels, in the support CSV.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CLASSES),
        default="AttMPN",
        help="Type of the checkpoint's model.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="predictions.csv",
        help="Path of the CSV to write the predictions to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling the support set.",
    )
    parser.add_argument(
        "--dataset-size-limit",
        type=int,
        default=None,
        help="Optional cap on the number of support rows used.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Load a model, fit it on the support set, and write the query predictions.
    Returns:
        None.
    """
    args = parse_args()

    model = MODEL_CLASSES[args.model].load_model(args.model_path)
    model.eval()
    fingerprint_size = get_fingerprint_size(model)

    support_x, support_y = featurize(
        pd.read_csv(args.support_path),
        args.smiles_column,
        args.target_column,
        fingerprint_size,
    )
    if len(support_y) == 0:
        sys.exit(f"No usable labelled rows in {args.support_path}.")

    if args.dataset_size_limit and len(support_y) > args.dataset_size_limit:
        rng = np.random.default_rng(args.seed)
        subset = rng.choice(len(support_y), args.dataset_size_limit, replace=False)
        support_x, support_y = support_x[subset], support_y[subset]

    query_df = pd.read_csv(args.query_path)
    if args.smiles_column not in query_df.columns:
        sys.exit(f"Column '{args.smiles_column}' not found in {args.query_path}.")
    query_x, kept = featurize_smiles(
        query_df[args.smiles_column].tolist(), fingerprint_size
    )
    if len(kept) == 0:
        sys.exit(f"None of the molecules in {args.query_path} could be encoded.")

    # Standardise by the support statistics, mirroring training, then undo the
    # scaling so the predictions come back in the units of the support labels.
    task_mean, task_std = support_statistics(support_y)
    model.fit(support_x, (support_y - task_mean) / task_std)

    # Rows whose SMILES could not be encoded keep a missing prediction.
    predictions = np.full(len(query_df), np.nan)
    stds = None
    if isinstance(model, AttMPN) and model.predict_variance:
        mean_scaled, std_scaled = model.predict_with_uncertainty(query_x)
        predictions[kept] = mean_scaled * task_std + task_mean
        stds = np.full(len(query_df), np.nan)
        stds[kept] = std_scaled * task_std
    else:
        predictions[kept] = model.predict(query_x) * task_std + task_mean

    query_df["prediction"] = predictions
    if stds is not None:
        query_df["std"] = stds

    query_df.to_csv(args.output, index=False)
    print(f"Support: {len(support_y)}, predicted: {len(kept)}/{len(query_df)}")
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()

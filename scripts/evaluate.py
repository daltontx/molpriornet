"""Evaluate a trained AttMPN or KernelMPN model on a single target in a CSV.

Rows with a missing label are dropped. The support and query sets are then
formed in one of two ways:

  - without ``--query-path``: the data is split randomly (80/20 by default);
  - with ``--query-path``: the whole input file is the support set and the
    whole query file is the query set, used as is.

Reports MAE (in the original label units) and Kendall tau between the
predictions and the true labels on the query set, for the checkpoint and for
any baselines requested with ``--baselines``.

To predict labels for molecules that do not have any, use ``predict.py``.

Usage:
    python evaluate.py <model_path> <data_path> --smiles-column SMILES \\
        --target-column <column> [--model {AttMPN,KernelMPN}]
        [--query-path <csv>] [--query-size 0.2] [--seed 0]
        [--dataset-size-limit N] [--baselines rf]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from molpriornet.utils import (
    MODEL_CLASSES,
    featurize,
    get_fingerprint_size,
    support_statistics,
)

# Reference models to score alongside the checkpoint, keyed by their CLI name:
# (label used in the results table, factory taking the random seed).
BASELINES = {
    "rf": (
        "RandomForest",
        lambda seed: RandomForestRegressor(n_estimators=100, random_state=seed),
    ),
}


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an AttMPN or KernelMPN checkpoint on one target "
        "of a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", help="Path to the trained model checkpoint")
    parser.add_argument("data_path", help="Path to the CSV holding the support data")
    parser.add_argument(
        "--smiles-column", required=True, help="Column containing the SMILES strings."
    )
    parser.add_argument(
        "--target-column", required=True, help="Column containing the labels."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CLASSES),
        default="AttMPN",
        help="Type of the checkpoint's model.",
    )
    parser.add_argument(
        "--query-path",
        default=None,
        help="Optional CSV to use as the query set as is, instead of splitting "
        "the input randomly.",
    )
    parser.add_argument(
        "--query-size",
        type=float,
        default=0.2,
        help="Fraction held out as the query set. Ignored with --query-path.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINES),
        default=[],
        metavar="NAME",
        help=f"Reference models to score alongside the checkpoint. "
        f"Choices: {', '.join(BASELINES)}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling and the support/query split.",
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
    Load a model and the data, evaluate one target, and print the metrics.
    Returns:
        None.
    """
    args = parse_args()

    model = MODEL_CLASSES[args.model].load_model(args.model_path)
    model.eval()
    fingerprint_size = get_fingerprint_size(model)

    x, y = featurize(
        pd.read_csv(args.data_path),
        args.smiles_column,
        args.target_column,
        fingerprint_size,
    )

    if args.query_path:
        # Use both files as they are: all of the input supports all of the query set.
        support_x, support_y = x, y
        query_x, query_y = featurize(
            pd.read_csv(args.query_path),
            args.smiles_column,
            args.target_column,
            fingerprint_size,
        )
    else:
        support_x, query_x, support_y, query_y = train_test_split(
            x, y, test_size=args.query_size, random_state=args.seed
        )

    if args.dataset_size_limit and len(support_y) > args.dataset_size_limit:
        rng = np.random.default_rng(args.seed)
        subset = rng.choice(len(support_y), args.dataset_size_limit, replace=False)
        support_x, support_y = support_x[subset], support_y[subset]

    # Standardise by the support statistics, mirroring training/validation.
    task_mean, task_std = support_statistics(support_y)
    support_y_scaled = (support_y - task_mean) / task_std

    models = {args.model: model}
    for name in args.baselines:
        models[BASELINES[name][0]] = BASELINES[name][1](args.seed)

    records = []
    for name, m in models.items():
        m.fit(support_x, support_y_scaled)
        preds = m.predict(query_x) * task_std + task_mean
        records.append(
            {
                "model": name,
                "MAE": mean_absolute_error(query_y, preds),
                "kendall_tau": kendalltau(preds, query_y).statistic,
            }
        )

    print(f"Target: {args.target_column}")
    print(f"Support: {len(support_y)}, query: {len(query_y)}")
    print(pd.DataFrame(records).set_index("model").to_string())


if __name__ == "__main__":
    main()

import glob
import gzip
import json
import os
import pickle
import random
from collections.abc import Generator
from enum import StrEnum
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import FingerprintGenerator64


def _list_partition_paths(
    data_dir: str,
    partition_file: str | None = None,
    partition: str | None = None,
) -> list[str]:
    """
    List ``*.jsonl.gz`` files in ``data_dir``, optionally filtered to a partition.

    Args:
        data_dir: The directory containing the dataset files.
        partition_file: (Optional) path to a JSON file mapping partition names
            (e.g. ``"train"``, ``"valid"``, ``"test"``) to lists of dataset
            identifiers (file basenames without extension, e.g. ``"CHEMBL12345"``).
        partition: (Optional) the partition name to select from ``partition_file``.

    Returns:
        A sorted list of dataset file paths. If both ``partition_file`` and
        ``partition`` are provided, only files whose basename (without
        extensions) appears in the selected partition are returned.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.jsonl.gz")))
    if partition_file is None and partition is None:
        return paths
    if partition_file is None or partition is None:
        raise ValueError(
            "Both 'partition_file' and 'partition' must be provided together."
        )
    with Path(partition_file).open("r") as f:
        partition_data = json.load(f)
    if partition not in partition_data:
        raise ValueError(
            f"Partition '{partition}' not found in {partition_file}; "
            f"available partitions: {sorted(partition_data.keys())}"
        )
    allowed_ids = set(partition_data[partition])
    return [p for p in paths if os.path.basename(p).split(".")[0] in allowed_ids]


class FSMolLoader:
    """
    A dataloader for the FS-Mol benchmark.
    """

    class TargetType(StrEnum):
        PROPERTY = "Property"
        REGRESSION_PROPERTY = "RegressionProperty"
        LOG_REGRESSION_PROPERTY = "LogRegressionProperty"

    def __init__(
        self,
        data_dir: str,
        target_type: TargetType,
        shuffle: bool = False,
        fingerprint_size: int | None = None,
        cache_dir: str | None = None,
        partition_file: str | None = None,
        partition: str | None = None,
    ) -> None:
        """
        Initialize the FSMolLoader.
        Args:
            data_dir: The directory containing the dataset.
            target_type: The target type of the dataset.
            shuffle: Whether to shuffle the dataset.
            fingerprint_size: The size of the fingerprint.
            cache_dir: The directory to cache the dataset.
            partition_file: (Optional) path to a JSON file defining dataset
                partitions (e.g. ``"train"``, ``"valid"``, ``"test"``). When
                provided together with ``partition``, only files belonging to
                the selected partition are loaded.
            partition: (Optional) name of the partition to load from
                ``partition_file``.
        Returns:
            None.
        """
        self.data_dir = data_dir
        self.target_type = target_type
        self.shuffle = shuffle
        self.fingerprint_size = fingerprint_size
        if self.fingerprint_size:
            self.morgan_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=self.fingerprint_size
            )
        else:
            self.morgan_generator = None
        self.cache_dir = cache_dir
        self.partition_file = partition_file
        self.partition = partition

        if self.cache_dir:
            cache_dir_fp_size = os.path.join(self.cache_dir, str(self.fingerprint_size))
            if not os.path.exists(cache_dir_fp_size):
                os.makedirs(cache_dir_fp_size)

    def _list_paths(self) -> list[str]:
        return _list_partition_paths(self.data_dir, self.partition_file, self.partition)

    def __iter__(
        self,
    ) -> Generator[tuple[str, list[np.ndarray], list[float]], None, None]:
        """
        Iterate over the dataset.
        Returns:
            A generator of tuples containing the path, molecules, and labels.
        """
        paths = self._list_paths()
        if self.shuffle:
            random.shuffle(paths)
        for path in paths:
            if self.cache_dir:
                cache_path = self._get_cache_path(path)
                if os.path.exists(cache_path):
                    with Path(cache_path).open("rb") as f:
                        mols, labels = pickle.load(f)
                else:
                    mols, labels = FSMolLoader.load_dataset(
                        path, self.target_type, self.morgan_generator
                    )
                    with Path(cache_path).open("wb") as f:
                        pickle.dump((mols, labels), f)
            else:
                mols, labels = FSMolLoader.load_dataset(
                    path, self.target_type, self.morgan_generator
                )
            yield path, mols, labels

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        Returns:
            The length of the dataset.
        """
        return len(self._list_paths())

    def _get_cache_path(self, path: str) -> str:
        return os.path.join(
            self.cache_dir,
            str(self.fingerprint_size),
            os.path.basename(path).split(".")[0] + ".pkl",
        )

    @staticmethod
    def load_dataset(
        path: str,
        target_type: TargetType,
        fingerprint_generator: FingerprintGenerator64 | None = None,
    ) -> tuple[list[str | np.ndarray], list[float]]:
        """
        Load the dataset from a file.
        Args:
            path: The path to the dataset file.
            target_type: The selected target type.
            fingerprint_generator: (Optional) the fingerprint generator to encode the molecules.
        Returns:
            A tuple containing the list of molecules and the list of their labels.
        """
        with gzip.open(path, "rt") as f:
            datapoints = [json.loads(line) for line in f]

        mols = []
        labels = []
        for dp in datapoints:
            try:
                s = dp["SMILES"]
                if fingerprint_generator:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None:
                        mol = Chem.MolFromSmiles(s, sanitize=False)
                        mol.UpdatePropertyCache(strict=False)
                        Chem.FastFindRings(mol)
                    m = fingerprint_generator.GetFingerprint(mol)
                else:
                    m = s
                lb = float(dp[target_type])
            except Exception as e:  # noqa: BLE001 - any malformed datapoint is skipped
                logger.debug(f"Skipping unparseable datapoint in {path}: {e}")
                continue
            mols.append(m)
            labels.append(lb)

        if fingerprint_generator:
            mols = np.array(mols, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

        return mols, labels

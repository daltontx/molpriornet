# Molecular Prior Networks

Meta-learned models for few-shot molecular property prediction. A Molecular
Prior Network is trained across many assays at once, so at inference time it
learns a new assay and predicts for it in a single forward pass — no
per-dataset retraining and no gradient steps. This is the approach of
Prior-data Fitted Networks (PFNs), applied to molecules.

The method is described in the [Molecular Prior
Networks](https://arxiv.org/abs/2611.05103) (arXiv:2611.05103) paper, accepted
to the NeurIPS 2026 Workshop on AI for Drug Discovery.

The package provides two models:

| Class | Description |
| ----- | ----------- |
| `AttMPN` | Attention-based network. The stronger model. Requires label normalization. |
| `KernelMPN` | Learned-kernel soft nearest neighbours. Small and fast, useful as a strong baseline. |

Both expose a scikit-learn-like `fit` / `predict` interface, where `fit` only
stores the support set.

## Installation

The package is not yet on PyPI; you can install it from GitHub:

```bash
pip install git+https://github.com/daltontx/molpriornet.git
```

To get the associated scripts in [scripts/](scripts/) as well,
work from a clone with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/daltontx/molpriornet.git
cd molpriornet
uv sync
```

Two packages are optional and used only if they are installed:

- [`mlflow`](https://mlflow.org/) — experiment tracking during training, also
  available as the `molpriornet[mlflow]` extra.
- [`cloudpathlib`](https://cloudpathlib.drivendata.org/) — lets `train.py`
  write checkpoints straight to cloud storage. Install it with the backend you
  use, for example `pip install cloudpathlib[s3]`.

## Featurization

The models expect the same fingerprint format they saw in training. The released
checkpoints use Morgan fingerprints of radius 2 and length 2048:

```python
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fingerprint = morgan_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
```

Each checkpoint records its own fingerprint size, so you can also read it back
from the loaded model:

```python
from molpriornet.utils import get_fingerprint_size

fingerprint_size = get_fingerprint_size(model)
```

`molpriornet.utils` also exposes `featurize_smiles`, which applies the above,
and `featurize`, which takes a DataFrame with a SMILES and a label column.

## Using a pretrained model

### KernelMPN

`KernelMPN` predicts in the units of the support labels, so no scaling is
needed:

```python
from molpriornet import KernelMPN

model = KernelMPN.load_model("checkpoint.pth")
model.eval()

model.fit(support_x, support_y)
predictions = model.predict(query_x)
```

### AttMPN

`AttMPN` is trained on labels standardized per assay, so the support labels
must be normalized before `fit` and the predictions scaled back afterwards:

```python
import numpy as np
from molpriornet import AttMPN

model = AttMPN.load_model("checkpoint.pth")
model.eval()

support_mean = np.mean(support_y)
support_std = np.std(support_y)

model.fit(support_x, (support_y - support_mean) / support_std)
predictions = model.predict(query_x) * support_std + support_mean
```

## Checkpoints

Pretrained checkpoints are published on [Zenodo](https://doi.org/10.5281/zenodo.24375098). All three are
trained on the official FS-Mol training split (see [Getting
FS-Mol](#getting-fs-mol)), using Morgan fingerprints of radius 2 and length 2048.

| Checkpoint | Description |
| ----- | ---------- |
| `att-mpn-16.pth` | 16-layer attention MPN |
| `att-mpn-4.pth` | 4-layer attention MPN |
| `kernel-mpn.pth` | Kernel MPN |

`load_model` reads the architecture — layer count, hidden size and fingerprint
size — from the checkpoint itself, so no configuration is needed to load one.

## Command-line scripts

### Predicting unlabelled molecules

Takes a labelled support CSV and a query CSV that only needs a SMILES column,
and writes the query rows back out with a `prediction` column. Scaling is
handled for you:

```bash
python scripts/predict.py checkpoint.pth support.csv query.csv \
    --smiles-column SMILES --target-column pIC50 \
    --output predictions.csv
```

### Scoring against known labels

Reports MAE and Kendall tau on a held-out query set, optionally alongside a
random forest baseline fitted on the same support set:

```bash
python scripts/evaluate.py checkpoint.pth data.csv \
    --smiles-column SMILES --target-column pIC50 \
    --query-size 0.2 --baselines rf
```

Pass `--query-path` to use a second CSV as the query set as-is instead of
splitting the input randomly.

## Training

Meta-training runs on FS-Mol-style data: a directory of `*.jsonl.gz` files, one
per assay, each line a record with a `SMILES` field and a target field
(`Property`, `RegressionProperty` or `LogRegressionProperty`).

```bash
python scripts/train.py scripts/train_example.yaml
```

See [scripts/train_example.yaml](scripts/train_example.yaml) for the available
options. Each epoch iterates over the assays, splits each into a support and a
query set, standardizes the labels by the support statistics, and takes a
gradient step on the query loss. The checkpoint with the best validation MAE is
written to `output_path`. Featurized assays are cached under `loader_cache/`,
so the first epoch is substantially slower than later ones.

MLflow logging is optional and off by default; set `experiment_name` and
`tracking_uri` in the config to enable it.

### Getting FS-Mol

FS-Mol is described at [microsoft/FS-Mol](https://github.com/microsoft/FS-Mol)
and can be downloaded
[directly](https://figshare.com/ndownloader/files/31345321) as a tar archive.
Unpack it and point `train_dir` and `val_dir` at its `train/` and `valid/`
subdirectories.

Note that the archive's `train/` directory is a superset of the official
training split: it contains 26,868 assays, of which the official split uses
4,938. To train on the official split, pass FS-Mol's own split definition,
[`datasets/fsmol-0.1.json`](https://github.com/microsoft/FS-Mol/blob/main/datasets/fsmol-0.1.json),
together with the partition names:

```yaml
train_dir: "data/fs-mol/train/"
val_dir: "data/fs-mol/valid/"
partition_file: "data/fsmol-0.1.json"
train_partition: "train"
val_partition: "valid"
```

Leaving `partition_file` unset trains on everything in `train_dir` instead —
more than five times as many assays — but the results will not be comparable
with models trained on the official split. In particular, many of the assays
target proteins overlapping with the validation and test assays.

## Reproducing the FS-Mol comparison

The benchmark comparison runs inside
[ADKF-IFT](https://github.com/Wenlin-Chen/ADKF-IFT), which provides the FS-Mol
evaluation harness and the baselines it is compared against.
[adkf-ift.patch](adkf-ift.patch) holds the additions we made to it:

| Added file | Purpose |
| ---------- | ------- |
| `fs_mol/mpn_test.py` | Evaluates `AttMPN` and `KernelMPN` on FS-Mol tasks |
| `fs_mol/tabpfn_test.py`, `fs_mol/tabicl_test.py` | TabPFN and TabICL baselines |
| `fs_mol/runtime_comparison.py` | Fit + predict wall-clock time against support-set size |
| `fs_mol/aggregate_results.py` | Aggregates the per-task CSVs into a summary table |

Beyond those, the patch makes small compatibility fixes across the existing
code so that it runs on current Python.

Apply it to the commit it was generated against:

```bash
git clone https://github.com/Wenlin-Chen/ADKF-IFT.git
cd ADKF-IFT
git checkout b82497e1382e4edd821d1c6132d080035b456884
git apply /path/to/adkf-ift.patch
```

The evaluation scripts need `molpriornet` alongside ADKF-IFT's own
dependencies. With both installed, evaluate a checkpoint against the `test/`
split of an unpacked FS-Mol dataset:

```bash
python fs_mol/mpn_test.py checkpoint.pth data/fs-mol/ --model AttMPN
```

## Citation

```bibtex
@misc{gomoluch2026molecular,
  title         = {Molecular Prior Networks},
  author        = {Gomoluch, Pawel and Bradley, Anthony},
  year          = {2026},
  eprint        = {2611.05103},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {NeurIPS 2026 Workshop on AI for Drug Discovery}
}
```

## License

The code in this repository is MIT — see [LICENSE](LICENSE).

The pretrained checkpoints are released separately, under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

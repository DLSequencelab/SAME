# SAME: Source-Preserving Additive Modeling

A standalone release of the **four-branch model used by the manuscript**, together
with data conversion, training, and final classification evaluation. This is not
the later SparseMoE or FixedFusion architecture.

This package does not include baselines, ablations, synthetic explanation tasks,
perturbation experiments, statistical comparisons, manuscript figures, datasets,
or pretrained checkpoints.

## 1. Model

The input has shape `[batch, time, channels]`. An optional Boolean mask has shape
`[batch, time]`; `True` denotes an observed time position.

1. **Multiscale Representation Encoding (MRE)** constructs four patch-aligned sources:
   - **Local state:** an ordered, flattened patch projection, optional channel
     compression, and a fixed position code.
   - **Short context:** one residual depthwise-separable convolution over adjacent
     patch positions.
   - **Long context:** residual depthwise-separable convolutions with the configured
     dilations over patch positions.
   - **Local spectrum:** masked weighted centering, a Hann window, rFFT power,
     non-DC frequency-band averaging, log compression, and projection.
2. **Branch-Specific Transformations (BST)** apply a separately parameterized
   residual bottleneck to each source. All available branches are processed.
3. A **shared bias-free class mapping**, linear or cosine, produces local scores.
4. **Branch--Temporal Aggregation (BTA)** combines branch-axis weights and
   within-branch temporal weights. The sequence logit is the sum of weighted local
   class scores.

The auxiliary branch loss supplies an additional training signal, with a linearly
annealed coefficient. The prediction computation is additive in **logit space**,
not probability space.

The tensor `signed_contributions` is returned by
`model(x, mask, return_aux=True)`. Its shape is
`[batch, patch, branch, class]`, and summing over patch and branch recovers logits.
No explanation-evaluation or visualization scripts are included.

Some parameter/state-dictionary names (`expert_dim`, `experts`, `route_heads`,
`router_temperature`, and `route_floor`) are retained from the original code for
compatibility. Here, `route_heads` compute **dense branch-composition weights**;
they do not select an expert pool. `router_temperature` is the branch-softmax
temperature, and `route_floor` is its uniform-mixture coefficient.

## 2. Requirements

Python 3.10 or newer and a working PyTorch installation are required.

Reference training environment (reported versions):

| Component | Version |
| --- | --- |
| Python | 3.10.16 |
| PyTorch | 2.5.1+cu121 |
| CUDA (PyTorch build) | 12.1 |
| cuDNN | 9.1.0 |
| NumPy | 1.26.4 |
| sktime | 1.1.0 |
| SciPy | 1.15.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |

The requirements file pins the reported package versions. Python, CUDA, and cuDNN
are environment information, not pip requirements. The CUDA entry identifies the
PyTorch build, not the installed NVIDIA driver or a separately installed toolkit.
Release verification checks are documented separately in Section 7.

```bash
python -m pip install -r requirements.txt
```

If using a CUDA environment, keep or install a PyTorch build matching the target
machine. The training entry point accepts `--device cpu`, `cuda`, `cuda:0`,
or `auto`. The single requirements file includes dependencies for both model
training and UEA `.ts` data conversion.

The core training code only requires NumPy and PyTorch; it does not import
matplotlib, scikit-learn, or sktime. sktime is imported only by the optional
`.ts` conversion command.

## 3. Data

Use one NPZ file per dataset. The runner searches these locations, rejecting
ambiguous matches:

```text
<data_root>/<Dataset>.npz
<data_root>/<Dataset>/<Dataset>.npz
<data_root>/<Dataset>/data.npz
```

Required arrays:

| Array | Shape | Description |
|---|---|---|
| `train_x` | `[N_train, T_train, C]` | Official training inputs |
| `train_y` | `[N_train]` | Training labels |
| `test_x` | `[N_test, T_test, C]` | Official test inputs |
| `test_y` | `[N_test]` | Test labels |

Optional arrays: `train_mask[N_train,T_train]` and
`test_mask[N_test,T_test]`. Without masks, every position is considered valid.
TRAIN and TEST must have the same channel count, but can have different lengths.
Each sample must have at least one valid position; valid observations must be
finite. Invalid padding is set to zero.

Labels are mapped using TRAIN classes only. TEST labels not present in TRAIN are
rejected. No validation subset is created, and TRAIN and TEST are not merged.
The original order of each partition is retained in prediction exports.
Input NPZ files must already represent the intended official partitions; the
runner checks their structure, not their provenance against the UEA archive.

To convert an official pair of `.ts` files:

```bash
python convert_uea_ts.py \
  --data_root /path/to/UEA/Cricket \
  --dataset Cricket \
  --output data/Cricket.npz
```

The converter interpolates missing values within each individual channel, uses
zeros for entirely missing channels, and preserves each partition. If channel
lengths differ within a sample, it uses their common prefix. Padding does not fit
a statistical transform to TEST. The converter refuses to overwrite an existing
output file. If preprocessing was already performed for the existing project,
use those NPZ files directly rather than converting them again.

## 4. Training and final-only TEST evaluation

Each dataset and random seed is trained independently.

1. Use **all official TRAIN samples** for gradient updates.
2. Optimize with AdamW. Cosine learning-rate annealing and linear auxiliary-loss
   annealing use the configured maximum epoch budget.
3. Monitor the **sample-weighted online training classification cross-entropy**
   once per epoch. This is computed from training-mode minibatch forwards, with
   dropout active and parameters updated between minibatches. It is not a
   separate evaluation-mode pass, and is not the total loss containing the
   annealed auxiliary term.
4. Stop after `patience` consecutive epochs without an improvement greater than
   `early_stop_min_delta`, or after `epochs`, whichever comes first.
   Defaults are `epochs=100`, `patience=20`, and
   `early_stop_min_delta=0.0`. If TRAIN CE keeps decreasing, training can reach
   the full epoch budget.
5. Save the **actual last trained model** as `final.pt`, before computing any
   TEST prediction. Do not restore the epoch with minimum TRAIN loss or maximum
   TRAIN accuracy.
6. Run **one complete evaluation pass over official TEST** with that model,
   reporting ACC and macro-F1. There are no epoch-by-epoch TEST metrics and no
   TEST-based stopping or checkpoint selection.

Reading TEST labels for schema/class checks before training does not run model
evaluation. The only TEST model-forward pass is after final checkpoint saving.

Training-loss early stopping is not validation-based model selection. This
release changes the evaluation workflow; existing manuscript numbers are not
newly verified results from this release. Rerun the desired dataset/seed set
before attributing performance numbers to this protocol.

### Preflight only

Run commands from this package directory:

```bash
python train.py \
  --config configs/uea18.json \
  --data_root /path/to/existing/npz_data \
  --datasets Cricket --seeds 42 \
  --dry_run
```

Preflight checks configuration recipes and NPZ keys/labels without training or
writing outputs. Full observation/mask and model validation occurs on a real run.

### One dataset, one seed

```bash
python -u train.py \
  --config configs/uea18.json \
  --data_root /path/to/existing/npz_data \
  --datasets Cricket --seeds 42 \
  --device cuda \
  --output result/cricket_seed42_final_test
```

### All 18 datasets and five seeds

```bash
python -u train.py \
  --config configs/uea18.json \
  --data_root /path/to/existing/npz_data \
  --seeds 42 123 777 2024 3407 \
  --device cuda \
  --output result/uea18_final_test_5seeds
```

Runs are sequential. For the fourth physical GPU on Linux, prefix the command
with `CUDA_VISIBLE_DEVICES=3` and continue using `--device cuda`.

For a short environment check, use `--datasets BasicMotions --seeds 42 --epochs 2`
and a new output directory. This is a smoke test, not a performance experiment.

Optional overrides: `--epochs`, `--patience`, `--early_stop_min_delta`, and
`--num_workers`. Model/training hyperparameters otherwise come from the merged
default and dataset entries in `configs/uea18.json`.

The supplied 18 dataset recipes preserve the original full model's active
hyperparameter values. Inactive legacy sparse-routing, orthogonality,
reliability, ablation, validation-split, and explanation-export options have been
removed. Unsupported fields raise errors rather than being silently ignored.

An existing output directory is never overwritten. If `--output` is omitted,
the runner creates a UTC-timestamped directory under `result/`. Failed runs
retain available outputs and write `failure.json`; automatic retry/resume is
not implemented. Choose a new output directory for a subsequent run.

## 5. Outputs

```text
result/<run>/
  config_source.json
  experiment_manifest.json
  data_manifests/<Dataset>.json
  progress.json
  <Dataset>/seed_<seed>/
    resolved_config.json
    train.log
    history.json
    final.pt
    metrics.json
    test_outputs.npz
  seed_metrics.csv
  dataset_summary.csv
  summary.json
```

- `history.json`: online TRAIN accuracy, total loss, classification CE, branch
  CE, auxiliary coefficient, epoch learning rate, and early-stop state. No
  per-epoch TEST or validation metrics are stored.
- `final.pt`: final weights, model configuration, class mapping, seed, final
  epoch, and stopping metadata. It is saved before TEST inference.
- `metrics.json`: final TEST ACC/macro-F1, stop reason, epoch count, partition
  counts, and the one-evaluation protocol.
- `test_outputs.npz`: logits, mapped labels, predictions, confusion matrix,
  classes, and original TEST sample indices.
- `seed_metrics.csv`: metrics on the 0--1 scale.
- `dataset_summary.csv`: per-dataset seed means and sample standard deviations
  in percent.
- `summary.json`: seed summaries and dataset-equally-weighted averages.
  Standard deviations use `ddof=1`; a single-seed SD is `null`.

Summaries are updated after each run. Check `seed_count` and `progress.json`
before interpreting them as complete five-seed results. Macro-F1 averages over
all class coordinates defined by TRAIN, with zero-denominator classes assigned
F1 = 0.

## 6. Loading a saved model

```python
import torch
from same import SAME, SAMEConfig

checkpoint = torch.load(
    "result/your_run/Cricket/seed_42/final.pt",
    map_location="cpu",
    weights_only=True,
)
model = SAME(SAMEConfig(**checkpoint["config"])).eval()
model.load_state_dict(checkpoint["state_dict"], strict=True)
# x: float tensor [batch, time, channels]
# mask: Boolean tensor [batch, time]
with torch.no_grad():
    logits, auxiliary = model(x, mask, return_aux=True)
    prediction_indices = logits.argmax(-1)
    original_classes = checkpoint["classes"]
```

Only load checkpoints from trusted sources. Full-model weight keys are retained
from the original implementation. Legacy checkpoint configurations may contain
removed fields; this release intentionally does not silently accept those
configurations. Use the new resolved configuration when constructing this model,
and do not load ablated or sparse-routing checkpoints as if they were full SAME.

## 7. Tests and verification scope

```bash
python -m unittest discover -s tests -v
```

Tests cover masks and padding, both class mappings/tokenizer paths, additive
reconstruction, normalized weights, finite gradients, auxiliary annealing,
recipe validation, label mapping, metrics, output protection, maximum-epoch
termination, controlled TRAIN-CE early stopping, final-checkpoint replay, and a
single TEST evaluation after checkpoint saving.

See `verification.json` for the actual release checks. CPU unit/integration and
legacy-model equivalence checks do not establish CUDA correctness, dataset
accuracy, runtime, or replication of historical paper results. Seed and cuDNN
settings improve repeatability but do not promise bitwise reproducibility across
devices or PyTorch versions.

## 8. Package layout

```text
same/
  __init__.py
  model.py
configs/
  uea18.json
tests/
  test_release.py
train.py
convert_uea_ts.py
requirements.txt
README.md
verification.json
```

All source comments, docstrings, messages, configuration descriptions, and this
README are in English. Original project code, data, checkpoints, manuscript
sources, and result directories are left untouched.

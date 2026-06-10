# Staged partial evaluations

These scripts let you inspect **what goes in and what comes out at every stage**
of the EEG-TUH pipeline. Each script is self-contained, prints clear `INPUT` and
`OUTPUT` sections, and can be run on its own.

## The stages

| # | Script | What it inspects | Needs corpus? |
|---|--------|------------------|:---:|
| 1 | `stage1_metadata.py` | `TUHEEGEpilepsy.__init__` → the `descriptions` table (per-file subject/montage/sfreq/duration/n_seizure/epilepsy) and montage definitions | yes |
| 2 | `stage2_windowing.py` | `load_data(window_len_s=...)` → `{split: (X, y, meta)}`; verifies subject-disjoint splits | yes |
| 3 | `stage3_datamodule.py` | `TUHEEGDataModule.setup()` → TensorDatasets, `*_df` metadata, one `(X, y)` batch per dataloader | yes (full corpus) |
| 4 | `stage4_hydra_transform.py` | `HydraTransformer` → feature matrix `F`; dimension formula + seed determinism | no (synthetic) |
| 5 | `stage5_sparse_scaler.py` | `_SparseScaler` → scaled `Fs`; fitted mu/sigma/epsilon, mask effect | no (synthetic) |
| 6 | `stage6_classifier_scoring.py` | `make_pipeline(scaler, clf)` fit → decision scores, window- and subject-level accuracy | no (synthetic) |

## Data flow

```
raw EDF corpus
  └─(1)─> descriptions DataFrame
        └─(2)─> {train/val/test: (X[b,c,t], y, meta)}
              └─(3)─> dataloaders -> (X, y) batches
                    └─(4)─> HYDRA features F[b, n_features]
                          └─(5)─> scaled Fs[b, n_features]
                                └─(6)─> classifier -> window + subject scores
```

Stages 4–6 generate their own synthetic, correctly-shaped inputs, so they run
anywhere. Stages 1–3 need the unpacked TUH EEG Epilepsy corpus under
`data/<version>/` (`00_epilepsy/`, `01_no_epilepsy/`, `DOCS/`); if it is absent
they say so and exit cleanly.

## Running

```bash
# data-free stages (run from code/):
uv run python tests/stage4_hydra_transform.py
uv run python tests/stage5_sparse_scaler.py
uv run python tests/stage6_classifier_scoring.py

# data-dependent stages (where the corpus is present):
uv run python tests/stage1_metadata.py --limit 20
uv run python tests/stage2_windowing.py --subjects-per-class 3 --window-len-s 60
uv run python tests/stage3_datamodule.py --yes-full --window-len-min 1

# toy walk-through of one HYDRA group (C=4, T=6, h=2, k=2), prints every
# intermediate tensor of the gather/sum/conv/argmax steps. Only needs torch.
uv run python tests/toy_hydra_group.py

# rank the most-used and the most class-discriminative HYDRA kernels (global +
# per-class win-count matrices) and plot each kernel's waveform (ms) + frequency
# response (Hz), plus a peak-frequency histogram split by favored class. Variants:
#   --weighting frequency|magnitude  --by max|min|total  --score difference|ratio|logodds
#   --sfreq <Hz>   (sets the frequency axis)
# Only needs torch (+ matplotlib for the plots); writes to tests/outputs/.
uv run python tests/top_kernels.py --top 12 --by max --weighting frequency --score difference
```

Each script takes `--help`.

## Requirements

The scripts exercise the real code, so they need the project runtime. Run
`uv sync` in `code/` first. Note that several packages the code imports are **not
yet declared in `pyproject.toml`** and may need adding before these run:
`aeon`, `braindecode`, `mne`, `mne-icalabel`, `scikit-learn`, `joblib`, `pandas`.
The scripts detect missing imports and print which packages are needed.

These are diagnostic scripts, not pass/fail unit tests; they are named
`stageN_*.py` (not `test_*.py`) so pytest will not collect them.

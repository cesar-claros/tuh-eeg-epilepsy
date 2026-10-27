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
| 7 | `stage7_shapeconv_sae.py` | `ShapeConvSAE.fit_unsupervised` on an independent synthetic train draw (spike-and-wave of random polarity), monitored on a val draw, scored on a test draw; deterministic invariants (dimension, chunk invariance, seed reproducibility of two fits, checkpoint round trip, refusal without provenance) FAIL the run; the gate (`--strict`) is event-level precision and recall > 0.9 with maximum-cardinality matching within `--match-tol`; template `\|xcorr\|` and window AUROCs are diagnostics; stress and lever flags in the docstring | no (synthetic) |
| - | `test_shapeconv_sae.py` | pass/fail unit invariants (pytest or plain python, 25 tests): projection, adjoint identity, analytic shrink response, NMS ties, checkpoint round trip, refusals and migrations, chunk invariance, provenance guards, context crops, residual pool, AR whitening, calibrated thresholds against a brute-force order statistic, unusable rows, split preflight, event matching | no |
| - | `test_waveform_sae.py` | one maximum per atom and window and top-R atoms per window; full-segment encoder and window-restricted decoder, tie rules and gradients, competition only within a window, contextual crop/full-segment response, code, reconstruction, loss and gradient agreement without a full-segment mask, segment edges, random/tiled sampling, full-segment patch initialization, usage and won-window statistics, synthetic fitting and refusal of checkpoint formats 1 to 4 | no |
| - | `test_waveform_full_segments.py` | default full-segment training without an energy gate; direct R=1 per-window atom competition reference for loss and gradient, competition within but not across windows, complete per-epoch coverage, partial batches, unusable-channel masking, validation aliases and refusal of silent tail trimming | no |
| - | `test_waveform_probe.py` | decision rules of the first format-5 probe (`src/waveform_probe.py`): paired reconstruction verdict, starvation thresholds, a next step for every verdict, epoch selection by minimum `val_objective`, incomplete and undertrained probes, on synthetic CSVs | no |
| - | `test_start_transient.py` | recording-start transient measurement (`src/start_transient.py`): per-second rms, line length and clip fraction on usable channels, removal of a uniform window offset, transient end and skip rounding, patient bootstrap, class-confound flag, end-to-end decision and figure on synthetic decays | no |
| - | `test_waveform_kr_sweep.py` | K x R x initialization reconstruction sweep (`src/waveform_kr_sweep.py`): arm names, atom-health thresholds scaled by K, the seed-spread tolerance, the knee per R with starving arms skipped and the saturation flag, the initialization verdict, and an end-to-end summary on synthetic arms | no |
| - | `test_sae_coverage.py` | metadata-only systematic coverage: disjoint scored centers, real context, final short chunks, arbitrary lengths and argument validation | no |
| - | `test_local_competition_sae.py` | local competition against a brute-force reference, conservative ties, support bounds, gradients, tied/untied channel factors and biases, contextual chunks, masked tails, synthetic fit and checkpoint validation | no |
| - | `test_train_sae.py` | shared entry-point dispatch and calibration capability checks with mocked training; Waveform/local-competition training and diagnostic filter presets | no |
| - | `test_temporal_filter_config.py` | high-pass/low-pass nullable cutoffs, validation and forwarding through data configuration | no |
| - | `test_temporal_filtering.py` | context for sequential FIR filters and synthetic whole-signal versus contextual-window filtering | no |
| - | `test_window_resampling.py` | the windowed loader (`WindowDataset._process`) on a synthetic 250 Hz recording against filtering and resampling the whole recording: resampling before the trim and sample-exact trimming, including a last window with a cut margin | no |
| - | `test_patient_waveform_sweep.py` | campaign pairing and resume guards, the fixed sweep configuration including the audit offset, and `check_audit` on synthetic reference and offset audits (changed reference manifests, lost patients, a fresh split, a foreign reference, extra recordings) | no |
| - | `test_native_sfreq_exclusion.py` | native-rate exclusions independent of label and resampling target, explicit-path union, metadata validation, eager/lazy forwarding, fixed CSVs with old path prefixes, split preservation and provenance | no |
| - | `test_datamodule_repair_flags.py` | the datamodule refuses `interpolate_bad_channels`, `drop_bad_segments`, `drop_seizure_segments` and `apply_aas` without lazy loading (the eager loader ignores them), names every refused flag, and passes each flag to the engine under lazy loading | no |
| - | `test_skip_start.py` | recording-start skip: window counts and starts on the grid `skip_start_s + k * stride`, windows inside their recording, earlier windows dropped from a supplied plan, refusal of negative or non-finite values, forwarding to the engine, and the provenance record | no |
| - | `test_screen_local_competition.py` | hyperparameter screen: best-lag atom similarity and matching against direct sums, edge energy and span, plateau epoch, density bound, in-memory window cache, composition of the shipped config and every arm, common-sample scoring against the training forward pass, a three-arm screen on synthetic CPU data with resume and refusals | no |

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

Stages 4–7 generate their own synthetic, correctly-shaped inputs and do not need
the corpus (stage 7 replaces stage 4 when `feature=shapeconv_sae`). Run all tests
on the HPC, following the project's execution policy. Stages 1–3 need the unpacked TUH EEG Epilepsy corpus under
`data/<version>/` (`00_epilepsy/`, `01_no_epilepsy/`, `DOCS/`); if it is absent
they say so and exit cleanly.

## Running

Run the waveform full-segment training, preprocessing, selection and legacy crop regressions in the HPC container:

```bash
python -m pytest -q tests/test_waveform_sae.py tests/test_waveform_full_segments.py tests/test_train_sae.py \
  tests/test_temporal_filter_config.py tests/test_temporal_filtering.py tests/test_native_sfreq_exclusion.py
```

Run the new local-competition tests from `code/` in the HPC container:

```bash
python -m pytest -q tests/test_sae_coverage.py tests/test_local_competition_sae.py tests/test_train_sae.py \
  tests/test_temporal_filter_config.py tests/test_temporal_filtering.py tests/test_screen_local_competition.py \
  tests/test_datamodule_repair_flags.py tests/test_skip_start.py
```

These tests have been added but have not yet been executed on the HPC. They do
not read EEG files. Test tooling is declared in the `dev` dependency group in
`container/pyproject.toml`; the default container build installs runtime
dependencies only.

```bash
# data-free stages (run from code/):
uv run python tests/stage4_hydra_transform.py
uv run python tests/stage5_sparse_scaler.py
uv run python tests/stage6_classifier_scoring.py
uv run python tests/test_shapeconv_sae.py
uv run python tests/test_waveform_sae.py
uv run python tests/stage7_shapeconv_sae.py --n-windows 64 --epochs 20 --strict
uv run python tests/stage7_shapeconv_sae.py --mode topk --amp-min 3

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

The scripts exercise the real code, so they need the HPC project container.
`container/pyproject.toml` is the runtime dependency source of truth; the template
manifest in `code/` is stale. For a writable test environment that inherits the
container's packages, create a virtual environment with
`uv venv --system-site-packages <test-env>`, then install the declared tools with
`uv pip install --python <test-env>/bin/python --group <container-dir>/pyproject.toml:dev`.
Run pytest using `<test-env>/bin/python -m pytest` with the paths above.

The `stageN_*.py` scripts are diagnostic (stage 7 also fails on broken
invariants); the `test_*.py` files are pass/fail tests that pytest collects.

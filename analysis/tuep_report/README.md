# Post-hoc reads of the TUEP measurement report

These scripts produce the post-hoc numbers and Figure 1 of `documentation/tuep_measurement_report.md`. They read result files only, fit no model and need no corpus, so they run locally with numpy (and matplotlib for the figure). They are a record of descriptive reads, and they decide nothing.

| Script | Arguments | What it gives |
|---|---|---|
| `read_frozen_scores.py` | readout root | Paired differences and bag-size strata (frozen readout, section 13.4) |
| `read_bands.py` | readout root | Band power by class and dictionary near-copies (section 13.4) |
| `read_acquisition.py` | readout root, corpus listing directory | Session year and montage against labels and scores (section 13.4) |
| `read_age_gate.py` | audit root | Why the age gate failed, with no class (audit, section 15.2) |
| `read_drug_cells.py` | audit root, readout root | Scores by class and wide-list status (section 15.6) |
| `read_narrow_cells.py` | audit root, readout root | Scores by class and maintenance-drug status, with bag matching (section 16.8) |
| `read_report_numbers.py` | audit root, readout root, output json | Drug baseline, EEG added to the drug, cell means |
| `read_encoder_matched.py` | code folder, check root, audit root, readout root | Age-matched read of the encoder check (its protocol, section 13) |
| `make_report_figure.py` | numbers json, output png | Figure 1 |
| `audit_power.py`, `drug_power.py` | none | Rough power by simulation, with no project data (sections 11 and 16.5) |

Run each with `uv run --no-project --with numpy python -I <script> <arguments>`, adding `--with matplotlib` for the figure.

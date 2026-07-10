# Preprocessing

This folder contains the preprocessing pipeline used to convert raw acquisition/behavior data into the traces and metadata consumed by the analysis scripts in `../analysis`. Sections are numbered by pipeline stage; each describes the relevant files and their role.

## 0. Directory Structure & Experiment Metadata

Raw data is organized under a single base directory, separating single-color functional (GCaMP) data from 2-color structural (mCherry) reference data, with folder names free of whitespace. Each session subfolder contains the raw Bruker `ome.tiff` series and its associated XML metadata files.

```
Base_directory/
├── GCaMP_raw/
│   ├── Animal1/
│   │   ├── Session1-Gcamp-TSeries/
│   │   └── Session2-Gcamp-TSeries/
│   └── Animal2/
│       ├── Session1-Gcamp-TSeries/
│       └── Session2-Gcamp-TSeries/
├── mCherry_raw/
│   ├── Animal1/
│   │   ├── Session1-mCherry-TSeries/
│   │   └── Session2-mCherry-TSeries/
│   └── Animal2/
│       ├── Session1-mCherry-TSeries/
│       └── Session2-mCherry-TSeries/
└── Experiment_metadata_info.csv
```

`Experiment_metadata_info.csv` (the "file key") is read by most preprocessing and analysis scripts to look up file paths and associate mCherry structural data with the corresponding GCaMP functional session. For animals with multiple sessions, session names must sort in the intended chronological order alphabetically (e.g. `[Baseline, Post]` or `[0_FC, 1_recall]`) — an ordering like `[FC, EXT1, EXT2]` would sort incorrectly.

**Format:** one row per animal/FOV/session, with columns:

| Column | Description |
|---|---|
| `Animal` | Animal identifier |
| `Group` | Experimental group/condition |
| `Session` | Session name (e.g. `Baseline`, `Recall1`) |
| `FOV` | Field-of-view identifier |
| `TSeries_g` | GCaMP TSeries folder name for this animal/FOV/session |
| `TSeries_mch` | mCherry TSeries folder name for this animal/FOV/session |

## 1. Metadata & Raw Image Preprocessing

**`fiji_macros/preprocess_for_fiji.py`** — Moves XML, `.env`, and References metadata files out of each TSeries folder into a `Ref` subfolder so Fiji doesn't slow down loading the raw images.

**`fiji_macros/Pre process caiman single.ijm`** — Resizes/chunks a single raw GCaMP `ome.tif` TSeries into smaller multipage TIFFs for CaImAn.

**`fiji_macros/Pre process caiman batch.ijm`** — Batch version of the above, processing multiple GCaMP TSeries folders in one run.

**`fiji_macros/Save2ColorTifs.ijm`** — Splits a raw 2-color mCherry TSeries into separate Ch1/Ch2 TIFF stacks.

## 2. Motion Correction

**`motion_correction/motion_correction_functions.py`** — Core CaImAn (NoRMCorre) motion correction functions, including `motion_correct` (single-color GCaMP rigid/piecewise-rigid registration) and `motion_correct_2color_single` (aligns a 2-color mCherry TSeries by registering one channel and applying the resulting shifts to the other), plus helpers for locating TSeries and loading saved templates/parameters.

**`motion_correction/motion_correct.py`** — Interactive entry point for running single-color GCaMP motion correction on a chosen TSeries or set of sessions.

**`motion_correction/motion_correct_job.py`** — Cluster batch-job version of GCaMP motion correction, run from the command line per animal/session/FOV, with support for multiple parameter sets or repeated rounds of correction.

**`motion_correction/motion_correct_2color_single.ipynb`** — Interactive notebook for two-color (GCaMP/mCherry) motion correction and channel alignment on a single session, used for troubleshooting.

**`visualization/play_movie.py`** — Interactively plays back a raw or motion-corrected movie (single-color GCaMP or two-color GCaMP/mCherry) for a user-selected TSeries, for visually checking motion correction quality.

Motion corrected outputs are saved to `BaseDirectory/GCaMP/Animal_subfolders/caiman_output/motion_correct/`.

## 3. mCherry Registration

**`mCherry/mCherry_registration.py`** — Registers the 2-color mCherry TSeries to the GCaMP motion-correction template by aligning the green channel to it, then applies the resulting shifts to the red (mCherry) channel; supports single-session or batch runs across animals.

**`mCherry/mCherry_registration_Ch1.py`** — Fallback registration used when green-channel alignment fails; aligns the red channel directly to a reference template instead.

**`mCherry/mcherry_functions.py`** — Loads registered mCherry images and computes per-ROI mCherry intensity for downstream cell classification.

## 4. CNMF (Source Extraction)

**`CNMF/CNMF.py`** — Runs CaImAn's CNMF pipeline on motion-corrected data to extract spatial and temporal components, evaluate them, and compute dF/F; supports single-session, per-animal, and batch (multi-parameter) runs, with optional seeding from Cellpose masks.

**`CNMF/CNMF_from_tif.py`** — Variant of `CNMF.py` that runs directly from raw TIFFs instead of requiring motion-correction `.pkl` output.

**`CNMF/CNMF_single.ipynb`** — Interactive, cell-by-cell notebook version of `CNMF.py` for running/inspecting CNMF on a single session.

## 5. Calcium Trace Deconvolution

All dF/F traces underwent calcium deconvolution to infer discrete event times, using an [L0 penalty method](https://jewellsean.github.io/fast-spike-deconvolution/) (Jewell, Hocking, Fearnhead, and Witten, 2020; *Biostatistics* 21, 709–726, [https://doi.org/10.1093/biostatistics/kxy083](https://doi.org/10.1093/biostatistics/kxy083)). Each cell's dF/F trace is modeled as a noisy readout of underlying calcium fluorescence dynamics, with two parameters fit to each trace: gamma, determined by the indicator half-life (0.14 s for Thy1-GCaMP6f GP.5.17), and lambda, which controls the strictness of the L0 penalty and therefore the sparseness of detected events. Lambda was estimated per cell with the minimum event size set to 5 standard deviations of each cell's dF/F noise. Mean event rates were calculated as the weighted sum of deconvolved event magnitudes over the session, divided by total session time, so that larger-amplitude transients contributed proportionally more than smaller ones.

**`deconvolution/get_dff.py`** — Extracts dF/F traces from CNMF `.hdf5` results and writes them to `.pkl` files for deconvolution.

**`deconvolution/L0_analysis.py`** — Class (adapted from S. Jewell's L0 spike-inference repo) that automatically selects an L0 penalty (lambda) per cell so the smallest detected event meets a minimum size threshold in units of noise std.

**`deconvolution/L0_deconv.py`** — Runs L0 event detection on dF/F traces using `L0_analysis` to fit lambda per cell, then extracts event times/magnitudes and estimated calcium traces via `FastLZeroSpikeInference`.

**`deconvolution/FastLZeroSpikeInference/`** — Vendored Python bindings (adapted from S. Jewell's repo) to the compiled L0 spike-inference library used by `L0_analysis.py`/`L0_deconv.py`.

**`deconvolution/plot_deconv.ipynb`** — Loads deconvolution results for a session and plots traces/events for visual quality control.

**`deconvolution/SpikeInference.yml`** — Conda environment file for running L0 deconvolution (e.g. on a compute cluster).

Deconvolution output `.pkl` files are saved to `Base_directory/deconvolution/deconvolution_results/`.

## 7. Multi-Day Registration

Registering the same cells across sessions/days requires [CellReg](https://github.com/zivlab/CellReg) (MATLAB), run separately from the Python steps below.

**`cell_registration/get_footprints_CellReg.py`** — Reshapes CaImAn's sparse spatial footprints (`A`) into the N x Y x X format CellReg expects, and saves them as `.mat` files per animal/FOV for input to CellReg.

Run CellReg in MATLAB on the saved `.mat` footprints (`>>> CellReg`) to register cells across sessions.

**`cell_registration/generate_regind_csv.py`** — After running CellReg in MATLAB, builds a `reg_indices.csv` lookup table of matched cell indices across sessions for each animal/FOV, saved in the `/Tagging/CellReg_output` directory. 

**`cell_registration/summary_images.py`** — Saves mean/correlation summary images for each session into the CellReg output directory.

**Evaluating the registration:**

**`cell_registration/plot_cellreg.ipynb`** — Loads the most recent CellReg output for an animal/FOV and visualizes registration performance (similar to Supplementary Figure 2A).

**`cell_registration/save_cellpair_plots.py`** — Saves HTML plots of ROI contours (single-session, common-across-all-sessions, and pairwise) for manual visual inspection of registration quality.

## 9. mCherry Cell Classification

**`mCherry/save_mch_thr_plots.py`** — Run twice: first with `plot_thr=False` to plot a histogram of per-ROI mCherry intensities, then with `plot_thr=True` to plot ROIs split into high/low populations across a range of candidate intensity thresholds. After inspecting the plots, the chosen intensity threshold for that animal should be added to `/Tagging/mcherry_thresholds.csv`.

**`mCherry/split_mcherry.ipynb`** — Applies the chosen threshold to classify tagged vs. non-tagged cells, with interactive prompts for manually correcting false positives/negatives by ROI. Also exports ROI's to Fiji importable files to allow user to visualize over mCherry images.

After mCherry classification, the final data table of cell indices (`indices_split.csv`) should be saved to the `/Tagging/` subdirectory.

## 10. Behavior: Wheel Position & Velocity

**`behavior/running.py`** — Parses raw running-wheel encoder data and converts it into velocity, rest/run state, and percent-time-behavior metrics aligned to imaging frames. Raw wheel position is loaded and cleaned, aligned to imaging frame times using the TSeries XML metadata, converted to velocity via a sliding window, thresholded to remove noise, and then segmented into discrete rest/run epochs and summary percent-time-behavior statistics per animal/FOV/session.

**`behavior/timestamps.py`** — Computes and loads imaging frame times and, for tone/trace fear conditioning sessions, CS onset/offset times: reads the voltage TTL recording and XML acquisition metadata for a TSeries, thresholds the voltage trace to detect tone events, and saves/reloads frame and tone timestamps as `.npz` files.

**`behavior/save_timestamps.py`** — Entry-point script that calls `write_timestamps` (from `timestamps.py`) to compute and save timestamps for a given animal/session.

---
*This README was generated with the assistance of Claude (Anthropic).*

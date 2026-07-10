# Analysis

This folder contains the analysis scripts and notebooks used to generate the quantitative results reported in the manuscript. Each entry below lists the relevant script/notebook, the figure(s) it supports, and a brief summary of the analysis.

## Utilities

Shared support modules, located in `../preprocessing/utilities/`, used throughout the analysis scripts below.

**`utilities/animal.py`** — Defines the `animal` class, the primary interface for loading one animal's preprocessed data: per-session traces (computing and caching them from deconvolution output if not already saved), cell registration/tagging indices, tone/CS timestamps, position/velocity, and CNMF spatial components.

**`utilities/traces.py`** — Trace-level functions used across the analysis scripts: event rate calculation (`event_rate`, `weight_event_rate`), trace binning (`bin_traces`, `bin_traces_overlap`, `bin_traces_time`, `bin_overlap_time`), Z-scoring (`zscore_traces`), and deconvolution trace/plotting helpers (`ev_trace`, `get_traces`, `plot_deconv`, `ridge_plot`).

**`utilities/plotting.py`** — Plotting and spatial helper functions, including `get_contours` (footprint contour extraction, used by `animal.load_footprints`), `com` (center-of-mass calculation), and group-level summary plotting functions.

## mCherry

**`mcherry/local_background_.py`** — Figure 1D

Quantifies mCherry intensity relative to each cell's local background by scaling each cell's ROI 1.5x around its center of mass, subtracting the original ROI from this expanded ROI to isolate a surrounding background region, and dividing mean soma intensity by the mean background intensity to yield the Relative Fluorescence metric.

**`mcherry/percent_leaky.py`** — Supplementary Figure 1E

Computes the percentage of "leaky" mCherry-positive cells (flagged by `remove_bad_cells`) among all cells and among cells registered across sessions, and plots the leaky vs. non-leaky proportions as pie charts.

## Event Rate

**`event_rate/get_rates_all.py`** — Figures 2-3

Computes per-cell weighted calcium event rates (L0-penalized deconvolution of dF/F traces, with events summed by magnitude and normalized by session time) for Day 0 and Day 4 sessions, organized by tagged/non-tagged population and by whether cells were registered across sessions or active in only one.

**`event_rate/combine_cohorts_rates.ipynb`** — Figures 2-3

Merges per-cohort weighted event rate CSVs (output of `get_rates_all.py`) across cohorts into combined rest, run, and total event rate tables.

## Rastermap & Sequence Analysis

**`rastermap/run_rastermap.py`** — Figure 3D

To identify sequential dynamics in the deconvolved activity, binarized event traces are given to [Rastermap](https://github.com/MouseLand/rastermap) (`n_clusters=None, nPCs=10, time_lag_window=60, locality=0.1, time_bin=15`, similar to recommended parameters for single-cell sorting in hippocampal navigation data), which sorts cells so that sequentially active cells are grouped together; the sorted similarity matrix and sorted dF/F/event rasters are saved per animal.

## Correlation

**`correlation/correlation_binned_rest_spont_equalTime.py`** — Figure 4

Computes pairwise Spearman correlations between overlapping-binned traces (4-125 frame bins) of engram (tagged) and non-engram (non-tagged) cells co-active on Day 0 and Day 4, restricted to rest epochs of matched duration, to yield engram/engram, engram/non-engram, and non-engram/non-engram correlation distributions per animal.

**`correlation/Correlation.py`** — Figure 4, Figure 5F(supporting module)

Defines the `correlation` class and `corr_matrix` function used by `correlation_binned_rest_spont_equalTime.py` to compute pairwise and cross-correlation matrices (Spearman, Kendall, or Jaccard) between cell traces, along with helpers for extracting non-redundant correlation values and plotting sorted correlation heatmaps.

**`correlation/D0_D4_Riemannian_Distance_CFC_rest_matched.py`** — Figure 4B

Compares network topology between Day 0 and Day 4 for engram and non-engram populations using a log-Euclidean Riemannian (geometric) distance between unbinned, rest-frame-matched Spearman correlation matrices, normalized by population size.

**`correlation/bin_traces_cfc.py`** — Figure 4A-C (supporting script)

Splits each CFC animal/FOV's Baseline and Post event traces into tagged/non-tagged cell groups and bins them (non-overlapping and 50%-overlapping, across a range of bin sizes), saving the binned traces to `.npz` files consumed by `correlation_binned_rest_spont_equalTime.py`.

**`correlation/bin_traces_tfc_recall1.py`** — Figure 5F-G (supporting script)

Same binning procedure as `bin_traces_cfc.py`, applied to the Recall1 session's event traces for TFC animals, also binning the frame timestamps; output feeds the trace fear conditioning correlation scripts.

## Trace Fear Conditioning

**`tone/pairwise_corr_cstrace_behmatch_concat.py`** — Figure 5F

Per animal, builds a running-composition-matched baseline epoch (sampled to match the run/rest makeup of the CS+/trace period) and computes pairwise Spearman correlations between engram and non-engram cells for both the baseline and CS+/trace epochs, saving the result for downstream analysis.

**`tone/pairwise_corr_utils.py`** — Figure 5F (supporting module)

Utility functions used by `pairwise_corr_cstrace_behmatch_concat.py`: epoch/mask construction from time windows, equal-time frame sampling for behavior matching, the core pairwise correlation computation (`pairwise_corr_epochs`), and a diagnostic plot of epoch selection over the velocity trace.

**`tone/plot_corr_cstrace_behmatch_concat.py`** — Figure 5F

For each behavior-matched cell pair (p<0.05), computes the change in Spearman correlation from pre-CS+ to concatenated CS+/trace-period activity, z-scores this difference across all pairwise comparisons within each animal, and plots the absolute z-scored change by pair type (engram/non-engram) across binned activity time scales.

**`tone/tone_rates_recall1.ipynb`** — Figures 5D-E, Supplementary Figure 8B

Calculates deconvolved event rates for engram/non-engram cells during the Recall1 session across three windows — the whole session, the pre-CS baseline period (Supplementary Figure 8B), and each individual 20s CS1 trial — and compares baseline vs. CS1-trial-averaged rates (Figure 5D) and z-scores per-trial rates to the baseline period (Figure 5E), with linear mixed-effects models comparing engram vs. non-engram rates.

**`tone/D0_D4_Riemanninan_Distance.ipynb`** — Figure 5G

Computes log-Euclidean Riemannian distance between engram and non-engram Spearman correlation matrices to quantify network topology change from Day 0 to Day 4 (Recall1), using rest-matched frames, CS-matched (running-composition-matched) frames, and a within-Recall1 baseline-vs-CS comparison.

---
*This README was generated with the assistance of Claude (Anthropic).*

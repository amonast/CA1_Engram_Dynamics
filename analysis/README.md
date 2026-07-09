# Analysis

This folder contains the analysis scripts and notebooks used to generate the quantitative results reported in the manuscript. Each entry below lists the relevant script/notebook, the figure(s) it supports, and a brief summary of the analysis.

## mCherry

**`mcherry/local_background_.py`** — Figure 1D

Quantifies mCherry intensity relative to each cell's local background by scaling each cell's ROI 1.5x around its center of mass, subtracting the original ROI from this expanded ROI to isolate a surrounding background region, and dividing mean soma intensity by the mean background intensity to yield the Relative Fluorescence metric.

## Event Rate

**`event_rate/get_rates_all.py`** — Figures 2-3

Computes per-cell weighted calcium event rates (L0-penalized deconvolution of dF/F traces, with events summed by magnitude and normalized by session time) for Day 0 and Day 4 sessions, organized by tagged/non-tagged population and by whether cells were registered across sessions or active in only one.

**`event_rate/combine_cohorts_rates.ipynb`** — Figures 2-3

Merges per-cohort weighted event rate CSVs (output of `get_rates_all.py`) across cohorts into combined rest, run, and total event rate tables.

## Correlation

**`correlation/correlation_binned_rest_spont_equalTime.py`** — Figure 4A-C

Computes pairwise Spearman correlations between overlapping-binned traces (4-125 frame bins) of engram (tagged) and non-engram (non-tagged) cells co-active on Day 0 and Day 4, restricted to rest epochs of matched duration, to yield engram/engram, engram/non-engram, and non-engram/non-engram correlation distributions per animal.

**`correlation/Correlation.py`** — Figure 4A-C (supporting module)

Defines the `correlation` class and `corr_matrix` function used by `correlation_binned_rest_spont_equalTime.py` to compute pairwise and cross-correlation matrices (Spearman, Kendall, or Jaccard) between cell traces, along with helpers for extracting non-redundant correlation values and plotting sorted correlation heatmaps.

**`correlation/D0_D4_Riemannian_Distance_CFC_rest_matched.py`** — Figure 4B

Compares network topology between Day 0 and Day 4 for engram and non-engram populations using a log-Euclidean Riemannian (geometric) distance between unbinned, rest-frame-matched Spearman correlation matrices, normalized by population size.

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

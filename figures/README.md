# Figures

This folder contains the Python scripts and notebooks used to generate each manuscript figure panel from the outputs of `../analysis` and `../preprocessing`. 

## Figure 1

**`Fig1_representative_rois.py`** — **Figure 1** (representative ROIs)

Exports representative ROI contours to be displayed in FIJI over the imaging FOV.

**`Fig1D_mcherry_localbg.ipynb`** — **Figure 1D**

Plots the mCherry Relative Fluorescence metric computed by [`analysis/mcherry/local_background_.py`](../analysis/mcherry/local_background_.py).

**`Fig1E_reptraces.ipynb`** — **Figure 1E**

Plots representative dF/F traces for example tagged/non-tagged cells.

**`Fig1F_proportion_cells.ipynb`** — **Figure 1F**

Computes and plots the proportion of tagged/reactivated cells per animal (registered vs. session-unique, engram vs. non-engram), using cell registration/tagging indices directly (`remove_bad_cells`).

## Figure 2

**`Fig2A_decilefootprints.ipynb`** — **Figure 2A**

Plots cell footprints sorted/colored by event-rate decile, using per-animal decile assignment (`calculate_deciles_ani`) on top of [`analysis/event_rate/get_rates_all.py`](../analysis/event_rate/get_rates_all.py) output.

**`Fig2B-C.ipynb`** — **Figure 2B-C**

Plots event rate by decile, using the combined rates table from [`analysis/event_rate/combine_cohorts_rates.ipynb`](../analysis/event_rate/combine_cohorts_rates.ipynb).

**`Fig2D-F_decile_v_ptag-baseline.ipynb`** — **Figure 2D-F**

Plots baseline event-rate decile vs. proportion tagged, using the same combined rates table.

**`Fig2G_deciles_rest&run.ipynb`** — **Figure 2G**

Plots event-rate deciles split by rest vs. run epochs, using the rest/run rate tables from `combine_cohorts_rates.ipynb`.

**`Fig2H-I_animalmean_RestvRun.ipynb`** — **Figure 2H-I**

Plots animal-averaged rest vs. run event rates by decile.

## Figure 3

**`Fig3_Rates.ipynb`** — **Figure 3A-C**

Plots event rate distributions/comparisons (histograms, violin/point plots) using the combined rates tables from [`analysis/event_rate/`](../analysis/event_rate/).

**`Fig3D_rastermap.ipynb`** — **Figure 3D**

Plots Rastermap-sorted activity (dF/F and event rasters) using the sort order computed by [`analysis/rastermap/run_rastermap.py`](../analysis/rastermap/run_rastermap.py).

**`Fig3E_prop_sequences.py`** — **Figure 3E**

Identifies and plots the proportion of cells participating in Rastermap-sorted sequences.

**`Fig3F_velocity_correlation.ipynb`** — **Figure 3F**

Plots the Pearson correlation between population activity (engram/non-engram) and velocity.

## Figure 4

**`Fig4A_plot_network.ipynb`** — **Figure 4A**

Plots an example cell-pair correlation network graph (`networkx`) from a saved pairwise correlation matrix.

**`Fig4B.ipynb`** — **Figure 4B**

Plots Day 0 vs. Day 4  Riemannian distance using the output of [`analysis/correlation/D0_D4_Riemannian_Distance_CFC_rest_matched.py`](../analysis/correlation/D0_D4_Riemannian_Distance_CFC_rest_matched.py).

**`Fig4C-D.ipynb`** — **Figure 4C-D**

Plots pairwise correlation changes and p-value-thresholded correlation ratios across bin sizes, using the pairwise correlation dataframes from [`analysis/correlation/correlation_binned_rest_spont_equalTime.py`](../analysis/correlation/correlation_binned_rest_spont_equalTime.py) (binned via [`bin_traces_cfc.py`](../analysis/correlation/bin_traces_cfc.py)).

## Figure 5

**`Fig5B-C.ipynb`** — **Figure 5B-C**

Plots trial-aligned (peri-CS+) velocity.

**`Fig5D.ipynb`** — **Figure 5D**

Plots baseline vs. CS+-trial-averaged event rates, using output from [`analysis/tone/tone_rates_recall1.ipynb`](../analysis/tone/tone_rates_recall1.ipynb).

**`Fig5E.ipynb`** — **Figure 5E**

Plots event rates z-scored to baseline across CS+ trials, using output from `tone_rates_recall1.ipynb`.

**`Fig5F.ipynb`** — **Figure 5F**

Plots the z-scored change in pairwise correlation from baseline to CS+/trace period, using output from [`analysis/tone/pairwise_corr_cstrace_behmatch_concat.py`](../analysis/tone/pairwise_corr_cstrace_behmatch_concat.py) / [`plot_corr_cstrace_behmatch_concat.py`](../analysis/tone/plot_corr_cstrace_behmatch_concat.py).

**`Fig5G.ipynb`** — **Figure 5G**

Plots Day 0 vs. Day 4 (Recall1) network topology (Riemannian distance), using output from [`analysis/tone/D0_D4_Riemanninan_Distance.ipynb`](../analysis/tone/D0_D4_Riemanninan_Distance.ipynb).

## Supplementary Figures

**`SuppFig1.ipynb`**, **`SuppFig1_cellreg_suppl_descriptive_stats.ipynb`** — **Supplementary Figure 1**

Descriptive statistics and plots for cell registration (CellReg) quality/yield.

**`SuppFig1E_percent_leaky.py`** — **Supplementary Figure 1E**

Plots the percentage of "leaky" mCherry-positive cells computed by [`analysis/mcherry/percent_leaky.py`](../analysis/mcherry/percent_leaky.py).

**`SuppFig2_cellreg_suppl_plots.ipynb`**, **`SuppFig2D_deconvolution_suppl_plots.ipynb`** — **Supplementary Figure 2**

Supplementary CellReg registration plots and deconvolution quality-control plots.

**`SuppFig4A_ bayes_equations.ipynb`** — **Supplementary Figure 4A**

Documents the Bayesian model equations (no data processing).

**`SuppFig4B-D.ipynb`** — **Supplementary Figure 4B-D**

Plots Bayesian model outputs/diagnostics using the combined event-rate tables from [`analysis/event_rate/`](../analysis/event_rate/).

**`SuppFig5B-C.ipynb`** — **Supplementary Figure 5B-C**

Plots rest/run event-rate comparisons using the combined rest/run rate tables.

**`SuppFig6A-B.ipynb`**, **`SuppFig6D_corr_animals_tot.ipynb`**, **`SuppFig6E_corr_animals_rest_equal.ipynb`**, **`SuppFig6E_corr_animals_run_equal.ipynb`**, **`SuppFig6F-G.ipynb`** — **Supplementary Figure 6**

Supplementary pairwise correlation plots (total, rest-equal, run-equal time) using output from [`analysis/correlation/correlation_binned_rest_spont_equalTime.py`](../analysis/correlation/correlation_binned_rest_spont_equalTime.py).

**`SuppFig6C_Fig4C_plot_corr_matrix_example.ipynb`** — **Supplementary Figure 6C**

Plots an example correlation matrix (supplementary to Figure 4C).

**`SuppFig7A_tone_running.ipynb`** — **Supplementary Figure 7A**

Analyzes running behavior around tone presentation, using [`../preprocessing/behavior/running.py`](../preprocessing/behavior/running.py) and [`../preprocessing/behavior/timestamps.py`](../preprocessing/behavior/timestamps.py) directly.

**`SuppFig8.ipynb`** — **Supplementary Figure 8**

Plots velocity and baseline/tone event-rate comparisons using output from [`analysis/tone/tone_rates_recall1.ipynb`](../analysis/tone/tone_rates_recall1.ipynb).

**Supplementary Figure 9** - Model simulation and figures generated by code in [`github.com/lienkaemper/engram`](https://github.com/lienkaemper/engram)  
9B - `sim_low_inhib.py`, `theory_low_inhib.py`, `plot_low_inhib.py`.  
9D - `sim_vary_inhib.py`, `theory_vary_inhib.py`, `plot_correlation_ratio.py`.  
9F-G - `disinhibition.py`   

---
*This README was generated with the assistance of Claude (Anthropic).*

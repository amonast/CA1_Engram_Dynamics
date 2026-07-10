# CA1 Engram Dynamics

Code accompanying the manuscript [https://doi.org/10.1101/2024.04.16.589790](https://doi.org/10.1101/2024.04.16.589790)).

This repository contains the preprocessing pipeline, analysis code, and figure-generation notebooks/scripts used to produce the results in the paper. Processed imaging data (cell traces) and behavioral (freezing/velocity) can be found at : [].

## Repository Structure

- **[`preprocessing/`](preprocessing/README.md)** — Pipeline for converting raw two-photon acquisition and behavioral data into registered cell traces: raw image preprocessing, motion correction, mCherry registration, CNMF source extraction, calcium event deconvolution, multi-day cell registration (CellReg), mCherry-based engram cell classification, and wheel/behavior processing. See the [preprocessing README](preprocessing/README.md) for the full numbered pipeline and file-by-file descriptions.

- **[`analysis/`](analysis/README.md)** — Analysis scripts and notebooks that compute the quantitative results reported in the manuscript (event rates, pairwise correlations, Riemannian distance, trace fear conditioning analyses, mCherry intensity), organized by analysis type and mapped to the figures they support. See the [analysis README](analysis/README.md) for details.

- **`figures/`** — Python notebooks and scripts used to generate manuscript figures from the outputs of `analysis/` and `preprocessing/`.

## Pipeline Overview

1. Organize raw data and experiment metadata (see [preprocessing/README.md](preprocessing/README.md), Step 0)
2. Run the preprocessing pipeline (Steps 1-10) to produce registered, deconvolved cell traces and behavior data
3. Run analysis scripts in `analysis/` on the preprocessed data to compute figure-level results
4. Run the corresponding notebook/script in `figures/` to generate each manuscript figure

## Requirements

Key dependencies used across the pipeline include [CaImAn](https://github.com/flatironinstitute/CaImAn) (motion correction, CNMF), [CellReg](https://github.com/zivlab/CellReg) (MATLAB, multi-day cell registration), and an [L0 spike-inference](https://jewellsean.github.io/fast-spike-deconvolution/) method for calcium event deconvolution. See `preprocessing/deconvolution/SpikeInference.yml` for the deconvolution environment.

---
*This README was generated with the assistance of Claude (Anthropic).*

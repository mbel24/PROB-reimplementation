# PROB-reimplementation
This repository contains a reimplementation of PROB (Progression-Based Bayesian Method) applied to Alzheimer’s disease (AD) transcriptomic data. The pipeline infers latent disease progression and gene regulatory networks from cross-sectional data, enabling reproducible analyses of AD progression using human brain microarray datasets.

# Backgroud
Alzheimer’s disease (AD) is a progressive neurodegenerative disorder characterized by molecular and cellular changes that accumulate over time.
PROB addresses this challenge by:
1. Inferring a pseudotemporal ordering of samples (progression pseudotime distance, PPD) using a diffusion-based random walk framework.
2. Inferring a directed gene regulatory network via ordinary differential equation models with Bayesian Lasso regression to enforce sparsity.

Key findings from this reimplementation:
- PPD strongly correlates with Braak stage (Spearman ρ = 0.93), demonstrating that transcriptomic data alone can reconstruct disease chronology.
- Regulatory network analysis highlights microglial activation and lipid metabolism as central drivers.
- Hub genes such as CLU and APOE influence immune-related genes like TREM2, TYROBP, CD33, consistent with known AD pathology.

This repository contains all code, scripts, and processed data required to reproduce the analyses and figures in the study. All results in the report are reproducible using the provided pipeline

# Usage
## Option 1: Run the pipeline via Python scripts
    python scripts/run_pipeline.py
This script executes the pipeline end-to-end:
1. Load and preprocess data.
2. Compute progression pseudotime (PPD) for all samples.
3. Infer the gene regulatory network using Bayesian Lasso regression.
4. Generate summary statistics and figures.

## Option 2: Explore the analysis via the notebooks
Open Jupyter notebooks for interactive exploration:
    jupyter notebook
- Notebooks in notebooks/ reproduce all figures and analyses.
- Run cells sequentially to ensure reproducibility
- Before running make sure you download all phyton files in prob and the ExampleData


# Pipeline Workflow
1. Data Loading and Preprocessing (prob/preprocessing.py)
   - Normalization
   - Filtering
   - Formatting for downstream analysis
2. Pseudotemporal Ordering (prob/progression.py)
    - Diffusion-based random walk
    - Compute progression pseudotime distances (PPD)
    - Evaluate correlation with Braak stage
3. Gene Regulatory Network Inference (prob/ode_bayesian_lasso.py)
    - ODE-based modeling
    - Bayesian Lasso regression for sparsity
    - Extract statistically significant regulatory interactions
4. Analysis and Visualization (prob/utils.py + notebooks)
    - Plot trajectories, networks, and hub genes
    - Compare inferred networks to curated pathway databases


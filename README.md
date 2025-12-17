# PROB Reimplementation for Alzheimer’s Disease
This repository contains a reimplementation of PROB (Progression-Based Bayesian Method) applied to Alzheimer’s disease (AD) transcriptomic data. The pipeline infers latent disease progression and gene regulatory networks from cross-sectional data.

## Backgroud
Alzheimer’s disease (AD) is a progressive neurodegenerative disorder. PROB addresses this challenge by:

- Inferring pseudotemporal ordering of samples (PPD) using a diffusion-based random walk.
- Inferring a directed gene regulatory network via ODE models with Bayesian Lasso regression.

**Key findings:**

- PPD strongly correlates with Braak stage (Spearman ρ = 0.93).
- Regulatory network analysis highlights microglial activation and lipid metabolism.
- Hub genes such as CLU and APOE influence immune-related genes like TREM2, TYROBP, CD33.

All code, scripts, and processed data required to reproduce the analyses and figures are included.

## Usage Instructions
Steps to produce the results from the Alzheimer's dataset. All steps assume you are starting from a fresh clone of the repository, and it must be noted that the dataset will automatically download when running the pipeline.

### 1. Clone the repository 
Open a terminal or PowerShell, navigate to your working folder, and clone the repo:

```bash
    git clone https://github.com/mbel24/PROB-reimplementation.git
    cd PROB-reimplementation
```

   
### 2. Create and activate a virtual environment
Windows:
```powershell
   python -m venv venv
   .\venv\Scripts\activate
```
Linux/max OS:
```bash
   python3 -m venv venv
   source venv/bin/activate
```
### 3. Install dependencies

This installs all necessary Python packages

`pip install -r requirements.txt`

### 4. Run the pipeline:

The pipeline is run using the `run_pipeline.py` script located in the `scripts/` folder:

`python scripts/run_pipeline.py`

This executes all steps:

- Load and preprocess data.
- Compute progression inference (PPD) for all samples.
- Infer the gene regulatory network using Bayesian Lasso regression.
- Generate summary statistics and figures.
    
### 5. View results

All output is saved in the results/ folder (created automatically):

- ppd_values.csv — pseudotemporal progression distances for each sample
- grn_adjacency_matrix.csv — inferred regulatory network weights
- grn_confidence_matrix.csv — confidence probabilities for edges
- PNG plots:

    - gene_regulatory_network.png — network visualization
    - ppd_vs_stage.png — PPD distribution and clinical stage
    - gene_expression_heatmap.png — gene expression along pseudotime
    - adjacency_matrix_and_presence_probability.png — adjacency & confidence matrices
 

### 6. Notes for reproducibility

- As we run it on Windows, the pipeline uses a non-interactive Matplotlib backend (`Agg`) to avoid Tcl/Tk errors.
- All scripts in `prob/` are self-contained.
- The virtual environment ensures dependencies are isolated.
- We also facilitated some jupyter notebooks for interactive exploration. The notebooks reproduce all figures (including the ones produced by the synthetic and original data), make sure you run the cells sequentially to ensure reproducibility and that you have downloaded all phyton files in `prob/` and the necesary data (`ExampleData`) which you can find in the `data/` folder

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
4. Analysis and Visualization (prob/analyze_and_plot_results.py + notebooks)
    - Plot trajectories, networks, and hub genes


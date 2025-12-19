# PROB Reimplementation for Alzheimer’s Disease
This repository contains a reimplementation of PROB (Progression-Based Bayesian Method) applied to Alzheimer’s disease (AD) transcriptomic data. The pipeline infers latent disease progression and gene regulatory networks from cross-sectional data, enabling disease staging and mechanistic insight without longitudinal samples.

## Background
Alzheimer’s disease (AD) is a progressive neurodegenerative disorder. PROB addresses this challenge by:

- Inferring pseudotemporal ordering of samples (PPD) using a diffusion-based random walk.
- Inferring a directed gene regulatory network via ODE models with Bayesian regression.

**Key findings:**

- PPD strongly correlates with Braak stage (Spearman ρ = 0.93).
- Regulatory network analysis highlights microglial activation and lipid metabolism.
- Hub genes such as CLU and APOE influence immune-related genes like TREM2, TYROBP, CD33.

All code, scripts, and processed data required to reproduce the analyses and figures are included.

## Usage Instructions
Steps to produce the results from the Alzheimer's dataset. All steps assume you are starting from a fresh clone of the repository, and it must be noted that the dataset will automatically download when running the pipeline.

### 1. Clone the repository 
Open a terminal or PowerShell, navigate to your working folder, and clone the repo:

```Bash
    git clone https://github.com/mbel24/PROB-reimplementation.git
    cd PROB-reimplementation
```

   
### 2. Create and activate a virtual environment
Windows:
```Powershell
   python -m venv venv
   .\venv\Scripts\activate
```
Linux/maxOS:
```Bash
   python3 -m venv venv
   source venv/bin/activate
```
### 3. Install dependencies

This installs all necessary Python packages
```Bash
pip install -r requirements.txt
```

### 4. Run the pipeline:

The pipeline is run using the `run_pipeline.py` script located in the `scripts/` folder:

`python scripts/run_pipeline.py`

This executes all steps:

- Load and preprocess data.
- Compute progression inference (PPD) for all samples.
- Infer the gene regulatory network using Bayesian regression.
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
    - real_data_summary.png - summary figure including: PPD histogram, ordered PPD values,  GRN heatmap, and out-degree distribution of regulators.
    - key_gene_trajectories.png - expression trajectories of key genes along PPD. Braak stage is represented by color.
 

### 6. Notes for reproducibility
- Random seeds are fixed where applicable.
- All figures can be regenerated using either:
  - `scripts/run_pipeline.py`, or
  - the provided Jupyter notebooks 
- Tested on:
  - Windows 10 (that why the pipeline uses a non-interactive Matplotlib backend (`Agg`) to avoid Tcl/Tk errors)
  - Python ≥ 3.9

## Pipeline Workflow and repository structure
1. Data Loading and Preprocessing (prob/preprocessing.py)
   - Normalization
   - Filtering
   - Formatting for downstream analysis
2. Pseudotemporal Ordering (prob/progression.py)
    - Diffusion-based random walk
    - Compute progression pseudotime distances (PPD)
    - Evaluate correlation with Braak stage
3. Gene Regulatory Network Inference (prob/ode_bayesian.py)
    - ODE-based modeling
    - Bayesian regression for sparsity
    - Extract statistically significant regulatory interactions
4. Analysis and Visualization (prob/analyze_and_plot_results.py + notebooks)
    - Plot trajectories, networks, and hub genes

## Other information
This whole repository is a reimplementation of PROB, specifically the model pressented in the article _Inferring latent temporal progression and regulatory networks from cross-sectional transcriptomic data of cancer samples published_ by Sun et al. (2021)

### Differences from Original PROB Implementation
- Rewritten entirely in Python for improved accessibility.
- Adapted to Alzheimer’s disease transcriptomic data.
- Modular pipeline with automated data download.
- Additional visualization and summary statistics.

### References: 
* Sun, X., Zhang, J., & Nie, Q. (2021). Inferring latent temporal progression and regulatory networks from cross-sectional transcriptomic data of cancer samples. PLoS Computational Biology, 17(3), e1008379. https://doi.org/10.1371/journal.pcbi.1008379
* SunXQlab. (n.d.). GitHub - SunXQlab/PROB: pseudotemporal progression-based Bayesian method for inferring GRNs from cross-sectional clinical transcriptomic data. GitHub. https://github.com/SunXQlab/PROB?tab=readme-ov-file#functions


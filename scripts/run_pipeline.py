"""
run_pipeline.py

End-to-end PROB reimplementation pipeline:
1. Download & preprocess AD microarray data (GSE48350)
2. Infer disease progression pseudotime (PPD)
3. Infer gene regulatory network using ODE + Bayesian regression
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prob'))

import numpy as np
import pandas as pd

from preprocessing import (load_GSE48350, select_genes, filter_and_normalize)
from progression import Progression_Inference
from ode_bayesian import ODE_Bayesian
from analysis_and_plots import (analyze_and_plot_results, plot_real_data_summary, plot_key_gene_trajectories)

def main():

    print("\n================ PROB PIPELINE START ================\n")

    # --------------------------------------------------
    # Step 0: Load and preprocess data
    # --------------------------------------------------
    print("Step 0: Data loading & preprocessing\n")

    input_df = load_GSE48350()
    prob_input, gene_names = select_genes(
        input_df,
        candidate_genes=None,
        top_n=20
    )

    prob_input = filter_and_normalize(
        prob_input,
        stage_min=3
    )

    # --------------------------------------------------
    # Step 1: Progression inference (PPD)
    # --------------------------------------------------
    print("Step 1: Disease progression inference\n")

    Data_ordered, PPD, TimeSampled = Progression_Inference(prob_input)

    os.makedirs("results", exist_ok=True)

    pd.DataFrame({
        "PPD": PPD
    }).to_csv("results/ppd_values.csv", index=False)

    print("✓ PPD saved to results/ppd_values.csv\n")

    # --------------------------------------------------
    # Step 2: Gene regulatory network inference
    # --------------------------------------------------
    print("Step 2: Gene regulatory network inference\n")

    Para_Post_pdf, S, AM = ODE_Bayesian(
        Data_ordered,
        TimeSampled,
        verbose=True
    )

    # Save adjacency matrix
    grn_df = pd.DataFrame(
        AM,
        index=gene_names,
        columns=gene_names
    )
    grn_df.to_csv("results/grn_adjacency_matrix.csv")

    # Save confidence matrix
    conf_df = pd.DataFrame(
        S,
        index=gene_names,
        columns=gene_names
    )
    conf_df.to_csv("results/grn_confidence_matrix.csv")

    print("\n✓ Network results saved to results/")
    print("  - grn_adjacency_matrix.csv")
    print("  - grn_confidence_matrix.csv")

    print("\n================ PIPELINE COMPLETE ==================\n")

    analyze_and_plot_results(
        genes_of_interest=gene_names,
        AM=AM,
        S=S,
        Data_ordered=Data_ordered,
        PPD=PPD,
        prob_input_normalized=prob_input,
        outdir="results"
    )
    plot_real_data_summary(
        PPD_real=PPD,
        AM_real=AM,
        genes_of_interest=gene_names,
        outdir="results"
    )
    plot_key_gene_trajectories(
        genes_of_interest=gene_names,
        PPD_real=PPD,
        prob_input=prob_input,
        Data_ordered_real=Data_ordered,
        outdir="results"
    )


if __name__ == "__main__":
    main()





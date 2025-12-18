import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr

def analyze_and_plot_results(
    genes_of_interest,
    AM,
    S,
    Data_ordered,
    PPD,
    prob_input_normalized,
    outdir="results"
):
    """
    Analyze PROB outputs and generate summary statistics and publication-ready figures.
    
    Parameters
    ----------
    genes_of_interest : list of str
        List of gene symbols included in the GRN inference.
        The order must correspond to the rows and columns of `AM`, `S`, and the rows of `Data_ordered`.

    AM : numpy.ndarray, shape (n_genes, n_genes)
        Inferred adjacency matrix from the Bayesian ODE model.
        Entry AM[i, j] represents the regulatory effect of gene j (source) on gene i (target). Positive values indicate activation and negative values indicate inhibition.

    S : numpy.ndarray, shape (n_genes, n_genes)
        Presence probability matrix for regulatory edges.
        Entry S[i, j] gives the posterior probability that a regulatory interaction exists from gene j to gene i.

    Data_ordered : numpy.ndarray, shape (n_genes, n_samples)
        Gene expression matrix ordered by inferred pseudotime (PPD).
        Each row corresponds to a gene and each column to a sample.

    PPD : numpy.ndarray, shape (n_samples,)
        Pseudotemporal Progression Distance inferred by PROB.
        Represents the latent disease progression ordering of samples.

    prob_input_normalized : numpy.ndarray, shape (n_features, n_samples)
        Normalized input data matrix used by PROB.
        The last row is assumed to contain clinical stage information (e.g., Braak stage) for each sample.

    outdir : str, optional (default="results")
        Output directory where all figures and summary files will be saved.
        The directory is created if it does not exist.

    Outputs
    -------
    Files written to `outdir`:
        - gene_regulatory_network.png
            Visualization of the inferred directed GRN with activation (green) and inhibition (red) edges.
        - ppd_vs_stage.png
            Distribution of PPD values and their relationship to clinical stage.
        - gene_expression_heatmap.png
            Heatmap showing gene expression dynamics along pseudotime.
        - adjacency_matrix_and_presence_probability.png
            Heatmaps of the adjacency matrix (AM) and edge confidence matrix (S).
        - summary.txt
            Text summary including network statistics and correlation between PPD and clinical stage.

    Notes
    -----
    - The function uses a non-interactive Matplotlib backend ("Agg") to ensure
      compatibility with headless environments.
    - Strong regulatory edges (|weight| > 10) are labeled in the network plot.
    - Network layout is deterministic due to a fixed random seed.

    """
    os.makedirs(outdir, exist_ok=True)
    
    # Build directed graph
    G = nx.DiGraph()
    for gene in genes_of_interest:
      G.add_node(gene)
    
    edge_list = []
    for i, gene_target in enumerate(genes_of_interest):
      for j, gene_source in enumerate(genes_of_interest):
          if AM[i, j] != 0:
              G.add_edge(gene_source, gene_target, weight=AM[i, j])
              edge_list.append((gene_source, gene_target, AM[i, j]))
    # Separate edges by sign (activation vs inhibition)
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    node_degrees = dict(G.degree())
    node_sizes = [3000 + node_degrees[node] * 500 for node in G.nodes()]
    node_colors = [node_degrees[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                         cmap='YlOrRd', alpha=0.9, edgecolors='black', 
                         linewidths=2, ax=ax)
    if len(pos_edges) > 0:
      nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color='green',
                            width=3, alpha=0.7, arrows=True, arrowsize=25,
                            arrowstyle='->', connectionstyle='arc3,rad=0.1',
                            ax=ax, label='Activation')
    
    if len(neg_edges) > 0:
      nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='red',
                            width=3, alpha=0.7, arrows=True, arrowsize=25,
                            arrowstyle='->', connectionstyle='arc3,rad=0.1',
                            ax=ax, label='Inhibition')
    
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold',
                          font_family='sans-serif', ax=ax)
    edge_labels = {}
    for u, v, w in edge_list:
      if abs(w) > 10:  # Only label strong edges
          edge_labels[(u, v)] = f'{w:.1f}'
    
    if edge_labels:
      nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax)
    
    ax.set_title(f'Alzheimer\'s Gene Regulatory Network\n{len(G.edges())} regulatory interactions', 
              fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.axis('off')
  
    plt.tight_layout()
    plt.savefig(f"{outdir}/gene_regulatory_network.png", dpi=300)
    plt.close()
    
    stages = prob_input_normalized[-1, :].astype(int)
    corr, pval = spearmanr(stages, PPD)
    with open(f"{outdir}/summary.txt", "w") as f:
        f.write("PROB ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total genes: {len(genes_of_interest)}\n")
        f.write(f"Total edges: {len(G.edges())}\n")
        f.write(f"Network density: {nx.density(G):.3f}\n\n")

        f.write("PPD vs Braak Stage:\n")
        f.write(f"  Spearman r = {corr:.4f}\n")
        f.write(f"  p-value = {pval:.2e}\n\n")

        f.write("Top regulators (out-degree):\n")
        out_deg = dict(G.out_degree())
        for gene, deg in sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"  {gene}: {deg}\n")
          
    # --------------------------------------------------
    # PPD plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.hist(PPD, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Pseudotemporal Progression Distance (PPD)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Inferred Progression')
    ax.grid(True, alpha=0.3)
    
    # Stage distribution scatter
    ax = axes[1]
    stages = prob_input_normalized[-1, :].astype(int)
    for stage in np.unique(stages):
      mask = stages == stage
      ax.scatter([stage]*np.sum(mask), PPD[mask], label=f'Stage {stage}', alpha=0.6, s=80)
    ax.set_xlabel('Clinical Stage (Braak)')
    ax.set_ylabel('PPD')
    ax.set_title('Progression by Clinical Stage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/ppd_vs_stage.png", dpi=300)
    plt.close()
    
    #--------------------------------------------------
    # Gene expression heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(Data_ordered, aspect='auto', cmap='viridis')
    ax.set_xlabel('Pseudotime', fontsize=12)
    ax.set_ylabel('Genes', fontsize=12)
    ax.set_title('Gene Expression Along Inferred Progression Trajectory', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expression Level', fontsize=11)
    
    # Set y-axis labels with actual gene names
    ax.set_yticks(np.arange(Data_ordered.shape[0]))
    ax.set_yticklabels(genes_of_interest, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/gene_expression_heatmap.png", dpi=300)
    plt.close()
    
    #--------------------------------------------------
    # Inferred gene regulatory network
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Adjacency matrix (AM)
    ax = axes[0]
    vmax = np.max(np.abs(AM))
    im1 = ax.imshow(AM, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Source Gene', fontsize=12)
    ax.set_ylabel('Target Gene', fontsize=12)
    ax.set_title('Adjacency Matrix: Regulatory Strengths', fontsize=12)
    
    # Set tick labels with actual gene names
    ax.set_xticks(np.arange(len(genes_of_interest)))
    ax.set_yticks(np.arange(len(genes_of_interest)))
    ax.set_xticklabels(genes_of_interest, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(genes_of_interest, fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label('Coefficient', fontsize=10)
    n_edges = np.sum(AM != 0)
    ax.text(0.5, -0.25, f'Network edges: {n_edges}', transform=ax.transAxes, ha='center')
    
    # Presence probability (S)
    ax = axes[1]
    im2 = ax.imshow(S, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Source Gene', fontsize=12)
    ax.set_ylabel('Target Gene', fontsize=12)
    ax.set_title('Presence Probability: Edge Confidence', fontsize=12)
    
    # Set tick labels with actual gene names
    ax.set_xticks(np.arange(len(genes_of_interest)))
    ax.set_yticks(np.arange(len(genes_of_interest)))
    ax.set_xticklabels(genes_of_interest, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(genes_of_interest, fontsize=9)
    
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label('P(edge)', fontsize=10)
    high_conf = np.sum(S > 0.95)
    ax.text(0.5, -0.25, f'High confidence edges (P>0.95): {high_conf}', transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/adjacency_matrix_and_presence_probability.png", dpi=300)
    plt.close()
    
    
    
    

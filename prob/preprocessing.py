import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import GEOparse

def load_GSE48350(dest_dir="./data"):
    """
    Download and preprocess the GSE48350 Alzheimer’s disease transcriptomic dataset (GSE48350 microarray dataset from the Gene Expression).

    Parameters
    ----------
    dest_dir : str, optional (default="./data")
        Directory where GEO files and annotations will be downloaded and cached.

    Returns
    -------
    input_df : pandas.DataFrame
        Gene-level expression matrix with samples as columns.
        Rows correspond to gene symbols, with an additional final row:
            - 'braak stage' : numerical Braak stage for each sample
        Shape:
            (n_genes + 1, n_samples)

    Notes
    -----
    - Probe-level expression values are collapsed to gene-level expression by averaging probes mapping to the same gene symbol.
    - Braak stages are parsed from sample metadata and converted from Roman numerals (I–VI) to integers (1–6). Missing or unknown stages are set to 0.
    - The returned DataFrame is structured to be directly compatible with downstream PROB preprocessing steps.

    Raises
    ------
    ValueError
        If Braak stage information cannot be found in the GEO metadata.
    """
    
    print("📥 Downloading GSE48350 from GEO...")
    gse = GEOparse.get_GEO("GSE48350", destdir="./data", how="full")
    
    expr = gse.pivot_samples("VALUE")
    print(f"Expression matrix shape (probes x samples): {expr.shape}")
    
    # Build metadata table
    meta_rows = []
    for gsm_name, gsm in gse.gsms.items():
        row = {"sample_id": gsm_name}
        for field in gsm.metadata.get("characteristics_ch1", []):
            if ":" in field:
                key, val = field.split(":", 1)
                row[key.strip().lower()] = val.strip()
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows).set_index("sample_id")
    print(f"Metadata shape: {meta_df.shape}")
    print("Metadata columns:", meta_df.columns.tolist())
    
    # Map probe IDs to gene symbols using GPL570 annotation
    print("Fetching GPL570 annotation...")
    gpl = GEOparse.get_GEO("GPL570", destdir="./data")
    annot = gpl.table[["ID", "Gene Symbol"]].rename(columns={"ID": "ID_REF", "Gene Symbol": "gene_symbol"})
    
    expr = expr.merge(annot, on="ID_REF", how="left")
    expr = expr.dropna(subset=["gene_symbol"])
    expr = expr.groupby("gene_symbol").mean(numeric_only=True)
    print(f"Collapsed to gene-level expression: {expr.shape}")
    
    # Get braak stage from metadata (transpose metadata and map roman numerals)
    meta_df_t = meta_df.transpose()
    if 'braak stage' not in meta_df_t.index:
        raise ValueError("Could not find 'braak stage' in metadata characteristics. Please check metadata keys.")
    roman_to_int = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'unknown': np.nan}
    numerical_braak_stage = meta_df_t.loc['braak stage'].map(roman_to_int).fillna(0)
    numerical_braak_stage.index.name = 'sample_id'
    braak_stage_row = pd.DataFrame(numerical_braak_stage).T
    braak_stage_row.index = ['braak stage']
    
    # Merge expression + braak stage as rows (genes rows, samples columns)
    input_df = pd.concat([expr, braak_stage_row], axis=0)

    return input_df

def select_genes(input_df, candidate_genes=None, top_n=20):
    """
    This function selects a subset of genes for downstream PROB modeling. If a list of candidate genes is provided, the function selects the top
    `top_n` genes with the highest expression variance across samples.The Braak stage row is appended to the final input matrix.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Gene-level expression DataFrame with samples as columns and genes as rows.
        Must include a row labeled 'braak stage'.

    candidate_genes : list of str, optional (default=None)
        List of candidate gene symbols to consider.
        If None, a predefined list of Alzheimer’s disease–related genes (GWAS hits, pathway genes, and known risk factors) is used.

    top_n : int, optional (default=20)
        Number of genes to select based on highest variance.
        If fewer than `top_n` candidate genes are present in the dataset, all available genes are selected.

    Returns
    -------
    prob_input : numpy.ndarray
        PROB input matrix of shape (n_genes + 1, n_samples), where the final row corresponds to Braak stage.

    genes_of_interest : list of str
        List of selected gene symbols used for downstream analysis.
    """

    if candidate_genes is None:
        candidate_genes = [
            'APOE', 'APP', 'PSEN1', 'PSEN2', 'MAPT',  # Core AD genes
            'TREM2', 'TYROBP', 'SORL1', 'CD33', 'BIN1',  # GWAS hits
            'CLU', 'PICALM', 'ABCA7', 'EPHA1', 'MS4A6A',  # Risk factors
            'BACE1', 'ADAM10', 'GSK3B', 'NOTCH3', 'INPP5D',  # Pathways
            'MEF2C', 'ZCWPW1', 'PTK2B', 'CELF1', 'NME8',  # Additional hits
            'CASS4', 'FERMT2', 'HLA-DRB5', 'SLC24A4', 'CR1'  # More candidates
        ]
    
    # Check which genes are available in the dataset
    available_genes = [g for g in candidate_genes if g in input_df.index]
    
    if len(available_genes) >= top_n:
        gene_variances = input_df.loc[available_genes].var(axis=1)
        genes_of_interest = gene_variances.nlargest(top_n).index.tolist()
        print(f"\nSelected top 20 genes by variance:")
    else:
        genes_of_interest = available_genes

    for i, gene in enumerate(genes_of_interest):
        var = input_df.loc[gene].var()
        print(f"  {i+1}. {gene} (variance: {var:.2f})")

    final_rows = genes_of_interest + ['braak stage']
    # Guard: ensure all selected rows exist
    missing = [r for r in final_rows if r not in input_df.index]
    if len(missing) > 0:
        raise KeyError(f"These rows are missing from the processed expression dataframe: {missing}")
    
    prob_input = input_df.loc[final_rows].to_numpy()   # shape: (n_genes+1, n_samples)
    print(f"\nInitial PROB input matrix shape: {prob_input.shape} ({len(genes_of_interest)} genes x {prob_input.shape[1]} samples)")

    return prob_input, genes_of_interest

def filter_and_normalize(prob_input, stage_min=3):
    """
    This function removes samples below a specified Braak stage threshold and applies z-score normalization to gene expression values while preserving clinical stage information.

    Parameters
    ----------
    prob_input : numpy.ndarray
        Input matrix of shape (n_genes + 1, n_samples), where the final row corresponds to Braak stage and all preceding rows correspond to genes.

    stage_min : int, optional (default=3)
        Minimum Braak stage required for a sample to be retained.
        Samples with Braak stage < stage_min are removed.

    Returns
    -------
    prob_input : numpy.ndarray
        Filtered and normalized PROB input matrix of shape (n_genes + 1, n_filtered_samples).

    Notes
    -----
    - Gene expression values are standardized using z-score normalization (mean = 0, standard deviation = 1) across samples.
    - The Braak stage row is excluded from normalization.
    - This preprocessing step ensures comparable gene scales for pseudotime inference and GRN modeling.
    """
    
    stage_mask = prob_input[-1, :] >= stage_min
    prob_input = prob_input[:, stage_mask]

    scaler = StandardScaler()
    prob_input[:-1, :] = scaler.fit_transform(prob_input[:-1, :].T).T  # don't scale stage row
    print(f"\n✓ Expression data normalized (z-score)")
    print(f"  Mean per gene: {np.mean(prob_input[:-1, :], axis=1)[:3]} ... (should be ~0)")
    print(f"  Std per gene: {np.std(prob_input[:-1, :], axis=1)[:3]} ... (should be ~1)")
    
    print(f"\nFinal PROB input matrix shape: {prob_input.shape}")
    print("="*60 + "\n")

    return prob_input

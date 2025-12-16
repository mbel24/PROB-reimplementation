import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import GEOparse

def load_GSE48350(dest_dir="./data"):
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
    #SELECT GENES OF INTEREST & BUILD prob_input
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
        
    final_rows = genes_of_interest + ['braak stage']
    # Guard: ensure all selected rows exist
    missing = [r for r in final_rows if r not in input_df.index]
    if len(missing) > 0:
        raise KeyError(f"These rows are missing from the processed expression dataframe: {missing}")
    
    prob_input = input_df.loc[final_rows].to_numpy()   # shape: (n_genes+1, n_samples)
    return prob_intup, genes_of_interest

def filter_and_normalize(prob_input, stage_min=3):
    """Filter samples by Braak stage and normalize gene expression."""
    stage_mask = prob_input[-1, :] >= stage_min
    prob_input = prob_input[:, stage_mask]

    scaler = StandardScaler()
    prob_input[:-1, :] = scaler.fit_transform(prob_input[:-1, :].T).T  # don't scale stage row
    return prob_input

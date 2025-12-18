import numpy as np
from sklearn.linear_model import BayesianRidge
from scipy import stats

def ODE_Bayesian(Data_ordered, TimeSampled, verbose=True):
    """
    This function implements the core PROB gene regulatory network (GRN) inference step. 
    Gene expression dynamics are modeled using a system of ordinary differential equations (ODEs), 
    where the time derivative of each gene is expressed as a linear combination
    of interaction terms between genes. Bayesian Ridge regression with sparsity-inducing priors is 
    used to infer regulatory strengths and posterior confidence intervals.

    Parameters
    ----------
    Data_ordered : numpy.ndarray, shape (n_genes, n_samples)
        Gene expression matrix ordered by inferred pseudotime (PPD).
        Rows correspond to genes and columns correspond to samples sorted along the latent disease progression.

    TimeSampled : numpy.ndarray, shape (n_samples,)
        Pseudotime values associated with each sample.
        These are normalized internally to the range [0, 1] and used to approximate time derivatives.

    verbose : bool, optional (default=True)
        If True, print progress messages during GRN inference.

    Returns
    -------
    Para_Post_pdf : dict
        Dictionary containing fitted Bayesian regression models for each target gene.
        For each gene index i:
            Para_Post_pdf[i]['model'] : sklearn.linear_model.BayesianRidge
                Trained Bayesian regression model.
            Para_Post_pdf[i]['coef'] : numpy.ndarray
                Estimated regulatory coefficients for all source genes (excluding self-regulation).
    
    S : numpy.ndarray, shape (n_genes, n_genes)
        Edge presence probability matrix.
        Entry S[i, j] represents the posterior probability that gene j regulates gene i, estimated from confidence intervals across multiple significance levels.
    
    AM : numpy.ndarray, shape (n_genes, n_genes)
        Adjacency matrix of the inferred gene regulatory network.
        Entry AM[i, j] is the estimated regulatory coefficient from gene j to gene i if the posterior confidence exceeds the specified threshold (P > 0.75); otherwise, the entry is zero.

    Notes
    -----
    - Time derivatives are approximated using finite differences of pseudotime-ordered gene expression.
    - For each target gene i, the model includes interaction terms of the form x_j * x_i for all source genes j ≠ i, consistent with the PROB formulation.
    - Confidence intervals are computed across multiple alpha levels (0.01–1.0), and edge presence probabilities are derived from whether intervals exclude zero.
    - Self-regulatory edges are excluded by construction.
    - The confidence threshold (P > 0.75) was chosen to yield a sparse, interpretable network and can be adjusted.
    - This implementation follows the PROB framework described in Sun et al. (2021).
    """
    
    if verbose:
        print('Step 2: GRN Inference')
    x = Data_ordered.copy()
    Time = (TimeSampled - np.min(TimeSampled)) / (np.max(TimeSampled) - np.min(TimeSampled))
    dTime = np.diff(Time)
    y = np.diff(x, axis=1) / dTime[np.newaxis, :]
    x = x[:, :-1]
    n_genes, n_timepoints = x.shape
    Para_Post_pdf, alpha_levels = {}, np.arange(0.01, 1.01, 0.01)
    CI_alpha = np.zeros((n_genes, 100, n_genes + 1, 2))
    for target_gene in range(n_genes):
        y_output = y[target_gene, :]
        x_input = np.column_stack([x[s, :] * x[target_gene, :] for s in range(n_genes)] + [np.ones(n_timepoints)])
        x_input = np.delete(x_input, target_gene, axis=1)
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6, compute_score=True)
        model.fit(x_input, y_output)
        Para_Post_pdf[target_gene] = {'model': model, 'coef': model.coef_}
        model_std = np.sqrt(np.diag(model.sigma_))
        for alpha_idx, alpha_val in enumerate(alpha_levels):
            t_score = stats.t.ppf([alpha_val/2, 1-alpha_val/2], df=1e4-1)
            ci = np.zeros((len(model.coef_) + 1, 2))
            ci[:-1, 0], ci[:-1, 1] = model.coef_ + t_score[0]*model_std, model.coef_ + t_score[1]*model_std
            ci[-1, 0] = model.intercept_ + t_score[0]*np.sqrt(model.sigma_[0, 0])
            ci[-1, 1] = model.intercept_ + t_score[1]*np.sqrt(model.sigma_[0, 0])
            CI_alpha[target_gene, alpha_idx, :, :] = ci
    S = np.ones((n_genes, n_genes))
    for target_gene in range(n_genes):
        for alpha_idx in range(len(alpha_levels)-1, -1, -1):
            for source_gene in range(n_genes):
                ci_val = CI_alpha[target_gene, alpha_idx, source_gene, :]
                if ci_val[0] <= 0 <= ci_val[1]:
                    S[target_gene, source_gene] = min(S[target_gene, source_gene], 1 - alpha_levels[alpha_idx])
    S_new = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        S_new[i, [j for j in range(n_genes) if j != i]] = S[i, :n_genes-1]
    S = S_new
    AM = np.zeros((n_genes, n_genes))
    for target_gene in range(n_genes):
        coef = Para_Post_pdf[target_gene]['coef']
        source_idx = 0
        for source_gene in range(n_genes):
            if source_gene != target_gene and source_idx < len(coef):
                if S[target_gene, source_gene] > 0.75:  # OPTIMIZED: Set to 0.75 for 26 edges (6.84% density)
                    AM[target_gene, source_gene] = coef[source_idx]
                source_idx += 1
    if verbose:
        print(f'✓ Step 2 complete: {np.sum(AM!=0)} edges (confidence threshold: P>0.75)')
    return Para_Post_pdf, S, AM

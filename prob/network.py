def ODE_BayesianLasso(Data_ordered, TimeSampled, verbose=True):
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
                if S[target_gene, source_gene] > 0.95:
                    AM[target_gene, source_gene] = coef[source_idx]
                source_idx += 1
    if verbose:
        print(f'✓ Step 2 complete: {np.sum(AM!=0)} edges')
    return Para_Post_pdf, S, AM

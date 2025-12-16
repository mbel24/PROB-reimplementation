def knnsearch(Q, R=None, K=1):
    if R is None:
        R = Q
        fident = True
    else:
        fident = np.array_equal(Q, R)
    N = Q.shape[0]
    idx = np.zeros((N, K), dtype=int)
    D = np.zeros((N, K))
    for k in range(N):
        d = np.sum((R - Q[k, :]) ** 2, axis=1)
        if fident:
            d[k] = np.inf
        if K == 1:
            D[k, 0] = np.sqrt(np.min(d))
            idx[k, 0] = np.argmin(d)
        else:
            sorted_indices = np.argsort(d)
            idx[k, :] = sorted_indices[:K]
            D[k, :] = np.sqrt(d[sorted_indices[:K]])
    return idx, D

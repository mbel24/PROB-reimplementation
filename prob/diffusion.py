from .neighbors import knnsearch

def T_loc(data, nsig, W):
    n = data.shape[0]
    d2 = squareform(pdist(data, metric='euclidean') ** 2)
    idx, dists = knnsearch(data, data, K=nsig)
    sigma = dists[:, -1]
    S2 = sigma[:, np.newaxis] ** 2 + sigma[np.newaxis, :] ** 2
    Sw = S2 / (W + 1e-10)
    W1 = np.exp(-d2 / (Sw + 1e-10))
    D1 = np.sum(W1, axis=1)
    q = np.outer(D1, D1)
    W1 = W1 / (q + 1e-10)
    W1[d2 == 0] = 0
    D1_diag = np.diag(np.sum(W1, axis=1))
    D1_inv_sqrt = np.linalg.pinv(np.sqrt(D1_diag) + 1e-10)
    T = D1_inv_sqrt @ W1 @ D1_inv_sqrt
    phi0 = np.diag(D1_diag) / np.sqrt(np.sum(np.diag(D1_diag) ** 2))
    return T, phi0

def dpt_input(T, phi0):
    n = T.shape[0]
    I = np.eye(n)
    phi0_outer = np.outer(phi0, phi0)
    M = np.linalg.inv(I - T + phi0_outer) - I
    return M

def dpt_to_root(M, root):
    n = M.shape[0]
    dpt = np.zeros(n)
    for x in range(n):
        dpt[x] = np.sqrt(np.sum((M[int(root), :] - M[x, :]) ** 2))
    return dpt
  

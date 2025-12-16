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

def gausswin(N, w=2.5):
    n = np.arange(N)
    return np.exp(-0.5 * (w / N * (2 * n - (N - 1))) ** 2)

def ksmooth(vector, windowWidth):
    windowWidth = max(windowWidth, 3)
    gaussFilter = gausswin(windowWidth)
    gaussFilter = gaussFilter / np.sum(gaussFilter)
    return np.convolve(vector, gaussFilter, mode='same')

def Progression_Inference(X_stage):
    R = X_stage.shape
    data = X_stage[:-1, :].T
    grade = X_stage[-1, :]
    W = np.ones((len(grade), len(grade)))
    for i in range(len(grade)):
        for j in range(len(grade)):
            W[i, j] = 1.0 + abs(grade[i] - grade[j])
    T, phi0 = T_loc(data, 10, W)
    M = dpt_input(T, phi0)
    Ind_max = np.where(grade == np.max(grade))[0]
    x_ref = Ind_max[np.random.randint(len(Ind_max))]
    drn = dpt_to_root(M, x_ref)
    AA_sort = np.argsort(-drn)
    root = None
    for i in AA_sort:
        if grade[i] == np.min(grade):
            root = i
            break
    if root is None:
        root = AA_sort[0]
    PPD = dpt_to_root(M, root)
    indT = np.argsort(PPD)
    smoothL = max(int(10 ** (np.floor(np.log10(R[1])) - 1)), 3)
    Data_ordered = np.zeros((R[0] - 1, R[1]))
    for i in range(R[0] - 1):
        Data_ordered[i, :] = ksmooth(data[indT, i], smoothL)
    TimeSampled = np.linspace(0, 1, R[1])
    return Data_ordered, PPD, TimeSampled

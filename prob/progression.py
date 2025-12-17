import numpy as np
from scipy.spatial.distance import pdist, squareform

def knnsearch(Q, R=None, K=1):
    """
    Perform k-nearest neighbor search using Euclidean distance.

    Parameters
    ----------
    Q : numpy.ndarray, shape (n_query, n_features)
        Query data points.

    R : numpy.ndarray, shape (n_reference, n_features), optional
        Reference data points. If None, R is set equal to Q and self-matches are excluded.

    K : int, optional (default=1)
        Number of nearest neighbors to return.

    Returns
    -------
    idx : numpy.ndarray, shape (n_query, K)
        Indices of the K nearest neighbors in R for each query point.

    D : numpy.ndarray, shape (n_query, K)
        Euclidean distances to the K nearest neighbors.
        
    """
    
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
    """
    Construct a locally scaled diffusion transition matrix.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n_samples, n_features)
        Input data matrix (e.g., gene expression across samples).

    nsig : int
        Number of nearest neighbors used to estimate local scale (sigma).

    W : numpy.ndarray, shape (n_samples, n_samples)
        Weight matrix encoding sample similarity or dissimilarity
        (e.g., based on Braak stage differences).

    Returns
    -------
    T : numpy.ndarray, shape (n_samples, n_samples)
        Symmetrically normalized diffusion transition matrix.

    phi0 : numpy.ndarray, shape (n_samples,)
        First (trivial) diffusion eigenvector used for DPT normalization.
    """
    
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
    """
    Compute the diffusion pseudotime (DPT) matrix.
    """

    n = T.shape[0]
    I = np.eye(n)
    phi0_outer = np.outer(phi0, phi0)
    M = np.linalg.inv(I - T + phi0_outer) - I
    return M

def dpt_to_root(M, root):
    """
    Compute diffusion pseudotime distances from a specified root sample.
    """
    n = M.shape[0]
    dpt = np.zeros(n)
    for x in range(n):
        dpt[x] = np.sqrt(np.sum((M[int(root), :] - M[x, :]) ** 2))
    return dpt

def gausswin(N, w=2.5):
    """
    Generate a Gaussian smoothing window
    """
    n = np.arange(N)
    return np.exp(-0.5 * (w / N * (2 * n - (N - 1))) ** 2)

def ksmooth(vector, windowWidth):
    """
    Smooth a one-dimensional signal using Gaussian kernel smoothing.
    """
    windowWidth = max(windowWidth, 3)
    gaussFilter = gausswin(windowWidth)
    gaussFilter = gaussFilter / np.sum(gaussFilter)
    return np.convolve(vector, gaussFilter, mode='same')

def Progression_Inference(X_stage):
    """
    This function computes a pseudotemporal ordering of samples from cross-sectional transcriptomic data using 
    a diffusion-based random walk, incorporating clinical stage information to guide progression.

    Parameters
    ----------
    X_stage : numpy.ndarray, shape (n_genes + 1, n_samples)
        Input matrix where:
        - Rows 0:(n_genes) correspond to gene expression values
        - The final row corresponds to clinical stage (e.g., Braak stage)

    Returns
    -------
    Data_ordered : numpy.ndarray, shape (n_genes, n_samples)
        Gene expression matrix reordered and smoothed along inferred pseudotime.

    PPD : numpy.ndarray, shape (n_samples,)
        Pseudotemporal Progression Distance for each sample.

    TimeSampled : numpy.ndarray, shape (n_samples,)
        Uniformly sampled pseudotime values in the range [0, 1].

    Notes
    -----
    - Diffusion distances are computed using a locally scaled kernel weighted by clinical stage differences.
    - The root sample is selected from the earliest disease stage.
    - Gene expression trajectories are smoothed along pseudotime to reduce noise.
    - This implementation follows the PROB framework described in Sun et al. (2021).
    """
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

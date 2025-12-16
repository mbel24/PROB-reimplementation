from .diffusion import T_loc, dpt_input, dpt_to_root
from .kernels import ksmooth

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

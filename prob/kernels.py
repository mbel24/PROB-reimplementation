def gausswin(N, w=2.5):
    n = np.arange(N)
    return np.exp(-0.5 * (w / N * (2 * n - (N - 1))) ** 2)

def ksmooth(vector, windowWidth):
    windowWidth = max(windowWidth, 3)
    gaussFilter = gausswin(windowWidth)
    gaussFilter = gaussFilter / np.sum(gaussFilter)
    return np.convolve(vector, gaussFilter, mode='same')

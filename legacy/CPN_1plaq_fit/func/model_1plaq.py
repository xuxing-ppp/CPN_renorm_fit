import numpy as np

def norm(da, alpha, n_terms=None):
    if n_terms is None:
        n_terms = np.max([int(np.ceil(10 / np.sqrt(alpha))), 10])  # 根据 alpha 动态调整 n_terms
    # if da is a scalar
    n = np.arange(-n_terms, n_terms + 1)
    if np.isscalar(da):
        terms = np.exp(-alpha / 2 * (2 * np.pi * n + da) ** 2)
        return np.sum(terms)
    terms = np.exp(-alpha / 2 * (2 * np.pi * n[:, None] + da[None, :]) ** 2)
    return np.sum(terms, axis=0)

def Villain_dist_norm_fit(X, alpha):
    '''
    :param X: (bs, 2)-dim array, s, da
    '''
    s = X[:, 0]
    da = X[:, 1]

    return np.exp(-alpha / 2 * (da + 2 * np.pi * s) ** 2) / norm(da, alpha)

def Villain_dist_norm_visualize(f, alpha):
    '''
    :param f: (bs,)-dim array: s + da / (2 * pi)
    '''
    da = 2 * np.pi * (f - np.round(f))

    return np.exp(-alpha / 2 * (2 * np.pi * f) ** 2) / norm(da, alpha)
import numpy as np
from utils.fitting import multi_gaussian_with0

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def p0_func(*args, config=None, **kwargs):
    # print([config[2][0] ,0.7 , 5.6, 10000.,1000.,100. , config[1][0] ,0. , 100. ,10.])
    return [config[2][0], 0.7, 5.6, 10000., 1000., 100., config[1][0], 0., 100., 10.]
    # return [0.7 , 5.6, 10000.,1000.,100. , 0. , 100. ,10.]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def bounds_func(x, *args, config=None, **kwargs):
    param_min = [config[2][0] * 0.1, 0.01, 0., 100., 1., 0., config[1][0] - config[1][1], -100., 0., 0.]
    param_max = [config[2][0] * 10., 5., 100., np.inf, np.inf, np.inf, config[1][0] + config[1][1], 100., np.inf,
                 np.inf]
    # param_min = [0.01, 0. , 100.   , 1.    , 0.  ,-10., 0.    ,0.]
    # param_max = [5. , 100., np.inf, np.inf, np.inf,10. , np.inf,np.inf]
    return param_min, param_max


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def slice_func(x, *args, **kwargs):
    if np.where(x != 0)[0].shape[0] == 0:
        return [0, 1, 1]
    return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


# noinspection PyUnusedLocal
def fit_func(p, x, *args, **kwargs):
    p_new = [0.] * 12
    p_new[0] = 0.
    p_new[1] = p[0]
    p_new[2] = p[1]
    p_new[3] = p[2]
    p_new[4] = p[3]
    p_new[5] = p[4]
    p_new[6] = p[5]
    p_new[7] = p[6]
    p_new[8] = p[7]
    p_new[9] = 1.
    p_new[10] = p[8]
    p_new[11] = p[9]
    return multi_gaussian_with0(p_new, x)

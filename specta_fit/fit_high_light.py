import numpy as np
from scipy.optimize import curve_fit
import peakutils
from utils.pdf import generalized_poisson,gaussian

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyUnusedLocal,PyUnusedLocal
def p0_func(y, x, *args, config=None, **kwargs):
    """
    find the parameters to start a mpe fit with low light
    :param y: the histogram values
    :param x: the histogram bins
    :param args: potential unused positionnal arguments
    :param config: should be the fit result of a previous fit
    :param kwargs: potential unused keyword arguments
    :return: starting points for []
    """
    if type(config).__name__ != 'ndarray':
        raise ValueError('The config parameter is mandatory')

    mu = np.average(x, weights=y)-config[3,0]
    amplitude = np.sum(y)
    if np.isnan(mu ):mu=9.
    return [mu, config[1,0], config[2,0], config[3,0], config[4,0], config[5,0], amplitude, config[7,0]]



# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the histogram values
    :param x: the histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the histogram
    """
    # Check that the histogram has none empty values
    if np.where(x != 0)[0].shape[0] == 0:
        return [0, 1, 1]
    return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    offset = config[1]
    gain = config[5]
    sig = config[2]
    if np.any(np.isnan(offset)) or np.any(np.isnan(sig)) or np.any(np.isnan(gain)): return [-np.inf]*7,[np.inf]*7
    if type(config).__name__ == 'ndarray':
        param_min = [0., 0.,gain[0]-10*gain[1] , offset[0]-3*sig[0], 1.e-2,1.e-2,0.]
        param_max = [np.inf, 1, gain[0]+10*gain[1], offset[0]+3*sig[0],10., 10.,np.inf]
    else:
        param_min = [0., 0., 0., -np.inf, 0., 0., 0.]
        param_max = [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf]

    return param_min, param_max


def fit_func(p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude , offset= p
    temp = np.zeros(x.shape)
    x = x - baseline
    n_peak = 15
    for n in range(0, n_peak, 1):

        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2)

        temp += generalized_poisson(n, mu, mu_xt) * gaussian(x, sigma_n, n * gain + (offset if n!=0 else 0))

    return temp * amplitude

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

    # Get the list of peaks in the histogram
    threshold = 0.05
    min_dist = 15
    peaks_index = peakutils.indexes(y, threshold, min_dist)
    if len(peaks_index) == 0:
        return [np.nan] * 7
    # Get a primary amplitude to consider
    amplitude = np.sum(y)
    # Get previous estimation of the gain
    gain = config[2,0]
    sigma_start = np.zeros(peaks_index.shape[-1])
    for i in range(peaks_index.shape[-1]):
        start = max(int(peaks_index[i] - gain // 2), 0)  ## Modif to be checked
        end = min(int(peaks_index[i] + gain // 2), len(x))  ## Modif to be checked
        if start == end and end < len(x) - 2:
            end += 1
        elif start == end:
            start -= 1

        if i == 0:
            mu = -np.log(np.sum(y[start:end]) / np.sum(y))
        try:
            temp = np.average(x[start:end], weights=y[start:end])
            sigma_start[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))
        except Exception as inst:
            print(inst)
            print(y)
            print(start,end)
            print(y[start:end])
            print(np.any(np.isnan(y[start:end])))
            sigma_start[i] = config[4,0]
            mu = 5

    bounds = [[0., 0.], [np.inf, np.inf]]
    sigma_n = lambda x, y, n: np.sqrt(x ** 2 + n * y ** 2)
    sigma, sigma_error = curve_fit(sigma_n, np.arange(0, peaks_index.shape[-1], 1), sigma_start, bounds=bounds)
    sigma = sigma
    return [mu, config[1,0], gain, config[3,0], config[4,0], sigma[1], amplitude, config[7,0]]



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
    if np.where(y != 0)[0].shape[0] == 0:
        return []
    return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    baseline = config[3]
    gain = config[2]
    sig = config[4]
    if np.any(np.isnan(baseline)) or np.any(np.isnan(sig)) or np.any(np.isnan(gain)): return [-np.inf]*7,[np.inf]*7
    if type(config).__name__ == 'ndarray':
        param_min = [0., 0.,gain[0]-10*gain[1] , baseline[0]-3*sig[0], 1.e-4,1.e-4,0.]
        param_max = [np.inf, 1, gain[0]+10*gain[1], baseline[0]+3*sig[0],10., 10.,np.inf]
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

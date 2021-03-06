import numpy as np
import utils.pdf

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

    if config==None:

        mu = mu_xt = gain = baseline = sigma_e = sigma_1 = amplitude = offset = np.nan
        param = [mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset]

    else:
        mu = config[0, 0]
        mu_xt = config[1, 0]
        gain = config[2, 0]
        baseline = config[3, 0]
        sigma_e = config[4, 0]
        sigma_1 = config[5, 0]
        amplitude = config[6, 0]
        offset = config[7, 0]
        param = [mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset]

    if type(config).__name__ != 'ndarray':
        raise ValueError('The config parameter is mandatory')

    #print(x)
    param[0] = np.average(x-param[3], weights=y) / param[2]
    #print(param[0])
    param[4] = np.sqrt(np.average((x - np.average(x, weights=y))**2, weights=y))/ param[2]
    param[5] = param[4]
    param[6] = np.sum(y)

    #print(param)

    return param


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

    param_min = [0., 0., 0., -np.inf, 0., 0., 0., -np.inf]
    param_max = [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    #print(param_min, param_max)

    return param_min, param_max


def fit_func(p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """

    [mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset] = p
    sigma = sigma_e * gain
    x = x
    return amplitude * utils.pdf.gaussian(x, sigma,  mu * (1+mu_xt) * gain + baseline)

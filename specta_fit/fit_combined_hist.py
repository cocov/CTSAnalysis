import numpy as np
import utils.pdf
import peakutils
import matplotlib.pyplot as plt
__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def p0_func(y, x, *args, config=None, **kwargs):

    param = config
    n_peak = int(len(param)/3)
    param = param.reshape(n_peak, 3)


    slice = np.arange(np.where(y>0)[0][0],np.where(y>0)[0][-1],1)
    #slice = np.arange(0,len(x),1)
    print(slice)
    y = y[slice]
    x = x[slice]
    log_func = - np.diff(np.log(y))/np.diff(x)

    threshold = 0.6
    min_dist = 2

    print(len(x))
    print(len(y))

    plt.figure()
    plt.plot(x,y)
    plt.plot(x[0:-1], log_func)

    peak_index = peakutils.indexes(log_func, threshold, min_dist) - 1
    peak_index = peak_index[0:n_peak:1]
    print(len(peak_index))
    photo_peak = np.arange(0, len(peak_index), 1)

    gain, baseline = np.polynomial.polynomial.polyfit(photo_peak, x[peak_index], deg=1, w=1./np.sqrt(y[peak_index]))

    plt.plot(x[peak_index], y[peak_index], linestyle='None', marker='o')


    plt.show()

    for i in range(int(min(len(photo_peak), param.shape[0]))):

        param[i, 0] = y[peak_index[i]]
        param[i, 1] = x[peak_index[i]]
        param[i, 2] = gain / 2.

    if len(photo_peak) < param.shape[0]:

        param[len(photo_peak):param.shape[0]:1, 0] = param[len(photo_peak)-1, 0]
        param[len(photo_peak):param.shape[0]:1, 1] = param[len(photo_peak)-1, 0]
        param[len(photo_peak):param.shape[0]:1, 2] = param[len(photo_peak)-1, 0]

    param = param.ravel()

    print(param)




    return param


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def bounds_func(y, x, *args, config=None, **kwargs):

    param = config
    n_peak = len(param)/3

    param_min = np.zeros((int(n_peak), 3))
    param_max = np.ones((int(n_peak), 3)) * np.inf

    param_min[:,2] += 0.001
    param_max[:,0] = np.sum(y)
    param_max[:,1] = np.max(x)

    param_min = param_min.ravel()
    param_max = param_max.ravel()

    return param_min, param_max


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def slice_func(x, *args, **kwargs):
    if np.where(x != 0)[0].shape[0] == 0:
        return [0, 1, 1]
    return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


# noinspection PyUnusedLocal
def fit_func(param, x):

    return utils.pdf.gaussian_sum(param, x)

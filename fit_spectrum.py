from utils.histogram import histogram
import utils.pdf
import peakutils
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def p0_func(y, x, config=None, *args, **kwargs):

    threshold = 0.005
    min_dist = 4
    peaks_index = peakutils.indexes(y, threshold, min_dist)

    if len(peaks_index) == 0:
        return [np.nan] * 7
    amplitude = np.sum(y)
    offset, gain = np.polynomial.polynomial.polyfit(np.arange(0, peaks_index.shape[-1], 1), x[peaks_index], deg=1,
                                                    w=(np.sqrt(y[peaks_index])))
    sigma_start = np.zeros(peaks_index.shape[-1])
    for i in range(peaks_index.shape[-1]):

        start = max(int(peaks_index[i] - gain // 2), 0)  ## Modif to be checked
        end = min(int(peaks_index[i] + gain // 2), len(x))  ## Modif to be checked
        if start == end and end < len(x) - 2:
            end += 1
        elif start == end:
            start -= 1

        # print(start,end,y[start:end])
        if i == 0:
            mu = -np.log(np.sum(y[start:end]) / np.sum(y))

        temp = np.average(x[start:end], weights=y[start:end])
        sigma_start[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))

    bounds = [[0., 0.], [np.inf, np.inf]]
    sigma_n = lambda x, y, n: np.sqrt(x ** 2 + n * y ** 2)
    sigma, sigma_error = curve_fit(sigma_n, np.arange(0, peaks_index.shape[-1], 1), sigma_start, bounds=bounds)
    sigma = sigma / gain

    mu_xt = np.mean(y) / mu / gain - 1

    # print([mu, mu_xt, gain, offset, sigma[0], sigma[1]], amplitude)

    if config:

        if 'baseline' in config:

            offset = config['baseline']

        if 'gain' in config:

            gain = config['gain']

        return [mu, mu_xt, gain, offset, amplitude]

    #print (gain)
    #print (mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude)

    return [mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude]


def bound_func(y, x, config=None, *args, **kwargs):

    if config:

        param_min = [0., 0., 0., 0., 0.]
        param_max = [np.mean(y), 1, np.inf, np.inf, np.sum(y) + np.sqrt(np.sum(y))]


    else:

        param_min = [0., 0., 0., 0., 0., 0., 0.]
        param_max = [np.mean(y), 1, np.inf, np.inf, np.inf, np.inf, np.sum(y) + np.sqrt(np.sum(y))]

    return (param_min, param_max)


def slice_func(y, x, config=None, *args, **kwargs):
    if np.where(y != 0)[0].shape[0] == 0: return [0, 1, 1]
    return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]


if __name__ == '__main__':

    data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 11, 7, 11, 2, 3, 2, 1, 8, 20, 28, 42,
                      47, 27, 15, 10,
                      11, 16, 23, 60, 94, 106, 98, 61, 34, 13, 14, 20, 59, 108, 164, 196, 181, 79, 41, 32, 20, 54, 111,
                      193, 210, 215,
                      185, 113, 64, 42, 61, 100, 158, 206, 234, 251, 188, 129, 74, 62, 73, 102, 155, 218, 245, 202, 164,
                      133, 80, 51, 81,
                      111, 127, 182, 219, 195, 124, 77, 79, 65, 66, 86, 119, 129, 148, 107, 108, 87, 59, 42, 59, 62, 91,
                      73, 100, 106,
                      67, 59, 50, 33, 36, 39, 50, 53, 49, 51, 53, 35, 27, 25, 22, 36, 35, 32, 29, 22, 21, 14, 17, 12,
                      16, 12, 21, 18, 22,
                      11, 7, 5, 7, 11, 4, 9, 8, 3, 6, 10, 8, 5, 4, 2, 4, 4, 4, 5, 6, 2, 3, 7, 5, 2, 2, 2, 3, 4, 3, 1, 2,
                      2, 2, 1, 1, 1,
                      1, 1, 1, 1, 0, 1, 0, 1, 0]], dtype='float32')


    bin = np.arange(0, data.shape[1], 1)

    #bin, data = mpe_fit.mpe_distribution(mean_cherenkov_photon=500, normalized=False)

    data = data.reshape((1,data.shape[-1]))

    mpe = histogram(data, bin_centers=np.arange(0, data.shape[1], 1))
    #mpe.predef_fit()

    #fit_function = mpe.fit_function
    #parameters = mpe.fit_result

    #mpe.show(show_fit=True)

    #config = [{'baseline': 20, 'gain': 5.6}]
    config = None

    def reduced_mpe(p,x, config=None):

        new_p = [p[0], p[1], 9.7, 20., 0.09, 0.09, p[2]]

        return utils.pdf.mpe_distribution_general(new_p, x, config=config)


    #mpe.fit(reduced_mpe, p0_func, slice_func, bound_func, config=config)
    mpe.fit(utils.pdf.mpe_distribution_general, p0_func, slice_func, bound_func, config=config)

    mpe.show(show_fit=True)

    plt.show()

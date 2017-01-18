import numpy as np
import utils.histogram
import copy
import matplotlib.pyplot as plt
import scipy.optimize
import numpy.linalg
from specta_fit import fit_low_light

from specta_fit import fit_dark, fit_combined_hist
import sklearn.neighbors
import sklearn.mixture
import peakutils

def analyze_gain_shift(histo):

    return

def sigma_fit(x, y, y_err):

    param = np.array([x[0], np.sqrt(x[1]**2 - x[0]**2)])
    lower_bound = np.zeros(param.shape)
    upper_bound = np.zeros(param.shape)

    lower_bound[0] = 0.
    lower_bound[1] = 0.
    upper_bound[0] = np.inf
    upper_bound[1] = np.inf


    def residual(param, x, y, y_err):

        return (y - np.sqrt(param[0]**2 + x*param[1]**2)) / y_err

    fit_result = scipy.optimize.least_squares(residual, x0=param, args=(x, y, y_err), bounds=(lower_bound, upper_bound))

    #print(fit_result)

    return fit_result


def gaussian_sum_fit(x, y, n_peaks):

    param = np.ones((n_peaks, 3))
    lower_bound = np.zeros(param.shape)
    upper_bound = np.zeros(param.shape)

    x = x[y>0]
    y = y[y>0]
    log_func = - np.diff(np.log(y)) / np.diff(x)
    y_err = np.sqrt(y)
    y_err[y_err==0] = 1

    threshold = 0.05
    min_dist = 3
    peak_index = peakutils.indexes(log_func, threshold, min_dist) - 1
    #peak_index = peakutils.indexes(y, threshold, min_dist)
    if len(peak_index)==n_peaks:
        print('check # peaks : Ok !!!')
    else:
        print('detected number = of peaks ', len(peak_index), ' is different from set number of peaks = ', n_peaks)
    photo_peak = np.arange(0, len(peak_index), 1)
    gain, baseline = np.polyfit(photo_peak, x[peak_index], deg=1, w=1./y_err[peak_index])

    param[:, 0] = y[peak_index]# amplitude
    param[:, 1] = x[peak_index]# mean
    param[:, 2] = gain/2. # sigma
    lower_bound[:, 0] = param[:, 0] - 1. * np.sqrt(param[:, 0])
    lower_bound[:, 1] = x[peak_index] - gain/2.
    lower_bound[:, 2] = 0.5
    upper_bound[:, 0] = np.inf
    upper_bound[:, 1] = x[peak_index] + gain/2.
    upper_bound[:, 2] = gain/2.

    param = param.ravel()
    lower_bound = lower_bound.ravel()
    upper_bound = upper_bound.ravel()



    def residual(param, x, y, y_err):

        return (y - utils.pdf.gaussian_sum(param, x)) / y_err
        #return (y - utils.pdf.gaussian_sum_1gain(param, x)) / y_err


    fit_result = scipy.optimize.least_squares(residual, x0=param, args=(x, y, y_err), bounds=(lower_bound, upper_bound))

    #print(fit_result)

    return fit_result

def gaussian_sum_fit_1gain(x, y):


    threshold = 0.05
    min_dist = 3

    x = x[y > 0]
    y = y[y > 0]

    log_func = - np.diff(np.log(y)) / np.diff(x)

    peak_index = peakutils.indexes(log_func, threshold, min_dist) - 1
    #peak_index = peakutils.indexes(y, threshold, min_dist)
    photo_peak = np.arange(0, len(peak_index), 1)
    n_peaks = len(peak_index)

    n_param = 4 + n_peaks # baseline, gain, sigma_e, sigma_1, amplitudes (n_peak times)
    param = np.zeros(n_param)
    lower_bound = np.zeros(param.shape)
    upper_bound = np.zeros(param.shape)




    gain, baseline = np.polyfit(photo_peak, x[peak_index], deg=1, w=1./y_err[peak_index])


    param[0] = baseline
    param[1] = gain
    param[2] = gain/2. # sigma
    param[3] = gain/2. # sigma
    param[4:n_param+1] = y[peak_index]

    print(param)

    lower_bound[0] = np.min(x)
    lower_bound[1] = 0.
    lower_bound[2] = 0. # sigma_e
    lower_bound[3] = 0. # sigma_1
    lower_bound[4:n_param+1] = 0.

    upper_bound[0] = np.max(x)
    upper_bound[1] = gain * 1.5
    upper_bound[2] = gain/2. # sigma_e
    upper_bound[3] = gain/2. # sigma_1
    upper_bound[4:n_param+1] = np.inf

    def residual(param, x, y, y_err):

        return (y - utils.pdf.gaussian_sum_1gain(param, x)) / y_err


    fit_result = scipy.optimize.least_squares(residual, x0=param, args=(x, y, y_err), bounds=(lower_bound, upper_bound))

    print(fit_result)

    return fit_result


if __name__ == '__main__':

    #path = 'data/new/'
    #file = 'mpe_scan_0_195_5_200_600_10.npz'

    path = 'data/20161214/'
    file = 'mpe_scan_0_195_5.npz'

    k = np.load(path + file)
    data = copy.copy(k)
    y = data['mpes']
    x = data['mpes_bin_centers']

    y = np.sum(y, axis=0)
    #level = 31
    pixel = 700
    #y = y[level,pixel]
    y = y[pixel]


    start = int(np.where(y>0)[0][0])
    end = int(np.where(y>0)[0][-1])
    end = int(np.where(x==2200)[0][-1]) # 4 peaks = 2037, 15 peaks = 2097 , 25 peaks = 2157 ADC, 31 peaks = 2200
    n_peaks = 31

    #print(y[np.isnan(y)])
    #print(y[np.isinf(y)])

    print((np.max(y) - np.min(y)) + np.min(y))

    x = x[start:end]
    y = y[start:end]

    #print(y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=np.sqrt(y), linestyle='None', marker='o')
    ax.set_yscale('log')
    ax.set_xlabel('ADC')
    ax.set_ylabel('entries')
    ax.set_ylim((1, np.max(y) + np.sqrt(np.max(y))))


    histo = utils.histogram.histogram(data=y, bin_centers=x)

    photo_peak = np.arange(0, n_peaks, 1)

    fit_result = gaussian_sum_fit(x, y , n_peaks=n_peaks)
    xx = np.linspace(x[0], x[-1], num=1000)
    ax.plot(xx, utils.pdf.gaussian_sum(fit_result.x, xx))

    param = fit_result.x.reshape((n_peaks, 3))
    param_error = np.sqrt(np.diag(numpy.linalg.inv(np.dot(fit_result.jac.T, fit_result.jac)))).reshape((n_peaks, 3))

    #print(param_error)


    fit_result_sigma = sigma_fit(photo_peak, param[:,2], param_error[:,2])
    param_sigma = fit_result_sigma.x
    param_sigma_error = np.sqrt(np.diag(numpy.linalg.inv(np.dot(fit_result_sigma.jac.T, fit_result_sigma.jac))))


    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.errorbar(photo_peak, param[:,1], yerr=param_error[:,1],  label='mean', linestyle='None', marker='o')
    ax2.errorbar(photo_peak, param[:,2], yerr=param_error[:,2], label='sigma', linestyle='None', marker='o')
    temp = np.sqrt(np.array([param_error[i,1]**2 + param_error[i+1,1]**2  for i in range(param_error.shape[0]-1)]))
    ax3.errorbar(photo_peak[0:-1], np.diff(param[:,1]), yerr=temp, label='gain', linestyle='None', marker='o')
    ax2.plot(photo_peak, np.sqrt(param_sigma[0]**2 + photo_peak*param_sigma[1]**2), label='model')
    ax1.set_ylabel('photo peak [ADC]')
    ax2.set_xlabel('photo peak [p.e.]')
    ax2.set_ylabel('$\sigma_n$ [ADC]')
    ax3.set_ylabel('Gain [ADC/p.e.]')
    ax3.set_xlabel('photo peak [p.e.]')

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')


    fit_result_1gain_sigma_model = gaussian_sum_fit_1gain(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=np.sqrt(y), linestyle='None', marker='o')
    ax.set_yscale('log')
    ax.set_xlabel('ADC')
    ax.set_ylabel('entries')
    ax.set_ylim((1, np.max(y) + np.sqrt(np.max(y))))
    ax.plot(xx, utils.pdf.gaussian_sum_1gain(fit_result_1gain_sigma_model.x, xx))

    #print(fit_result_1gain_sigma_model)

    param_fit_low_light = np.array([7.04406566, 6.57244174e-02, 5.54263598e+00, 2.01466292e+03, 1.58933853e-01, 7.96533664e-02, 4.72556217e+03, 0.00000000e+00])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=np.sqrt(y), linestyle='None', marker='o', color='k')
    ax.plot(xx, utils.pdf.gaussian_sum_1gain(fit_result_1gain_sigma_model.x, xx), label='1 gain and sigma model')
    ax.plot(xx, utils.pdf.gaussian_sum(fit_result.x, xx), label='all free')
    #ax.plot(xx, fit_low_light.fit_func(param_fit_low_light, xx), label='fit low light')
    ax.legend(loc='best')


    plt.show()
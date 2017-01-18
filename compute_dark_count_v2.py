import matplotlib.pyplot as plt
import matplotlib
import read_trace
import peakutils
import scipy.stats
import utils.histogram
import utils.pulse_shape
import utils.pdf
import numpy as np
import scipy.optimize

filename_pulse_shape = '../../calibration/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file

def compute_dark_run_param_alternativ(histo, event_type='dark_run', pixel_list=None, config=None):

    if event_type == 'dark_run':

        gain = np.zeros(histo.data.shape[0])
        gain_error = np.zeros(histo.data.shape[0])
        sigma_e = np.zeros(histo.data.shape[0])
        sigma_e_error = np.zeros(histo.data.shape[0])
        electronic_baseline = np.zeros(histo.data.shape[0])
        electronic_baseline_error = np.zeros(histo.data.shape[0])
        dark_baseline = np.zeros(histo.data.shape[0])
        dark_baseline_error = np.zeros(histo.data.shape[0])
        dark_rate = np.zeros(histo.data.shape[0])
        dark_rate_error = np.zeros(histo.data.shape[0])
        dark_crosstalk = np.zeros(histo.data.shape[0])
        dark_crosstalk_error = np.zeros(histo.data.shape[0])

        print(histo.data.shape)

        indices_list = np.ndindex(histo.data.shape[0])

        print(indices_list)

        if config:

            print(config['cross_talk'])

            dark_crosstalk = config['cross_talk']
            dark_crosstalk_error = config['cross_talk_error']

        print('#################### dark run : ')

        for index in indices_list:

            #print(index[0])
            if index[0] not in pixel_list:
                continue

            #print(index)
            x = histo.bin_centers
            y = histo.data[index]
            x = x[y>0]
            y = y[y>0]
            print(x)
            print(y)

            plt.figure()
            plt.semilogy(x,y)

            #electronic_baseline[index] = x[np.argmax(y)[0]]
            #electronic_baseline_error[index] = 0.
            log_func = -np.diff(np.log(y)) / np.diff(x)
            threshold = 0.8
            min_dist = 2
            peak_spectrum = peakutils.indexes(log_func, threshold, min_dist)
            peak_spectrum = peak_spectrum - 1
            peak_spectrum = peak_spectrum[y[peak_spectrum]>10]

            plt.plot(x[peak_spectrum], y[peak_spectrum], linestyle='None', marker='o')
            plt.legend()

            [gain[index], electronic_baseline[index]], errors = np.polyfit(np.arange(0, len(x[peak_spectrum]) + 1, 1),
                                                            np.append(x[peak_spectrum], x[peak_spectrum][-1] + 5),
                                                            deg=1, cov=True,
                                                            w=np.append(np.sqrt(x[peak_spectrum]), 0.00001))

            param, cov = scipy.optimize.curve_fit(utils.pdf.gaussian, xdata=x[0:peak_spectrum[0]+int(gain[index]//2.)], ydata=y[0:peak_spectrum[0]+int(gain[index]//2.)], p0=[gain[index]/2., electronic_baseline[index], x[peak_spectrum[0]]], sigma=np.sqrt(y[0:peak_spectrum[0]+int(gain[index]//2.)]), bounds=(0, [gain[index], np.max(x), np.sum(y)]))

            print(param)

            print(errors)
            gain_error[index] = np.sqrt(errors[0, 0])
            electronic_baseline[index] =param[1]
            electronic_baseline_error[index] =  np.sqrt(np.diag(cov)[1])
            sigma_e[index] =  param[0]
            sigma_e_error[index] =  np.sqrt(np.diag(cov)[0])




            dark_baseline[index] = np.average(x, weights=y)
            dark_baseline_error[index] = np.sqrt(np.average((x-dark_baseline[index])**2, weights=y)/(np.sum(y)))

            delta_t = utils.pulse_shape.compute_normalized_pulse_shape_area()

            a = dark_baseline[index] - electronic_baseline[index]  # mean - mode
            a_err = np.sqrt(dark_baseline_error[index] ** 2 + electronic_baseline_error[index] ** 2)
            b = gain[index] * delta_t * (1 + dark_crosstalk[index])
            b_err = np.abs(b) * np.sqrt(
                (gain_error[index] / gain[index]) ** 2 + ((dark_crosstalk_error[index]) / (1 + dark_crosstalk[index])) ** 2)
            dark_rate[index] = a / b
            dark_rate_error[index] = np.abs(dark_rate[index]) * np.sqrt((a_err / a) ** 2 + (b_err / b) ** 2)

            print('########## pixel : ', index)
            print('Dark gain : ', gain[index], ' ± ', gain_error[index], ' [ADC/p.e.]')
            print('Dark baseline : ', dark_baseline[index], ' ± ', dark_baseline_error[index], ' [ADC]')
            print('Electronic baseline : ', electronic_baseline[index], ' ± ', electronic_baseline_error[index], ' [ADC]')
            print('sigma_e : ', sigma_e[index], ' ± ', sigma_e_error[index], ' [ADC]')
            print('Dark crosstalk : ', dark_crosstalk[index], ' ± ', dark_crosstalk_error[index], ' []')
            print('Dark count rate : ', dark_rate[index] * 1E3, ' ± ', dark_rate_error[index] * 1E3, ' [MHz]')



        return np.array([[gain, dark_baseline, electronic_baseline, dark_crosstalk, dark_rate],[gain_error, dark_baseline_error, electronic_baseline_error, dark_crosstalk_error, dark_rate_error]])

    else:

        return

def compute_dark_run_param(histo, event_type='dark_run', pixel_list=None, config=None):

    if event_type == 'dark_run':

        gain = np.zeros(histo.data.shape[0])
        gain_error = np.zeros(histo.data.shape[0])
        sigma_e = np.zeros(histo.data.shape[0])
        sigma_e_error = np.zeros(histo.data.shape[0])
        electronic_baseline = np.zeros(histo.data.shape[0])
        electronic_baseline_error = np.zeros(histo.data.shape[0])
        dark_baseline = np.zeros(histo.data.shape[0])
        dark_baseline_error = np.zeros(histo.data.shape[0])
        dark_rate = np.zeros(histo.data.shape[0])
        dark_rate_error = np.zeros(histo.data.shape[0])
        dark_crosstalk = np.zeros(histo.data.shape[0])
        dark_crosstalk_error = np.zeros(histo.data.shape[0])

        print(histo.data.shape)

        indices_list = np.ndindex(histo.data.shape[0])

        print(indices_list)

        if config:

            print(config['cross_talk'])

            dark_crosstalk = config['cross_talk']
            dark_crosstalk_error = config['cross_talk_error']

        print('#################### dark run : ')

        for index in indices_list:

            #print(index[0])
            if index[0] not in pixel_list:
                continue

            #print(index)
            x = histo.bin_centers
            y = histo.data[index]
            x = x[y>0]
            y = y[y>0]
            print(x)
            print(y)

            plt.figure()
            plt.semilogy(x,y)

            #electronic_baseline[index] = x[np.argmax(y)[0]]
            #electronic_baseline_error[index] = 0.
            log_func = -np.diff(np.log(y)) / np.diff(x)
            threshold = 0.8
            min_dist = 2
            peak_spectrum = peakutils.indexes(log_func, threshold, min_dist)
            peak_spectrum = peak_spectrum - 1
            peak_spectrum = peak_spectrum[y[peak_spectrum]>10]

            plt.plot(x[peak_spectrum], y[peak_spectrum], linestyle='None', marker='o')
            plt.legend()

            [gain[index], electronic_baseline[index]], errors = np.polyfit(np.arange(0, len(x[peak_spectrum]) + 1, 1),
                                                            np.append(x[peak_spectrum], x[peak_spectrum][-1] + 5),
                                                            deg=1., cov=True,
                                                            w=np.append(np.sqrt(x[peak_spectrum]), 0.00001))

            param, cov = scipy.optimize.curve_fit(utils.pdf.gaussian, xdata=x[0:peak_spectrum[0]+int(gain[index]//2.)], ydata=y[0:peak_spectrum[0]+int(gain[index]//2.)], p0=[gain[index]/2., electronic_baseline[index], x[peak_spectrum[0]]], sigma=np.sqrt(y[0:peak_spectrum[0]+int(gain[index]//2.)]), bounds=(0, [gain[index], np.max(x), np.sum(y)]))

            print(param)

            print(errors)
            gain_error[index] = np.sqrt(errors[0, 0])
            electronic_baseline[index] =param[1]
            electronic_baseline_error[index] =  np.sqrt(np.diag(cov)[1])
            sigma_e[index] =  param[0]
            sigma_e_error[index] =  np.sqrt(np.diag(cov)[0])




            dark_baseline[index] = np.average(x, weights=y)
            dark_baseline_error[index] = np.sqrt(np.average((x-dark_baseline[index])**2, weights=y)/(np.sum(y)))

            delta_t = utils.pulse_shape.compute_normalized_pulse_shape_area()

            a = dark_baseline[index] - electronic_baseline[index]  # mean - mode
            a_err = np.sqrt(dark_baseline_error[index] ** 2 + electronic_baseline_error[index] ** 2)
            b = gain[index] * delta_t * (1 + dark_crosstalk[index])
            b_err = np.abs(b) * np.sqrt(
                (gain_error[index] / gain[index]) ** 2 + ((dark_crosstalk_error[index]) / (1 + dark_crosstalk[index])) ** 2)
            dark_rate[index] = a / b
            dark_rate_error[index] = np.abs(dark_rate[index]) * np.sqrt((a_err / a) ** 2 + (b_err / b) ** 2)

            print('########## pixel : ', index)
            print('Dark gain : ', gain[index], ' ± ', gain_error[index], ' [ADC/p.e.]')
            print('Dark baseline : ', dark_baseline[index], ' ± ', dark_baseline_error[index], ' [ADC]')
            print('Electronic baseline : ', electronic_baseline[index], ' ± ', electronic_baseline_error[index], ' [ADC]')
            print('sigma_e : ', sigma_e[index], ' ± ', sigma_e_error[index], ' [ADC]')
            print('Dark crosstalk : ', dark_crosstalk[index], ' ± ', dark_crosstalk_error[index], ' []')
            print('Dark count rate : ', dark_rate[index] * 1E3, ' ± ', dark_rate_error[index] * 1E3, ' [MHz]')



        return np.array([[gain, dark_baseline, electronic_baseline, dark_crosstalk, dark_rate],[gain_error, dark_baseline_error, electronic_baseline_error, dark_crosstalk_error, dark_rate_error]])

    else:

        return

if __name__ == '__main__':

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)

    ### Run parameters ###
    #mc_number = 2 # Monte Carlo run number
    #path = '../../calibration/data_dark/'

    #adc_count, window_time, sampling_time, nsb_rate, mean_crosstalk_production, n_cherenkov_photon = read_trace.read_trace(mc_number=mc_number, path=path)

    path = 'data/DarkRun/20161130/'
    #path = 'data/20161214/'
    filename = 'darkrun_adc_hist.npz'
    #filename = 'spe_hv_on.npz'

    file = np.load(path + filename)
    adcs = utils.histogram.histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))

    path = 'data/20161214/'
    filename = 'adc_hv_off.npz'

    file = np.load(path + filename)
    adcs_hv_off = utils.histogram.histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))


    pixel_list = [700]

    plt.figure()
    plt.plot(adcs_hv_off.bin_centers, adcs_hv_off.data[pixel_list[0]])

    config = {'cross_talk': np.ones(adcs.data.shape[0]) * 0.06, 'cross_talk_error': np.ones(adcs.data.shape[0]) * 0.01}
    param = compute_dark_run_param(adcs, config=config, pixel_list=pixel_list)
    print(param.shape)


    plt.show()

    exit()

    n_forced_trigger = adc_count.shape[0]
    adc_count = adc_count.ravel()
    threshold = np.arange(0., np.max(adc_count), 1) / np.max(adc_count)
    peaks = np.zeros(len(threshold))
    min_dist = 1

    for i in range(len(threshold)):

        #print(peakutils.indexes(adc_count, threshold[i], min_dist))
        peaks[i] = len(peakutils.indexes(adc_count, threshold[i], min_dist))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.step(threshold*np.max(adc_count), peaks, where='mid')
    x = threshold[0:-1]*np.max(adc_count)
    y = -np.diff(np.log(peaks)) / np.diff(threshold*np.max(adc_count))
    ax.step(x, y, where='mid')
    peak_spectrum = peakutils.indexes(y, 0.7, min_dist)

#    dark_crosstalk = peaks[(peak_spectrum[2] - peak_spectrum[1]) // 2] / peaks[(peak_spectrum[1] - peak_spectrum[0]) // 2]
    ax.plot(x[peak_spectrum], y[peak_spectrum], linestyle='None', marker='o')
    plt.xlabel('threshold [ADC]')
    plt.ylabel('# peaks')
    ax.set_yscale('log')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #adc_count = adc_count[peakutils.indexes(adc_count, 0.3, min_dist=1)]
    hist = ax.hist(adc_count, bins=np.arange(np.min(adc_count), np.max(adc_count), 1), align='left')
    x = hist[1][0:-1]
    y = hist[0]
    y_err = np.sqrt(y)
    log_func = -np.diff(np.log(y))/np.diff(x)
    x = x[0:-1] - 1
    ax.plot(x,log_func)
    peak_spectrum = peakutils.indexes(log_func, 0.7, min_dist)
    ax.plot(x[peak_spectrum], y[peak_spectrum], linestyle='None', marker='o')
    plt.xlabel('[ADC]')
    plt.ylabel('#')
    ax.set_yscale('log')

    x[peak_spectrum[0]] = scipy.stats.mode(adc_count)[0]
    [dark_gain, dark_baseline], errors = np.polyfit(np.arange(0, len(x[peak_spectrum]) + 1, 1), np.append(x[peak_spectrum], x[peak_spectrum][-1]+5), deg=1., cov=True, w=np.append(np.ones(len(x[peak_spectrum])), 0.002))
    #dark_gain, dark_baseline, r_value, p_value, errors = scipy.stats.linregress(np.arange(0, len(x[peak_spectrum]), 1), x[peak_spectrum])
    dark_gain_error = np.sqrt(errors[0,0])
    dark_baseline_error = 0 # np.sqrt(errors[1,1])
    dark_baseline = x[peak_spectrum[0]]

    a = (np.sum(y[peak_spectrum]) - y[peak_spectrum][1] - y[peak_spectrum][0])
    a_err = np.sqrt(np.sum(y[peak_spectrum]) + y[peak_spectrum][1] + y[peak_spectrum][0])
    b = np.sum(y[peak_spectrum])
    b_err = np.sqrt(b)
    dark_crosstalk = a / b
    dark_crosstalk_error = np.abs(dark_crosstalk) * np.sqrt((1./a_err)**2 + (1./b_err)**2)

    start = max(int(peak_spectrum[0] - dark_gain // 2), 0)
    end = min(int(peak_spectrum[0] + dark_gain // 2), len(x))
    a = np.sum(y[start:end])
    a_err = np.sqrt(a)
    b = np.sum(y)
    b_err = np.sqrt(b)
    c = a/b
    c_err = np.abs(c) * np.sqrt((1./a_err)**2 + (1./b_err)**2)
    dark_mu = - np.log(c)
    dark_mu_err = np.abs(c_err / c)

    #start = max(int(peak_spectrum[1] - dark_gain // 2), 0)
    #end = min(int(peak_spectrum[1] + dark_gain // 2), len(x))
    #dark_crosstalk = 1 - np.sum(y[start:end])/np.sum(y) / dark_mu / np.exp(-dark_mu)
    #print (peaks[peak_spectrum[1]+dark_gain//2])
    #print (peaks[peak_spectrum[0]+dark_gain//2])
    #dark_crosstalk = peaks[peak_spectrum[1]+dark_gain//2] / peaks[peak_spectrum[0]+dark_gain//2]


    # dark_gain = 5.6
    # dark_gain_err = 0
    # dark_baseline = 10
    # dark_baseline_err = 0
    dark_crosstalk = 0.08
    dark_crosstalk_error = 0


    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
    amplitudes = amplitudes / min(amplitudes)
    delta_t = np.trapz(amplitudes, time_steps)

    mean = np.mean(adc_count)
    mean_err = np.std(adc_count)/np.sqrt(len(adc_count))
    a = mean - dark_baseline # mean - mode
    a_err = np.sqrt(mean_err**2 + dark_baseline_error**2)
    b = dark_gain * delta_t * (1 + dark_crosstalk)
    b_err = np.abs(b) * np.sqrt((dark_gain_error/dark_gain)**2 + ((dark_crosstalk_error)/(1 + dark_crosstalk))**2)
    dark_rate_0 = a / b
    dark_rate_0_err = np.abs(dark_rate_0) * np.sqrt((a_err/a)**2 + (b_err/b)**2)

    dark_rate_1 = dark_mu / delta_t
    dark_rate_1_err = dark_mu_err / np.abs(delta_t)

    a = peaks[peak_spectrum[0]+int(dark_gain//2)]
    a_err = np.sqrt(a)
    b = (window_time * n_forced_trigger)
    b_err = np.abs(window_time) * np.sqrt(n_forced_trigger)
    dark_rate_2 = a / b
    dark_rate_2_err = np.abs(dark_rate_2) * np.sqrt((a_err/a)**2 + (b_err/b)**2)



    print('Dark gain : ', dark_gain, ' ± ', dark_gain_error, ' [ADC/p.e.]')
    print('Dark baseline : ', dark_baseline, ' ± ', dark_baseline_error, ' [ADC]')
    print('Dark crosstalk : ', dark_crosstalk, ' ± ', dark_crosstalk_error, ' []')
    print('Dark mu : ', dark_mu, ' ± ', dark_mu_err, ' [pulse shape area]')
    print('Mean adc : ', mean, ' ± ', mean_err, ' [ADC]')
    print('Dark count rate : ', dark_rate_0 * 1E3, ' ± ', dark_rate_0_err * 1E3, ' [MHz]')
    print('Dark count rate from first peak : ', dark_rate_1 * 1E3, ' ± ', dark_rate_1_err * 1E3, ' [MHz]')
    print('Dark count rate from Enrico : ', dark_rate_2 * 1E3, ' ± ', dark_rate_2_err * 1E3, ' [MHz]')

    plt.show()

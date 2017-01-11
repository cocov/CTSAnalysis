import numpy as np
from utils.histogram import histogram
import utils.pdf
from specta_fit import fit_dark, fit_low_light, fit_high_light, fit_hv_off
import matplotlib.pyplot as plt
import copy

def fit_consecutive_hist(hist, levels, pixels, config=None, fit_type=None):

    prev_fit_result = None

    for level in levels:

        print("################################# Level", level)

        for pix in pixels:


            if len(levels)<=1:
                y = hist.data[pix]
            else:
                y = hist.data[level, pix]

            x = hist.bin_centers

            fit_result = None

            if level == levels[0]:

                # prev_fit_result = np.array([[0.5, np.nan], [0.08, np.nan], [5.6, np.nan], [2020, np.nan], [0.07, np.nan], [0.09,np.nan], [3500, np.nan], [0., np.nan]])
                prev_fit_result = config

            else:

                prev_fit_result = hist.fit_result[level - 1, pix]

            if np.any(np.isnan(hist.data[level, pix])):
                print('----> Pix', pix, 'abort')
                continue

            if fit_type == 'dark':  ## Dark

                print('----> Pix', pix, 'as dark')

                fit_result = hist._axis_fit((level, pix,),
                                            fit_low_light.fit_func,
                                            fit_low_light.p0_func(y, x, config=prev_fit_result),
                                            slice=fit_low_light.slice_func(y, x, config=prev_fit_result),
                                            bounds=fit_low_light.bounds_func(y, x, config=prev_fit_result),
                                            fixed_param=None)

            elif fit_type == 'low_light':  ## Low

                print('----> Pix', pix, 'low light')

                fit_result = hist._axis_fit((level, pix,),
                                            fit_low_light.fit_func,
                                            fit_low_light.p0_func(y, x, config=prev_fit_result),
                                            slice=fit_low_light.slice_func(y, x, config=prev_fit_result),
                                            bounds=fit_low_light.bounds_func(y, x, config=prev_fit_result),
                                            fixed_param=np.array([[7], [0.]]))
                print(fit_result)


            elif fit_type == 'high_light':  ## High
                print('----> Pix', pix, 'high light')

                fit_result = hist._axis_fit((level, pix,),
                                            fit_low_light.fit_func,
                                            fit_low_light.p0_func(y, x, config=prev_fit_result),
                                            slice=fit_low_light.slice_func(y, x, config=prev_fit_result),
                                            bounds=fit_low_light.bounds_func(y, x, config=prev_fit_result),
                                            fixed_param=np.array([[1, 2, 3, 4, 5, 7], [prev_fit_result[1, 0], prev_fit_result[2, 0], prev_fit_result[3, 0], prev_fit_result[4,0], prev_fit_result[5,0], prev_fit_result[7, 0]]]))

            elif fit_type == 'hv_off':  ## HV off
                print('----> Pix', pix, 'hv off')

                fit_result = hist._axis_fit(pix,
                                            fit_hv_off.fit_func,
                                            fit_hv_off.p0_func(y, x, config=prev_fit_result),
                                            slice=fit_hv_off.slice_func(y, x, config=prev_fit_result),
                                            bounds=fit_hv_off.bounds_func(y, x, config=prev_fit_result),
                                            fixed_param=np.array([[0, 1, 2, 5, 7], [prev_fit_result[0, 0], prev_fit_result[1, 0], prev_fit_result[2, 0], prev_fit_result[5, 0], prev_fit_result[7, 0]]]))

            else:

                print(' Impossible to fit', ' pixel', pix, ' in level ', level, ' !!!!!!!!!!!!!!!')

            if len(levels)<=1:

                hist.fit_result[pix] = fit_result

            else:
                hist.fit_result[level, pix] = fit_result

    #return new_hist

def compute_start_parameter_from_previous_fit(fit_result, type_param=None):

    if type_param=='hv_off':

        return fit_result

    elif type_param=='low_light':

        print('hello')
        print(fit_result.shape)
        print(fit_result[:,:,:,0])
        for i in range(fit_result.shape[2]):
            temp = fit_result[:,:,i,1]
            temp_param = fit_result[:,:,i,0]
            temp[temp<=0] = np.inf
            best_params = np.argmin(temp)
            print(best_params)
            best_params = np.unravel_index(best_params, temp.shape)
            print(best_params)
            print(temp_param[best_params], ' $\pm$ ', temp[best_params])

        #print(fit_result[:,:,:0][best_params])
        #print(np.average(fit_result[:,:,:,0], weights=fit_result[:,:,:,1], axis=1))

def plot_param(hist, pixel=700, param='mu', error_plot=False):

    plt.figure()
    x = 5 * np.arange(0, hist.data.shape[0], 1)
    for j in range(hist.fit_result.shape[2]):

        param_names = ['$\mu$', '$\mu_{XT}$', 'gain', 'baseline', '$\sigma_e$', '$\sigma_1$', 'amplitude', 'offset']
        param_units = ['[p.e.]', '[p.e.]', '[ADC/p.e.]', '[ADC]', '[p.e.]', '[p.e.]', '[]', '[ADC]']
        plt.subplot(3, 3, j+1)

        if error_plot:
            y = hist.fit_result[:, pixel, j, 1]
            plt.plot(x, y, label='mpe fit', marker='o')
            plt.ylabel('$\sigma$' + param_names[j] + ' ' + param_units[j])
        else:

            y = hist.fit_result[:, pixel, j, 0]
            yerr = hist.fit_result[:, pixel, j, 1]
            if j==0:
                deg = int(4)
                #param = np.polyfit(x, y, deg=deg, w=1./yerr)
                #plt.plot(x, np.polyval(param, x), label='polyfit deg : ' + str(deg))
                #print(param)

            plt.errorbar(x, y, yerr=yerr, label='mpe fit', marker='o', linestyle=('None' if j == 0 else '-'))
            plt.plot()
            plt.ylabel(param_names[j] + ' ' + param_units[j])
            plt.legend(loc='best')

    plt.xlabel('DAC')



if __name__ == '__main__':

    data_path = 'data/20161214/'

    file_list = ['adc_hv_off.npz', 'mpe_scan_0_195_5.npz', 'spe_hv_on.npz', 'peaks.npz']

    for file in file_list:

        data = np.load(data_path + file)

        if file=='adc_hv_off.npz': # gaussienne sur baseline (prendre baseline)

            hist_hv_off = histogram(data=data['adcs'], bin_centers=data['adcs_bin_centers'])

        elif file=='mpe_scan_0_195_5.npz': # mpe

            hist_mpe = histogram(data=data['mpes'], bin_centers=data['mpes_bin_centers'])

        elif file=='spe_hv_on.npz': # peak finder sur du dark (prendre gain et sigma_e des fits de ca)

            hist_spe = histogram(data=data['adcs'], bin_centers=data['adcs_bin_centers'])

        elif file=='peaks.npz': #

            hist_peak = histogram(data=data['peaks'], bin_centers=data['peaks_bin_centers'])

    data = np.load('data/new/' + 'mpe_scan_0_195_5_200_600_10.npz')
    hist_new = histogram(data=data['mpes'], bin_centers=data['mpes_bin_centers'])


    print(np.average(np.tile(hist_mpe.bin_centers, hist_mpe.data.shape[0]*hist_mpe.data.shape[1]).reshape(hist_mpe.data.shape), weights=hist_mpe.data).shape)
    levels_hv_off = [0]
    levels_low = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    levels_high = [i for i in range(31, 57, 1)]
    pixels = [700]
    n_param = 8


    #hist_new.fit_result = fit_consecutive_hist(hist_new, levels_low, pixels, config=start_param, fit_type='hv_off')


    hist_hv_off.fit_result = np.zeros((hist_hv_off.data.shape[0], n_param, 2))
    hist_mpe.fit_result = np.zeros((hist_mpe.data.shape[0], hist_mpe.data.shape[1], n_param, 2))
    hist_new.fit_result = np.zeros((hist_new.data.shape[0], hist_new.data.shape[1], n_param, 2))

    start_param = np.array([[0.5, np.nan], [0.08, np.nan], [5.6, np.nan], [2020, np.nan], [0.07, np.nan], [0.09,np.nan], [3500, np.nan], [0., np.nan]])
    fit_consecutive_hist(hist_hv_off, levels_hv_off, pixels, config=start_param, fit_type='hv_off')
    start_param = hist_hv_off.fit_result[pixels[0]]
    print(start_param)
    #fit_consecutive_hist(hist_mpe, levels_low, pixels, config=start_param, fit_type='low_light')
    fit_consecutive_hist(hist_new, levels_low, pixels, config=start_param, fit_type='low_light')
    #compute_start_parameter_from_previous_fit(hist_mpe.fit_result, type_param='low_light')
    #start_param = hist_mpe.fit_result[levels_low[-1], pixels[0]]
    start_param = hist_new.fit_result[levels_low[-1], pixels[0]]
    #fit_consecutive_hist(hist_mpe, levels_high, pixels, config=start_param, fit_type='high_light')
    fit_consecutive_hist(hist_new, levels_high, pixels, config=start_param, fit_type='high_light')


    #plot_param(hist_mpe, pixels[0], param='all', error_plot=False)
    plot_param(hist_new, pixels[0], param='all', error_plot=False)
    #plot_param(hist_new, pixels[0], param='all', error_plot=True)
    plot_param(hist_mpe, pixels[0], param='all', error_plot=True)

    hist_hv_off.show(which_hist=(pixels[0],), show_fit=True, fit_func=fit_hv_off.fit_func)

    #for level in levels_low:
        #hist_mpe.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)
        #hist_new.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)

    for level in levels_high:
        #hist_mpe.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)
        #print(level)
        try:
            hist_new.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)
        except:
            print(level)



plt.show()
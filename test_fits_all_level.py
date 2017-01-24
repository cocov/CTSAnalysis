import numpy as np
from utils.histogram import histogram
import utils.pdf
from specta_fit import fit_dark, fit_low_light, fit_high_light, fit_hv_off
import matplotlib.pyplot as plt
import copy
from kapteyn import kmpfit

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
    mu = np.zeros(x.shape)
    sigma_mu = np.zeros(x.shape)
    x_fit = np.arange(0, 1000, 5)
    #x_fit = x
    y_fit = np.zeros(len(x_fit))
    y_fit_max = np.zeros(len(x_fit))
    y_fit_min = np.zeros(len(x_fit))
    upper_band = np.zeros(len(x_fit))
    lower_band = np.zeros(len(x_fit))
    y_hat = np.zeros(len(x_fit))
    sig_yi = np.zeros(len(x_fit))
    sigma_y = np.zeros(len(x_fit))
    n_sigma = 1

    for j in range(hist.fit_result.shape[2]):

        param_names = ['$\mu$', '$\mu_{XT}$', 'gain', 'baseline', '$\sigma_e$', '$\sigma_1$', 'amplitude', 'offset']
        param_units = ['[p.e.]', '[p.e.]', '[ADC/p.e.]', '[ADC]', '[ADC]', '[ADC]', '[]', '[ADC]']
        plt.subplot(3, 3, j+1)

        if error_plot:

            y = hist.fit_result[:, pixel, j, 0]
            y_err = hist.fit_result[:, pixel, j, 1]
            plt.plot(x, y_err/y, label='data', marker='o', linestyle='None')
            plt.ylabel('$\sigma$ / ' + param_names[j])

        else:

            y = hist.fit_result[:, pixel, j, 0] * (hist.fit_result[:, pixel, 2, 0] if (j==4 or j==5) else 1.)
            yerr = hist.fit_result[:, pixel, j, 1] * (hist.fit_result[:, pixel, 2, 0] if (j==4 or j==5) else 1.)

            if j==0:

                mu = y
                sigma_mu = yerr

                def model(p, x):
                    p0, p1, p2, p3, p4 = p
                    return p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4

                def residuals(p, data):

                    x, y = data
                    p0, p1, p2, p3, p4 = p
                    #return (y - model(p, x))/

                # fit with polyfit and compute symmetric CB

                deg = int(4)
                param, covariance = np.polyfit(x, y, deg=deg, w=1./yerr, cov=True)
                param_err = np.sqrt(np.diag(covariance))
                xx = np.vstack([x_fit ** (deg - i) for i in range(deg + 1)]).T
                yi = np.dot(xx, param)
                C_yi = np.dot(xx, np.dot(covariance, xx.T))
                sig_yi = np.sqrt(np.diag(C_yi))
                y_fit = np.polyval(param, x_fit)
                y_fit_max = np.polyval(param + param_err, x_fit)
                y_fit_min = np.polyval(param - param_err, x_fit)

                # fit with kmpfit module and compute CB

                fit = kmpfit.simplefit(model, param, x, y, err=yerr)
                dfdp = [np.ones(len(x_fit)), x_fit, x_fit ** 2, x_fit ** 3, x_fit ** 4]
                #dfdp = [x_fit ** 4, x_fit**3, x_fit**2, x_fit, 1]

                y_hat, upper_band, lower_band = fit.confidence_band(x_fit, dfdp, 0.68, model, abswei=True)


                # compute CB

                print(' covar matrix kmpfit', fit.covar)
                print(' covar matrix polyfit', covariance)

                #fitobj = kmpfit.Fitter(residual=residuals, data=(x, y))
                #fitobj.fit(params0=param)

                #print("\nFit status kmpfit:")
                #print("====================")
                #print("Best-fit parameters:        ", fitobj.params)
                #print("Asymptotic error:           ", fitobj.xerror)
                #print("Error assuming red.chi^2=1: ", fitobj.stderr)
                #print
                #"Chi^2 min:                  ", fitobj.chi2_min
                #print
                #"Reduced Chi^2:              ", fitobj.rchi2_min
                #print
                #"Iterations:                 ", fitobj.niter
                #print
                #"Number of free pars.:       ", fitobj.nfree
                #print
                #"Degrees of freedom:         ", fitobj.dof

                for j in range(len(fit.params)):
                    for k in range(len(fit.params)):

                        sigma_y +=  dfdp[j] * dfdp[k] * fit.covar[j,k] #covariance[j,k]#

                sigma_y = np.sqrt(sigma_y * fit.rchi2_min)

                plt.plot(x_fit, y_fit, label='best fit', color='red')
                plt.errorbar(x, y, yerr=yerr, label='data', marker='o', linestyle='None')
                #plt.plot(x_fit, y_fit_max, label='polyfit + 1 $\sigma$')
                #plt.plot(x_fit, y_fit_min, label='polyfit - 1 $\sigma$')
                #plt.plot(x_fit, y_hat, label='kmpfit')
                #plt.fill_between(x_fit, upper_band, lower_band, alpha=0.5, facecolor='blue', label='kmpfit confidence level')
                plt.fill_between(x_fit, yi+sig_yi, yi-sig_yi, alpha=0.5, facecolor='red', label='confidence level')
                plt.ylabel(param_names[0] + ' ' + param_units[0])
                print('param ', param , ' ± ', param_err)
                print('kmpfit : ', fit.params , ' ± ', fit.xerror)
                plt.legend(loc='best')

            else:

                plt.errorbar(x, y, yerr=yerr, label='data', marker='o', linestyle='None')
                plt.ylabel(param_names[j] + ' ' + param_units[j])
                plt.legend(loc='best')

    plt.xlabel('DAC')

    plt.figure()
    plt.plot(y_hat, (upper_band - lower_band) / y_hat / 2., label='kmpfit')
    plt.plot(y_fit, sig_yi/y_fit, label='polyfit')
    plt.plot(y_fit, (y_fit_max - y_fit_min)/ y_fit / 2., label='polyfit max-min')
    plt.plot(mu, sigma_mu / mu, label='from mpe fits')
    plt.plot(y_fit, 1./np.sqrt(y_fit), label='Poisson')
    #plt.plot(y_fit, sigma_y / y_fit, label='kpmfit redo')
    plt.xlabel('$\mu$')
    plt.ylabel('${\sigma} / {\mu}$')
    plt.legend(loc='best')


    #plt.figure()
    #plt.errorbar(y_fit, mu, xerr=sig_yi, yerr=sigma_mu)
    #plt.xlabel('incoming light [p.e.]')
    #plt.ylabel('measured light [p.e.]')
    #plt.legend(loc='best')

    plt.figure()
    plt.errorbar(x, mu, yerr=sigma_mu, label='LED calibration', marker='o', linestyle='None', color='k')
    plt.plot(x_fit, y_fit, color='r', label='best fit')
    plt.fill_between(x_fit, y_fit + sig_yi, y_fit - sig_yi, alpha=0.25, facecolor='red', label='1 $\sigma$ confidence level')
    plt.xlabel('DAC')
    plt.ylabel('$\mu$ [p.e.]')
    plt.legend(loc='best')


    plt.figure()
    #plt.plot(x_fit, (upper_band - lower_band) / y_hat / 2., label='kmpfit')
    plt.plot(x_fit, sig_yi / y_fit, label='polyfit')
    #plt.plot(x_fit, (y_fit_max - y_fit_min) / y_fit / 2., label='polyfit max-min')
    #plt.plot(x_fit, sigma_y / y_fit, label='kpmfit redo')
    plt.xlabel('DAC')
    plt.ylabel('${\sigma} / {\mu}$')
    plt.legend(loc='best')



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

    limit_low = 30
    limit_max = 40
    #print(np.average(np.tile(hist_mpe.bin_centers, hist_mpe.data.shape[0]*hist_mpe.data.shape[1]).reshape(hist_mpe.data.shape), weights=hist_mpe.data).shape)
    levels_hv_off = [0]
    levels_low = [i for i in range(0, limit_low, 1)]
    levels_high = [i for i in range(limit_low, limit_max, 1)]
    pixels = [700]
    n_param = 8


    #hist_new.fit_result = fit_consecutive_hist(hist_new, levels_low, pixels, config=start_param, fit_type='hv_off')


    hist_hv_off.fit_result = np.zeros((hist_hv_off.data.shape[0], n_param, 2))
    hist_mpe.fit_result = np.zeros((hist_mpe.data.shape[0], hist_mpe.data.shape[1], n_param, 2))
    hist_new.fit_result = np.zeros((hist_new.data.shape[0], hist_new.data.shape[1], n_param, 2))

    start_param = np.array([[0.5, np.nan], [0.08, np.nan], [5.6, np.nan], [2020, np.nan], [0.7, np.nan], [0.9,np.nan], [3500, np.nan], [0., np.nan]])
    fit_consecutive_hist(hist_hv_off, levels_hv_off, pixels, config=start_param, fit_type='hv_off')
    start_param = hist_hv_off.fit_result[pixels[0]]
    #print(start_param)
    fit_consecutive_hist(hist_mpe, levels_low, pixels, config=start_param, fit_type='low_light')
    #fit_consecutive_hist(hist_new, levels_low, pixels, config=start_param, fit_type='low_light')
    compute_start_parameter_from_previous_fit(hist_mpe.fit_result, type_param='low_light')
    start_param = hist_mpe.fit_result[levels_low[-1], pixels[0]]
    #start_param = hist_new.fit_result[levels_low[-1], pixels[0]]
    fit_consecutive_hist(hist_mpe, levels_high, pixels, config=start_param, fit_type='high_light')
    #fit_consecutive_hist(hist_new, levels_high, pixels, config=start_param, fit_type='high_light')


    plot_param(hist_mpe, pixels[0], param='all', error_plot=False)
    #plot_param(hist_new, pixels[0], param='all', error_plot=False)
    #plot_param(hist_new, pixels[0], param='all', error_plot=True)
    plot_param(hist_mpe, pixels[0], param='all', error_plot=True)

    hist_hv_off.show(which_hist=(pixels[0],), show_fit=True, fit_func=fit_hv_off.fit_func)

    #for level in levels_low:
        #hist_mpe.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)
        #hist_new.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)

    for level in levels_low + levels_high:
        hist_mpe.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)

        if level==31:
            print(hist_mpe.fit_result[level, pixels[0]])
        #print(level)
        #try:
        #    hist_new.show(which_hist=(level, pixels[0], ), show_fit=True, fit_func=fit_low_light.fit_func)
        #except:
        #    print(level)



plt.show()
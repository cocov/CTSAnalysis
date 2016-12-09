#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.io import zfits
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry
from utils.fitting import gaussian_residual,spe_peaks_in_event_list
from utils.plots import pickable_visu
from utils.fitting import multi_gaussian_residual_with0
from utils.fitting import multi_gaussian_with0
import sys
from ctapipe import visualization
from utils.histogram import histogram
import peakutils

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

# Setup configuration
parser.add_option("--cts_sector", dest="cts_sector",
                  help="Sector covered by CTS", default=1,type=int)
parser.add_option("-a", "--use_saved_histo_adcs", dest="use_saved_histo_adcs",action="store_true",
                  help="load the histograms from file", default=False)
parser.add_option("-s", "--use_saved_histo_spe", dest="use_saved_histo_spe",action="store_true",
                  help="load the histograms from file", default=False)

parser.add_option("-c", "--perform_adc_fit", dest="perform_adc_fit",action="store_false",
                  help="perform fit of adcs", default=True)

parser.add_option("-e", "--perform_spe_fit", dest="perform_spe_fit",action="store_false",
                  help="perform fit of spes", default=True)

# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="input filenames separated by ','", default='86' )
parser.add_option( "--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")
parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161130/")


parser.add_option( "--calibration_filename", dest="calibration_filename",
                  help="calibration file name", default="darkrun_calib.npz")
parser.add_option( "--calibration_directory", dest="calibration_directory",
                  help="calibration file directory", default="/data/datasets/CTA/DarkRun/20161130/")

parser.add_option( "--saved_histo_directory", dest="saved_histo_directory",
                  help="directory of histo file", default='/data/datasets/CTA/LevelScan/20161130/')
parser.add_option( "--saved_adc_histo_filename", dest="saved_adc_histo_filename",
                  help="name of histo file", default='darkrun_adc_hist.npz')
parser.add_option( "--saved_spe_histo_filename", dest="saved_spe_histo_filename",
                  help="name of histo file", default='darkrun_adc_hist.npz')

parser.add_option( "--saved_adc_fit_filename", dest="saved_adc_fit_filename",
                  help="name of adc fit file", default='darkrun_adc_fit.npz')
parser.add_option( "--saved_spe_fit_filename", dest="saved_spe_fit_filename",
                  help="name of spe fit file", default='darkrun_spe_fit.npz')

# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')


# Define Geometry
sector_to_angle = {1:0.,2:120.,3:240.} #TODO check and put it in cts
cts = cts.CTS('/data/software/CTS/config/cts_config_%d.cfg'%(sector_to_angle[options.cts_sector]),
              '/data/software/CTS/config/camera_config.cfg',
              angle=sector_to_angle[options.cts_sector], connected=False)
geom,good_pixels = generate_geometry(cts)
# Leave the hand
plt.ion()


adcs = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))
spes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))


peaks_index = peakutils.indexes(y, threshold, min_dist)

if not options.use_saved_histo_adc:
    for file in options.file_list:
        # Open the file
        _url = options.directory+options.file_basename%(file)
        inputfile_reader = zfits.zfits_event_source(
            url=_url
            , data_type='r1', max_events=100000)

        if options.verbose: print('--|> Moving to file %s'%(_url))
        # Loop over event in this file
        for event in inputfile_reader:
            for telid in event.r1.tels_with_data:
                if options.verbose and (event.r1.event_id) % 100 == 0:
                    print("Progress {:2.1%}".format(event.r1.event_id/10000), end="\r")
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # fill with a batch of n_sample
                adcs.fill_with_batch(data)

    if options.verbose : print('--|> Save the data in %s' % (options.saved_histo_directory+options.saved_histo_filename))
    np.savez_compressed(options.saved_histo_directory+options.saved_histo_filename,
                        adcs=adcs.data, adcs_bin_centers=adcs.bin_centers)
else:
    if options.verbose: print(
        '--|> Recover data from %s' % (options.saved_histo_directory + options.saved_histo_filename))
    file = np.load(options.saved_histo_directory + options.saved_histo_filename)
    adcs = histogram(data=file['acs'], bin_centers=file['adcs_bin_centers'])


if not options.perform_adc_fit:

    def slice_func(y, x, *args,  config=None, **kwargs):
        if np.where(y != 0)[0].shape[0] == 0: return [0, 1, 1]
        xmin = np.where(y != 0)[0].shape[0]
        xmax = np.argmax(y)+2
        return [xmin,xmax 1]

    adcs.predef_fit(type='Gauss',slice_func=slice_func)

    def p0_func(y, x, *args,config=None, **kwargs):

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

        if mu_xt<0.:mu_xt = 0.

        # print([mu, mu_xt, gain, offset, sigma[0], sigma[1]], amplitude)

        if config:

            if 'baseline' in config:
                offset = config['baseline']

            if 'gain' in config:
                gain = config['gain']

            return [mu, mu_xt, gain, offset, amplitude]

        # print (gain)
        # print (mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude)

        return [mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude]


    def bound_func(y, x, *args,config=None,  **kwargs):

        if config:

            param_min = [0., 0., 0., 0., 0.]
            param_max = [np.mean(y), 1, np.inf, np.inf, np.sum(y) + np.sqrt(np.sum(y))]


        else:

            param_min = [0., 0., 0., -np.inf, 0., 0., 0.]
            param_max = [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.sum(y) + np.sqrt(np.sum(y))]

        return param_min, param_max


    def slice_func(y, x, config=None, *args, **kwargs):
        if np.where(y != 0)[0].shape[0] == 0: return [0, 1, 1]
        return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]

    # Perform the actual fit
    mpes.fit(mpe_distribution_general,p0_func, slice_func, bound_func,limited_indices=[(0,700,),(1,700,),(2,700,),(3,700,)])

    # Save the parameters
    if options.verbose: print('--|> Save the fit result in %s' % (options.saved_histo_directory + options.saved_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_fit_filename, mpes_fit_results=mpes.fit_result)
else :
    if options.verbose: print('--|> Load the fit result from %s' % (options.saved_histo_directory + options.saved_fit_filename))
    h = np.load(options.saved_histo_directory + options.saved_fit_filename)
    mpes.fit_result = h['mpes_fit_results']
    mpes.fit_function = mpe_distribution_general










if not options.use_saved_histo_spe:
    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % (file)
        inputfile_reader = zfits.zfits_event_source(
            url=_url
            , data_type='r1', max_events=100000)

        if options.verbose: print('--|> Moving to file %s' % (_url))
        # Loop over event in this file
        for event in inputfile_reader:
            for telid in event.r1.tels_with_data:
                if options.verbose and (event.r1.event_id) % 100 == 0:
                    print("Progress {:2.1%}".format(event.r1.event_id / 10000), end="\r")
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # fill with a batch of n_sample
                adcs.fill_with_batch(data)

    if options.verbose: print(
        '--|> Save the data in %s' % (options.saved_histo_directory + options.saved_histo_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_histo_filename,
                        adcs=adcs.data, adcs_bin_centers=adcs.bin_centers)
else:
    if options.verbose: print(
        '--|> Recover data from %s' % (options.saved_histo_directory + options.saved_histo_filename))
    file = np.load(options.saved_histo_directory + options.saved_histo_filename)
    adcs = histogram(data=file['acs'], bin_centers=file['adcs_bin_centers'])










    for event in inputfile_reader:
        n_evt += 1
        if n_evt > max_evt: break
        if (n_evt-n_batch*1000)%10==0:print("Progress {:2.1%}".format(float(n_evt - batch_num*n_batch) / n_batch), end="\r")
        for telid in event.r1.tels_with_data:
            if n_evt%n_batch == 0:
                print('--|> Treating the batch #%d of %d events'%( batch_num,n_batch))
                # Update adc histo
                adcs.fill_with_batch(batch.reshape(batch.shape[0],batch.shape[1]*batch.shape[2]))
                # Update the spe histo
                if calib_unknown:
                    spes.fill_with_batch(spe_peaks_in_event_list(batch, None, None, l=1))
                else:
                    variance = calib['sigma_e'][:,0]
                    variance[np.isnan(variance)]=0
                    baseline = calib['baseline'][:,0]
                    baseline[np.isnan(baseline)]=0
                    if apply_calib:
                        spes.fill_with_batch(spe_peaks_in_event_list(batch, None, variance, l=1))
                    else:
                        spes.fill_with_batch(spe_peaks_in_event_list(batch, baseline, variance, l=1))

                # Reset the batch
                batch = None
                batch_num+=1
                print('--|> Reading  the batch #%d of %d events'%( batch_num,n_batch))

            # Get the data
            data = np.array(list(event.r1.tel[telid].adc_samples.values()))
            # Subtract baseline if calibration exists
            if 'baseline' in calib:
                pedestals = calib['baseline'][:,0].reshape(1296,1)
                pedestals[np.isnan(pedestals)]=0
                if apply_calib : data = data -pedestals
            # Append the data to the batch
            if type(batch).__name__!='ndarray':
                batch = data.reshape(data.shape[0],1,data.shape[1])
            else:
                batch = np.append(batch,data.reshape(data.shape[0],1,data.shape[1]),axis = 1)

    file_name = "histos_subtracted.npz" if apply_calib else "histos.npz"

    print('--|> Save in /data/datasets/DarkRun/%s' % (file_name))
    np.savez_compressed("/data/datasets/DarkRun/"+file_name, adcs=adcs.data ,adcs_bin_centers = adcs.bin_centers, spes = spes.data, spes_bin_centers = spes.bin_centers )
else:
    file_name = "histos_subtracted.npz" if apply_calib else "histos.npz"
    print('--|> Recover data from /data/datasets/DarkRun/%s' % (file_name))
    file = np.load("/data/datasets/DarkRun/"+file_name)
    adcs = histogram(data=file['adcs'],bin_centers=file['adcs_bin_centers'])
    spes = histogram(data=file['spes'],bin_centers=file['spes_bin_centers'])

fit_result = None

if calib_unknown:
    # Define the function getting the initial fit value, the parameters bound and the slice to fit
    def p0_func(x,config):
        norm = np.sum(x)
        mean = np.argmax(x)
        sigma = 0.9
        return [norm,mean,sigma]

    def bound_func(x,xrange,config):
        min_norm, max_norm = 1., np.inf
        min_mean, max_mean = xrange[np.argmax(x) - 2], xrange[np.argmax(x) + 2]
        min_sigma, max_sigma = 1e-6, 1e6
        return ([min_norm, min_mean, min_sigma], [max_norm, max_mean, max_sigma])

    def slice_func(x,config):
        return [max(np.argmax(x) -20,0),np.argmax(x) + 1,1]

    print('--|> Compute baseline and sigma_e from ADC distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = adcs.fit( gaussian_residual , p0_func, slice_func, bound_func)
    calib['baseline'] = fit_result[:, 1]
    calib['sigma_e'] = fit_result[:, 2]
    calib['norm'] = fit_result[:, 0]

    print('--|> Save in /data/datasets/DarkRun/calib.npz')
    np.savez_compressed("/data/datasets/DarkRun/calib.npz", sigma_e=calib['sigma_e'], baseline=calib['baseline'], norm=calib['norm'])


if perform_spe_fit:
    # Define the function getting the initial fit value, the parameters bound and the slice to fit
    def p0_func(x, xrange, config ):
        return [1000000.   ,0.8, 0.1 , 5.6, 100000.,10000.,1000. , config['baseline']      , 0.,0.8      , 100. ,10.]

    def bound_func(x,xrange,config):
        param_min = [0.    ,0.01, 0.01, 0. , 10.   , 1.    , 0.    , config['baseline'] - 10.,-10.,0.8*0.1, 0.    ,0.]
        param_max = [np.inf,8. , 5. , 100., np.inf, np.inf, np.inf, config['baseline'] + 10.,10. ,0.8*10., np.inf,np.inf]
        return (param_min, param_max)

    def slice_func(x,xrange ,config):
        if np.where(x != 0)[0].shape[0]==0: return[0,1,1]
        return [np.where(x != 0)[0][0]+3,np.where(x != 0)[0][-1],1]

    def remap_conf_dict(config):
        new_conf = []
        for i,pix in enumerate(config[list(config.keys())[0]]):
            new_conf.append({})
        for key in list(config.keys()):
            for i,pix in enumerate(config[key]):
                if np.isfinite(pix[0]):
                    new_conf[i][key]=pix[0]
                else:new_conf[i][key]=0.
        return new_conf

    print('--|> Compute Gain, sigma_e, sigma_i from SPE distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = spes.fit(multi_gaussian_with0, p0_func, slice_func, bound_func,config=remap_conf_dict(calib))
    calib_spe ={}
    calib_spe['gain'] = fit_result[:, 3]
    calib_spe['sigma_e_spe'] = fit_result[:, 1]
    calib_spe['sigma_i'] = fit_result[:, 2]
    calib_spe['baseline_spe'] = fit_result[:, 7]
    calib_spe['sigma_e'] = calib['sigma_e']
    calib_spe['baseline'] = calib['baseline']
    calib_spe['norm'] = calib['norm']
    calib_spe['full_spe_fitres']=fit_result
    print('--|> Save in /data/datasets/DarkRun/calib_spe.npz')
    np.savez_compressed("/data/datasets/DarkRun/calib_spe.npz",
                        sigma_e_spe=calib_spe['sigma_e_spe'],
                        sigma_e=calib_spe['sigma_e'],
                        sigma_i=calib_spe['sigma_i'],
                        baseline_spe=calib_spe['baseline_spe'],
                        baseline=calib_spe['baseline'],
                        gain=calib_spe['gain'],
                        norm=calib_spe['norm'],
                        full_spe_fitres = calib_spe['full_spe_fitres'])

    print('--|> Recover baseline and sigma_e from /data/datasets/DarkRun/calib_spe.npz')
    calib = np.load("/data/datasets/DarkRun/calib_spe.npz")


## Plot the baseline extraction
def slice_fun(x,**kwargs):
    return [np.where(x != 0)[0][0],np.where(x != 0)[0][-1],1]

fig1, axs1 = plt.subplots(1, 2,figsize=(30, 10))
plt.subplot(1, 2 ,1)
vis_baseline = pickable_visu([adcs,spes],axs1[1],fig1,slice_fun,calib,apply_calib,geom, title='',norm='lin', cmap='viridis',allow_pick=True)
vis_baseline.add_colorbar()
vis_baseline.colorbar.set_label('Gain [ADC/p.e.]')
plt.subplot(1, 2 ,1)
h = np.copy(calib['gain'][:,0])
h_err = np.copy(calib['gain'][:,1])
h[np.isnan(h_err)]=2
h[h>20]=2

ba =calib['baseline_spe'][:,1]
h[ba>100]=2
h[h<3]=3
h[h>6]=6

vis_baseline.axes.xaxis.get_label().set_ha('right')
vis_baseline.axes.xaxis.get_label().set_position((1,0))
vis_baseline.axes.yaxis.get_label().set_ha('right')
vis_baseline.axes.yaxis.get_label().set_position((0,1))
vis_baseline.image = h
fig1.canvas.mpl_connect('pick_event', vis_baseline._on_pick )
vis_baseline.on_pixel_clicked(374)

plt.subplots(1, 1,figsize=(7.5, 9))

ax= plt.subplot(1, 1 ,1,xlabel='Gain [ADC/p.e.]',ylabel='$\mathrm{N_{pixel}}$')
hh = np.copy(calib['gain'][:,0])
hh_fiterr= np.copy(calib['gain'][:,1])
hh_fin=hh[np.isfinite(hh_fiterr)]
hist ,bin = np.histogram(hh_fin,bins=np.arange(-0.05,10.15,0.1))
gain = histogram( data=hist.reshape(1,hist.shape[0]),bin_centers=np.arange(0.,10.1,0.1))
plt.errorbar(x=gain.bin_centers,y=gain.data[0],yerr = gain.errors[0], fmt = 'o')
gain.predef_fit('Gauss',x_range=[5.,6.],initials=[200.,5.5,0.3])
ax.plot(gain.fit_axis,gain.fit_function(gain.fit_result[0][:,0],gain.fit_axis))
ax.text(0.6,0.9,'$\mu$=%1.3f $\pm$ %1.3f\n$\sigma$=%1.3f $\pm$ %1.3f' % (
        gain.fit_result[0][1][0], gain.fit_result[0][1][1],
        gain.fit_result[0][2][0], gain.fit_result[0][2][1]),
        fontsize=15, transform=ax.transAxes, va='top', )
ax.xaxis.get_label().set_ha('right')
ax.xaxis.get_label().set_position((1,0))
ax.yaxis.get_label().set_ha('right')
ax.yaxis.get_label().set_position((0,1))
plt.show()
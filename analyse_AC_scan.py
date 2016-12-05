#!/usr/bin/env python3
import matplotlib.pyplot as plt
from ctapipe.io.hessio import hessio_event_source

from ctapipe import visualization
from ctapipe.calib import pedestals as pedestal_calib
from ctapipe.calib.camera import integrators
import numpy as np
from ctapipe.io import zfits
from matplotlib.ticker import LogFormatter
# Get CTS
from cts import cameratestsetup as cts
import scipy.stats
from scipy import optimize
from astropy.modeling import models, fitting
from numpy.linalg import inv
from scipy.stats import norm
from numpy import linspace
from utils.geometry import generate_geometry
from utils.fitting import gaussian_residual,spe_peaks_in_event_list
from utils.plots import pickable_visu
from utils.fitting import multi_gaussian_residual
from utils.fitting import multi_gaussian
from utils.fitting import get_poisson_err
from utils.fitting import cleaning_peaks,spe_peaks
from utils.peakdetect import peakdetect
from astropy import units as u
import sys
from utils.histogram import histogram

recompute,calib_unknown,apply_calib,perform_spe_fit,get_calib_from_spe = sys.argv[1]=='True',sys.argv[2]=='True',sys.argv[3]=='True',sys.argv[4]=='True',sys.argv[5]=='True'
if calib_unknown: apply_calib = False


# Define Geometry
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)
geom,good_pixels = generate_geometry(cts,availableBoard={1:[0,1,2,3,4,5,6,7,8,9],2:[],3:[]})
good_pixels_mask = np.repeat(good_pixels, 50).reshape(good_pixels.shape[0], 1, 50)

plt.ion()

# Some object definition
calib = {}
if not calib_unknown:
    if get_calib_from_spe:
        print('--|> Recover baseline and sigma_e from /data/datasets/DarkRun/calib_spe.npz')
        calib = np.load("/data/datasets/DarkRun/calib_spe.npz")
    else:
        print('--|> Recover baseline and sigma_e from /data/datasets/DarkRun/calib.npz')
        calib = np.load("/data/datasets/DarkRun/calib.npz")



adcs = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296))
spes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296))

if 'baseline' in calib and apply_calib:
    adcs = histogram(bin_center_min=-20., bin_center_max=50., bin_width=1., data_shape=(1296))
    spes = histogram(bin_center_min=-20., bin_center_max=50., bin_width=1., data_shape=(1296))


if recompute:
    # Open the file
    the_url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.66.fits.fz"
    inputfile_reader = zfits.zfits_event_source(
        url=the_url
        , data_type='r1', max_events=100000)

    # Creating the histograms

    # Reading the file
    n_evt,n_batch,batch_num,max_evt=0,1000,0,30000

    print('--|> Will process %d events from %s'%(max_evt,the_url))
    batch = None

    print('--|> Reading  the batch #%d of %d events' % (batch_num, n_batch))

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
    def p0_func(x,config):
        return [0.8, 0.0005, 5.6, 10000., 1000., 100., config['baseline'], 10.]

    def bound_func(x,xrange,config):
        param_min = [0.08, 0.0001, 3., 10., 1., 0., config['baseline'] - 10., 0.]
        param_max = [8., 5., 10., np.inf, np.inf, np.inf, config['baseline'] + 10., np.inf]
        return (param_min, param_max)

    def slice_func(x,config):
        return [np.where(x != 0)[0][0]+3,np.where(x != 0)[0][-1],1]


    def my_func(param, x, *args, **kwargs):
        p_new = [0.,
                 param[0],
                 param[1],
                 param[2],
                 param[3],
                 param[4],
                 param[5],
                 param[6],
                 0.,
                 1.,
                 param[7]
                 ]
        return multi_gaussian_residual_with0(p_new, x, *args, **kwargs)

    def remap_conf_dict(config):
        new_conf = []
        for i,pix in enumerate(config[list(config.keys())[0]]):
            new_conf.append({})
        for key in list(config.keys()):
            for i,pix in enumerate(config[key]):
                new_conf[i][key]=pix[0]
        return new_conf

    print('--|> Compute Gain, sigma_e, sigma_i from SPE distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = adcs.fit( my_func , p0_func, slice_func, bound_func,config=remap_conf_dict(calib))
    ## TODO debug the fit
    calib_spe ={}
    calib_spe['gain'] = fit_result[:, 2]

    print(calib['gain'][10])
    calib_spe['sigma_e_spe'] = fit_result[:, 0]
    calib_spe['sigma_i'] = fit_result[:, 1]
    calib_spe['baseline_spe'] = fit_result[:, 6]
    calib_spe['sigma_e'] = calib['sigma_e']
    calib_spe['baseline'] = calib['baseline']
    calib_spe['norm'] = calib['norm']
    print('--|> Save in /data/datasets/DarkRun/calib_spe.npz')
    np.savez_compressed("/data/datasets/DarkRun/calib_spe.npz",
                        sigma_e_spe=calib_spe['sigma_e_spe'],
                        sigma_e=calib_spe['sigma_e'],
                        sigma_i=calib_spe['sigma_i'],
                        baseline_spe=calib_spe['baseline_spe'],
                        baseline=calib_spe['baseline'],
                        gain=calib_spe['gain'],
                        norm=calib_spe['norm'])

print('--|> Recover baseline and sigma_e from /data/datasets/DarkRun/calib_spe.npz')
calib = np.load("/data/datasets/DarkRun/calib_spe.npz")




## Plot the baseline extraction
def slice_fun(x,**kwargs):
    return [np.where(x != 0)[0][0],np.where(x != 0)[0][-1],1]

fig1, (axs1, axs2) = plt.subplots(2, 2)
plt.subplot(2, 2 ,1)
axs1[0].set_title('Camera')
vis_baseline = pickable_visu([adcs,spes],axs1[1],fig1,slice_fun,calib,apply_calib,geom, title='Camera Gains', norm='lin', cmap='viridis',allow_pick=True)
vis_baseline.add_colorbar()

h = calib['gain'][:,0]
print(calib['gain'][10])
h_err = calib['gain'][:,1]
h1= np.where(np.isfinite(h_err), h , 3 )
vis_baseline.image = h1
fig1.canvas.mpl_connect('pick_event', vis_baseline._on_pick )


plt.subplot(2, 2 ,3)
axs2[0].set_title('Gain')
hh = calib['gain'][:,0]
hh_fin=hh[np.isfinite(hh)]
print(hh_fin)
axs2[0].hist(hh,bins=50)



plt.subplot(2, 2 ,4)
axs2[1].set_title('$\sigma_e$')
hh1 = calib['sigma_e_spe'][:,0]
hh1_fin=hh[np.isfinite(hh1)]
axs2[1].hist(hh1,bins=50)



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
from utils.fitting import gaussian_residual
from utils.fitting import multi_gaussian_residual
from utils.fitting import multi_gaussian
from utils.fitting import get_poisson_err
from utils.fitting import cleaning_peaks,spe_peaks
from utils.peakdetect import peakdetect
from astropy import units as u
import sys
from utils.histogram import histogram

recompute,recalib,refit = sys.argv[1]=='True',sys.argv[2]=='True',sys.argv[3]=='True'

# Define Geometry
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)
geom,good_pixels = generate_geometry(cts,availableBoard={1:[0,1,2,3,4,5,6,7,8,9],2:[],3:[]})
good_pixels_mask = np.repeat(good_pixels, 50).reshape(good_pixels.shape[0], 1, 50)

plt.ion()

# Some object definition
calib = {}
if not recalib:
    calib = np.load("/data/datasets/DarkRun/calib.npz")



adcs = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296))
spes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296))

if 'baseline' in calib:
    adcs = histogram(bin_center_min=0., bin_center_max=40., bin_width=1., data_shape=(1296))
    spes = histogram(bin_center_min=0., bin_center_max=40., bin_width=1., data_shape=(1296))


if recompute:
    # Open the file
    inputfile_reader = zfits.zfits_event_source(
        url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.66.fits.fz"
        , data_type='r1', max_events=100000)

    # Creating the histograms

    # Reading the file
    print('Reading the file')
    n_evt,n_batch=0,1000
    for event in inputfile_reader:
        n_evt += 1
        if n_evt > 2000: break
        for telid in event.r1.tels_with_data:
            if n_evt%n_batch == 0:
                print('treating the batch',n_evt/n_batch)
                adcs.fill_with_batch(batch.reshape(batch.shape[0],batch.shape[1]*batch.shape[2]))

            # Get the data
            data = np.array(list(event.r1.tel[telid].adc_samples.values()))
            # Subtract baseline if calibration exists
            if 'baseline' in calib:
                data = np.subtract(data, calib['baseline'])
            # Update adc histo
            adcs.fill_with_batch(data)
            # Update the spe histo
            spes.fill_with_batch(spe_peaks(data,l=1))

    np.savez_compressed("/data/datasets/DarkRun/histos.npz", adcs=adcs.data ,adcs_bin_centers = adcs.bin_centers, spe = spes.data, spe_bin_centers = spes.bin_centers )
else:
    file = np.load("/data/datasets/DarkRun/histos.npz")
    adcs = histogram(data=file['adcs'],bin_centers=file['adcs_bin_centers'])
    spes = histogram(data=file['spes'],bin_centers=file['spes_bin_centers'])


if recalib:

    # Define the function getting the initial fit value, the parameters bound and the slice to fit
    def p0_func(x):
        norm = np.sum(x)
        mean = scipy.stats.mode(x).mode[0]
        sigma = 0.9
        return [norm,mean,sigma]

    def bound_func(x,xrange):
        min_norm, max_norm = 1., np.inf
        min_mean, max_mean = xrange[scipy.stats.mode(x).mode[0] + 2], xrange[scipy.stats.mode(x).mode[0] + 2]
        min_sigma, max_sigma = 1e-6, 1e6
        return ([min_norm, min_mean, min_sigma], [max_norm, max_mean, max_sigma])

    def slice_func(x):
        return [0,scipy.stats.mode(x).mode[0] + 1,1]

    # Fit the baseline and sigma e of all pixels
    adcs.fit( gaussian_residual , p0_func, slice_func, bound_func)


"""
    fit_result = np.apply_along_axis(fit_baseline_hist, 1, adcs)
    calib['sigma_e'] = fit_result[:, 1]
    calib['baseline'] = fit_result[:, 0]
    np.savez_compressed("/data/datasets/DarkRun/calib.npz", sigma_e=calib['sigma_e'], baseline=calib['baseline'])


"""


'''
#
if redo_spe :

    for i in range(1,6):
        dataset = {}
        dataset['adcs'] = np.load("/data/datasets/DarkRun/darkrun" + str(i) + ".npz")['adcs']
        stacked_adcs = dataset['adcs'].reshape(dataset['adcs'].shape[0],
                                               dataset['adcs'].shape[1] * dataset['adcs'].shape[2])

        fit_result = np.apply_along_axis(fit_baseline, 1, stacked_adcs)
        dataset['sigma_e'] = fit_result[:, 1]
        dataset['baseline'] = fit_result[:, 0]

        for pix, pix_adcs in enumerate(dataset['adcs']):
            if not good_pixels[pix]: continue
            for event_adcs in pix_adcs:
                spe_pixels[pix] += cleaning_peaks(event_adcs, baseline=dataset['baseline'][pix],
                                                  threshold=dataset['sigma_e'][pix] * 3., l=1)
                h,b = np.histogram(stacked_adcs[pix], bins=np.arange(-0.5, 41.5, 1), density=False)
                full_adc[pix]= np.sum(full_adc[pix],h)

    np.savez_compressed("/data/datasets/DarkRun/darkrun_spe.npz", spe=np.array(spe_pixels))



spe_pixels=np.load("/data/datasets/DarkRun/darkrun_spe.npz")['spe']
dataset = {}
dataset['adcs'] = np.load("/data/datasets/DarkRun/darkrun1.npz")['adcs']
stacked_adcs = dataset['adcs'].reshape(dataset['adcs'].shape[0],
                                       dataset['adcs'].shape[1] * dataset['adcs'].shape[2])
fit_result = np.apply_along_axis(fit_baseline, 1, stacked_adcs)
dataset['sigma_e'] = fit_result[:, 1]
dataset['baseline'] = fit_result[:, 0]


X = []
ipix = 0
for pix,good in enumerate(good_pixels):
    if good:
        print(len(spe_pixels[ipix]))
        X.append(spe_pixels[ipix])

        ipix+=1
    else:
        X.append([0.]*50)

print(X[10])

fig, (ax, ax2) = plt.subplots(1, 2)
plt.subplot(1, 2,1)
ax.set_title('click on point to plot time series')
vis = pickable_visu(spe_pixels,ax2,geom, title='Pickable SPE', norm='lin', cmap='coolwarm',allow_pick=True)
vis.image = dataset['sigma_e']
fig.canvas.mpl_connect('pick_event', vis._on_pick )

plt.show()
'''
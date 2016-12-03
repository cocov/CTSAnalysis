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
from utils.peakdetect import peakdetect
from numpy.linalg import inv
from astropy import units as u


# Define Geometry
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)

geom, good_pixels = generate_geometry(cts,availableBoard={1:[0,1,2,3,4,5,6,7,8,9],2:[],3:[]})

# Prepare plots
'''
plt.figure(0)
displayType = {}
for i,title_plot in enumerate(['Baseline','Electronic_Noise']):
    plt.subplot(1, 2, i+1)
    displayType[title_plot]=visualization.CameraDisplay(geom, title=title_plot, norm='lin', cmap='coolwarm',allow_pick=True)
    displayType[title_plot].add_colorbar()
'''
plt.ion()

# some usefull functions
def fit_baseline(data):
    """
    baseline fit
    :param data: pixel adcs
    :return: [sigma,mu]
    """
    if np.isnan(data[0]): return [0.,0.]
    if np.std(data)>200: return [0.,0.]
    mode = scipy.stats.mode(data).mode[0]
    hist , bins = np.histogram(data,bins=np.arange(mode-10.5, mode+2.5, 1),density=False)
    out = optimize.least_squares(gaussian_residual, [data.shape[0], mode, 0.8], args=(np.arange(mode-10., mode+2., 1), hist),
                                 bounds=([0., mode-10., 1.e-6], [1.e8, mode+1., 1.e5]))
    return [out.x[1],out.x[2]]

# some usefull functions
def fit_multigaussian(data):
    """
    baseline fit
    :param data: pixel adcs
    :return: [sigma,mu]
    """
    hist , bins = np.histogram(data,bins=np.arange(3.5, 41.5, 1),density=False)
    out = optimize.least_squares(multi_gaussian_residual, [1000.,0.9,0.9,5.6,100.,10.,1.,5.6], args=(np.arange(4,41, 1), hist))
    return out


def cleaning_peaks( data, baseline, threshold=2., l=2):
    peaks = peakdetect(data, lookahead=l)[0]
    newpeaks = []
    for peak in peaks:
        if peak[0]<2 or peak[0]>47:continue
        if peak[1]<threshold + baseline: continue
        best_position = 0
        if max(data[peak[0]],data[peak[0]-1])==data[peak[0]]:
            best_position = peak[0]
        else :best_position = peak[0]-1
        if max(data[best_position],data[peak[0]+1])==data[peak[0]+1]:
            best_position = peak[0]+1
        newpeaks.append(data[best_position]-baseline)
    return newpeaks

dataset = {}

recompute = True

if recompute:
    ## Now get the events
    inputfile_reader = zfits.zfits_event_source(
        url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.66.fits.fz"
        , data_type='r1', max_events=1000)

    n_evt = 0

    print('Get all data')

    good_pixels_mask = np.repeat(good_pixels, 50).reshape(good_pixels.shape[0], 1, 50)

    for event in inputfile_reader:
        n_evt += 1
        if n_evt > 20000: break
        if n_evt % 100 == 0: print(n_evt)
        for telid in event.r1.tels_with_data:
            data = np.array(list(event.r1.tel[telid].adc_samples.values()))
            ## reshape the data (npix,nevent,nadc)
            adcs = data.reshape(data.shape[0], 1, data.shape[1])
            ## set bad pix to nan
            adcs = np.where(good_pixels_mask, adcs, np.nan)
            ## append to previous events
            if 'adcs' not in dataset:
                dataset['adcs'] = adcs
            else:
                dataset['adcs'] = np.append(dataset['adcs'], adcs, axis=1)

    stacked_adcs = dataset['adcs'].reshape(dataset['adcs'].shape[0],dataset['adcs'].shape[1] * dataset['adcs'].shape[2])
    fit_result = np.apply_along_axis(fit_baseline, 1, stacked_adcs)
    dataset['sigma_e'] = fit_result[:, 1]
    dataset['baseline'] = fit_result[:, 0]
    np.savez_compressed("/data/datasets/darkrun.npz", adcs=dataset['adcs'], sigma_e=dataset['sigma_e'], baseline=dataset['baseline'])

else :
    dataset = np.load("/data/datasets/darkrun.npz")


## Test the peak finding:

print('Go for spe')
spe_pixels = []

redo_spe = True

if redo_spe :
    for pix, pix_adcs in enumerate(dataset['adcs']):
        if not good_pixels[pix]:continue
        print(pix)
        spe_pixels.append([])
        for event_adcs in pix_adcs:
            spe_pixels[-1] += cleaning_peaks(event_adcs, baseline=dataset['baseline'][pix],
                                             threshold=dataset['sigma_e'][pix] * 3.5, l=2)

    np.savez_compressed("/data/datasets/darkrun_spe.npz", spe=np.array(spe_pixels))
else:
    spe_pixels=np.load("/data/datasets/darkrun_spe.npz")['spe']



X = []
ipix = 0
for pix,good in enumerate(good_pixels):
    if good:
        X.append(spe_pixels[ipix])
        ipix+=1
    else:
        X.append([0.]*50)


class pickable_visu(visualization.CameraDisplay):
    def __init__(self,pickable_data,extra_plot,*args, **kwargs):
        super(pickable_visu, self).__init__(*args, **kwargs)
        self.pickable_data = pickable_data
        self.extra_plot = extra_plot

    def on_pixel_clicked(self, pix_id):

        self.extra_plot.cla()
        histfull, bins_full = np.histogram(self.pickable_data[pix_id], bins=np.arange(3.5, 41.5, 1), density=False)

        self.extra_plot.errorbar(np.arange(4., 41., 1), histfull, yerr=np.vectorize(get_poisson_err)(histfull), fmt='o')

        self.extra_plot.set_ylim(1.e-1, np.max(histfull)*2)
        try :
            out = fit_multigaussian(self.pickable_data[pix_id])
            cov = inv(np.dot(out.jac.T,out.jac))
            self.extra_plot.text(0.5,0.9, 'gain=%1.3f $\pm$ %1.3f\n$\sigma_e$=%1.3f $\pm$ %1.3f\n$sigma_i$=%1.3f $\pm$ %1.3f' % (out.x[3], np.sqrt(cov[3][3]) , out.x[1], np.sqrt(cov[1][1]), out.x[2], np.sqrt(cov[2][2])),fontsize=20,transform=self.extra_plot.transAxes, va='top', )
            x_fit = np.linspace(0,40,200)
            self.extra_plot.plot(x_fit,multi_gaussian(out.x,x_fit))
        except Exception:
            print('Fit failed somhow')
        self.extra_plot.set_yscale('log')
        try:
            fig.canvas.draw()
        except ValueError:
            print('some issue to plot')



fig, (ax, ax2) = plt.subplots(1, 2)
plt.subplot(1, 2,1)
ax.set_title('click on point to plot time series')
vis = pickable_visu(X,ax2,geom, title='Pickable SPE', norm='lin', cmap='coolwarm',allow_pick=True)
vis.image = dataset['sigma_e']
fig.canvas.mpl_connect('pick_event', vis._on_pick )

plt.show()
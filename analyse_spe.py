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


# Define Geometry
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)

geom, good_pixels = generate_geometry(cts,availableBoard={1:[0,1,2,3,4,5,6,7,8,9],2:[],3:[]})

# Prepare plots
displayType = []
for title_plot in ['Baseline', 'Electronic noise']:
    displayType.append(visualization.CameraDisplay(geom, title=title_plot, norm='lin', cmap='coolwarm'))
    displayType[-1].add_colorbar()

plt.ion()

# some usefull functions
def fit_baseline(data):
    """
    baseline fit
    :param data: pixel adcs
    :return: [sigma,mu]
    """
    if np.mean(data)<-100 or np.std(data)>200: return 0.
    hist , bins = np.histogram(data,bins=np.arange(-10.5, 2.5, 1),density=True)
    out = optimize.least_squares(gaussian_residual, [1000., 0., 0.1], args=(np.arange(-10., 2., 1), hist),
                                 bounds=([0., -1., 1.e-6], [np.inf, 1., 1.e5]))
    return [out.x[1],out.x[2]]


pedestals = {}



## Now get the events
inputfile_reader = zfits.zfits_event_source(url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.66.fits.fz"
                                                , data_type='r1', max_events=1000)



print('Get all the pedestals')
nevt=0
for event in inputfile_reader:
    nevt+=1
    if nevt>1500/50:break
    for telid in event.r1.tels_with_data :
        data = np.array(list(event.r1.tel[telid].adc_samples.values()))
        # pedestal evaluation
        peds, pedvars = pedestal_calib.calc_pedestals_from_traces(data, 0, 50)
        if 'mean' not in pedestals: pedestals['mean'] = peds.reshape(1296,1)
        else : pedestals['mean'] = np.append(pedestals['mean'], peds.reshape(1296, 1), axis=1)
        # keep all ADCs
        if 'mode' not in pedestals: pedestals['mode']= data
        else: pedestals['mode']=np.append(pedestals['mode'],data,axis = 1)
        

pedestals['mean'] = np.mean(pedestals['mean'], axis=1)
pedestals['mean'] = np.where(pix_badid, pedestals['mean'], np.min(np.extract(pix_badid, pedestals['mean'])))

pedestals['meanUnique']  = np.mean(pedestals['mode'], axis=1)
pedestals['meanUnique']  = np.where(pix_badid, pedestals['meanUnique'], np.min(np.extract(pix_badid, pedestals['meanUnique'])))
pedestals['mode'] =   scipy.stats.mode(pedestals['mode'], axis= 1).mode.reshape(1296)
pedestals['mode'] =   np.where(pix_badid,pedestals['mode'],np.min(np.extract(pix_badid, pedestals['mode'])))
displayType[0].image = pedestals['mean']
displayType[1].image = pedestals['mode']
plt.show()


dataset = {}
inputfile_reader = zfits.zfits_event_source(url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.66.fits.fz"
                                                , data_type='r1', max_events=10000)
nevt = 0

print('Get all data')
for event in inputfile_reader:
    nevt+=1
    if nevt>100: break
    if nevt%100 == 0: print(nevt)
    for telid in event.r1.tels_with_data :
        data = np.array(list(event.r1.tel[telid].adc_samples.values()))
        adcs = np.subtract(data,pedestals['mode'].reshape(1296,1))
        if 'adcs' not in dataset:
            dataset['adcs'] = adcs.reshape(adcs.shape[0],1,adcs.shape[1])
        else:
            dataset['adcs'] = np.append(dataset['adcs'],adcs,axis = 1)

dataset['sigma_e']=  np.apply_along_axis(fit_baseline, 1, dataset['adcs'] )
displayType[2].image = dataset['sigma_e']
dataset['sigma_e'].dump("sigmae.dat")
print(dataset['adcs'].shape)

#print(sigma_e[10])
npix = 0


#for pix,valid in enumerate([526]):
for pix in [660]:
    #if not valid: continue
    if np.std(dataset['adcs'][pix])>10:continue
    print(pix)
    gaussian = lambda p, x: p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x-p[1])) ** 2 / (2. * p[2] ** 2))

    hist , bins = np.histogram(dataset['adcs'][pix],bins=np.arange(-10.5, 1.5, 1),density=False)
    histfull , bins_full = np.histogram(dataset['adcs'][pix],bins=np.arange(-10.5, 41.5, 1),density=False)
    hist_err = np.vectorize(poissonErr)(hist)

    gauss_residual = lambda p , x, y : errfunc_hist(gaussian, p, x, y)

    out = optimize.least_squares(gauss_residual, [1000.,0.,0.1], args=(np.arange(-10., 1., 1), hist),
                                 bounds = ([0.,-1.,1.e-6],[np.inf,0.,1.e5]))
    x = np.linspace(-10, 40,num=200)
    plt.subplot(2, 3, 4)
    plt.errorbar(np.arange(-10., 41., 1),histfull,yerr=np.vectorize(poissonErr)(histfull), fmt='o')
    plt.plot(x,gaussian(out.x,x), lw=2)
    cov=inv(np.dot(out.jac.T,out.jac))
    print(out.x,cov)
    plt.show()


    plt.subplot(2, 3, 5)
    print(np.arange(0, 50, 1).shape,dataset['adcs'][pix].shape)
    print(np.arange(0, 50, 1),dataset['adcs'][pix])
    plt.plot(np.arange(0, 50, 1),dataset['adcs'][pix])
    v = input('type')
    npix+=1
    if npix> 10 : break

print(526 in pix_badid)
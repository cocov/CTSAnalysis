#!/usr/bin/env python3
import matplotlib.pyplot as plt
from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.camera import CameraGeometry
from ctapipe.io.camera import find_neighbor_pixels
from astropy import units as u

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

from scipy.stats import norm
from numpy import linspace


availableSector = [1]
availableBoard = [0,1,2,3,4,5,6,7,8,9]
# Define cts
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)

pixels = []
pix_x = []
pix_y = []
pix_id = []
pix_badid = []
for pix in cts.camera.Pixels:
    if pix.sector in availableSector and pix.fadc in availableBoard:
        pix_x.append(pix.center[0])
        pix_y.append(pix.center[1])
        pix_id.append(pix.ID)
        pix_badid.append(True)
        #pixels.append(pix)
    else :
        pix_x.append(pix.center[0])
        pix_y.append(pix.center[1])
        pix_id.append(pix.ID)
        pix_badid.append(False)

pix_badid= np.array(pix_badid)
# pixels = list(self.cts.pixel_to_led['DC'].keys())
# pixels.sort()ter[0] for pix in pixels])
#pix_y = np.array([pix.center[1] for pix in pixels])
#pix_id = np.array([pix.ID for pix in pixels])
neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones((1296)) * 400., neighbors_pix, 'hexagonal')
formatter = LogFormatter(10)
#pix_x = np.array([pix.cen, labelOnlyBase=False)

def Gauss0(x, *param):
    A , sigma = param
    return A/sigma/np.sqrt(2.*np.pi)*np.exp(-(x)**2/(2.*sigma**2))

def fitGauss0(data):
    hist , bins = np.histogram(data,bins=np.arange(-10.5, 0.5, 1),density=True)
    coeff, var_matrix = optimize.curve_fit(Gauss0, xdata=np.arange(-10., 0., 1),ydata=hist,p0=np.array([1.,2.]) )
    return coeff[1]


fitGauss0_Vect = np.vectorize(fitGauss0)

'''
def create_halfhist(data):
    data[data<1]



def fit_gaussian(data):
    mu, std = norm.fit(data)



# Define model function to be used to fit to the data above:
def gauss0(x, *p):
    A, sigma = p
    return A*numpy.exp(-(x)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1000., 1]

coeff, var_matrix = curve_fit(gauss0, bin_centres, hist, p0=p0)
'''

plt.figure(0)
displayType = []

### First get the pedestal

inputfile_reader = zfits.zfits_event_source(url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.66.fits.fz"
                                                , data_type='r1', max_events=1000)

plt.subplot(2, 3, 1)
displayType.append(visualization.CameraDisplay(geom, title='Pedestals Average', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()

plt.subplot(2, 3, 2)
displayType.append(visualization.CameraDisplay(geom, title='Pedestals Mode', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()

plt.subplot(2, 3, 3)
displayType.append(visualization.CameraDisplay(geom, title='Sigma_e', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()

plt.ion()
pedestals = {}

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
                                                , data_type='r1', max_events=1000)
nevt = 0

for event in inputfile_reader:
    nevt+=1
    if nevt>100: break
    for telid in event.r1.tels_with_data :
        #print('telID',telid,'nevt',nevt)
        data = np.array(list(event.r1.tel[telid].adc_samples.values()))
        #print(scipy.stats.mode(data[10]).mode,pedestals['mode'][10])
        adcs = np.subtract(data,pedestals['mode'].reshape(1296,1))
        #print(adcs[18])
        #adcsTest = np.subtract(data.T,np.zeros(pedestals['mode'].shape)).T
        #print('Test',adcsTest[18])
        #print('Real',data[18])
        if 'adcs' not in dataset:
            dataset['adcs'] = adcs
        else:
            dataset['adcs'] = np.append(dataset['adcs'],adcs,axis = 1)

#dataset['sigma_e']= fitGauss0_Vect(dataset['adcs'])
#displayType[1].image = dataset['sigma_e']

#sigma_e = fitGauss0_Vect(dataset['adcs'])
#print(sigma_e[10])
plt.subplot(2, 3, 4)
npix = 0
for pix,valid in enumerate(pix_badid):
    if not valid: continue
    if np.std(dataset['adcs'][pix])>10:continue

    hist , bins = np.histogram(dataset['adcs'][pix],bins=np.arange(-10.5, 0.5, 1),density=False)

    width = (bins[1] - bins[0])
    h = plt.hist(dataset['adcs'][pix],bins=np.arange(-10.5, 40.5, 1))
    coeff, var_matrix = optimize.curve_fit(Gauss0, xdata=np.arange(-10., 0., 1),ydata=hist,
                                           p0=np.array([100.,2.]) )#,bounds=([0.,1e-8], [1000.,2.]))
    print(coeff)
    x = np.linspace(-10, 0,num=100)
    h1 = plt.plot(x, Gauss0(x,coeff[0],coeff[1]), lw=2)
    plt.show()
    v = input('type')
    npix+=1
    if npix> 10 : break

print('lendata2',dataset['adcs'][10].shape)
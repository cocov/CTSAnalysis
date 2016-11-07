#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.camera import CameraGeometry
from ctapipe.utils.datasets import get_datasets_path
from ctapipe import visualization
from ctapipe.calib import pedestals
from ctapipe.calib.camera import integrators
import numpy as np
import protozfitsreader as pf
from ctapipe.io import zfits
from matplotlib.ticker import LogFormatter
# Get CTS
from cts import cameratestsetup as cts

# Define cts
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle = 0., connected = False )

# first load in a very nasty way the camera geometry
filename = "/data/datasets/gamma_20deg_180deg_run100___cta-prod3-merged_desert-2150m--subarray-2-nosct.simtel.gz"

# get the geometry
geom = None

for event in hessio_event_source(filename):
    for telid in event.dl0.tels_with_data:
        if event.dl0.tel[telid].num_pixels != 1296 : continue
        print ("Telescope ID = ",telid)
        geom = CameraGeometry.guess(*event.meta.pixel_pos[telid],
                                event.meta.optical_foclen[telid])
        if geom.cam_id != 'SST-1m': break
    if geom!=None : break


# loop over all events, all telescopes and all channels and call
# the calc_peds function defined above to do some work:
datatype = 'DATA'


if datatype == 'MC':
    inputfile_reader = hessio_event_source(filename)
else:
    inputfile_reader = zfits.zfits_event_source(url= "/data/datasets/CTA/fakeEvt.fits.fz"
                                                ,data_type='r1',max_events=5)
formatter = LogFormatter(10, labelOnlyBase=False)

plt.figure(0)
displayType = []
plt.subplot(2, 3, 1)
displayType.append(visualization.CameraDisplay(geom, title='Pedestal Variation log', norm='lin',cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 2)
displayType.append(visualization.CameraDisplay(geom, title='Pedestals', norm='lin',cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 3)
displayType.append(visualization.CameraDisplay(geom, title='Pedestal With Low var', norm='lin',cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 5)
displayType.append(visualization.CameraDisplay(geom, title='Data ped subtracted', norm='lin',cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 6)
displayType.append(visualization.CameraDisplay(geom, title='Data without pedestal subtraction', norm='lin',cmap='coolwarm'))
displayType[-1].add_colorbar()

logVect = np.vectorize(np.log10)

def cutoff(x):
    y = 1.
    if np.log10(x)>1.8 : y = 0.
    return y

def cutoffBool(x):
    y = True
    if np.log10(x)>1.8 : y = False
    return y

cutoffVect = np.vectorize(cutoff)
cutoffBoolVect = np.vectorize(cutoffBool)
plt.ion()
#"/data/datasets/CameramyCamera_1_000.fits.fz"
for event in inputfile_reader:
    for telid in ( event.r1.tels_with_data if datatype != 'MC' else event.dl0.tels_with_data ):
        data = np.array(list(event.r1.tel[telid].adc_samples.values() if datatype != 'MC' else event.dl0.tel[telid].adc_samples.values()))
        if datatype== 'MC' and data.shape[1]!=1296:continue
        # pedestal subtraction
        peds, pedvars = pedestals.calc_pedestals_from_traces(data,0,20)
        # select only low variation pedestals
        pedsCut   = np.multiply(cutoffVect(pedvars),     peds)
        pedsCut2  = np.multiply(cutoffVect(pedvars),     peds)
        lowVarPed = np.extract( cutoffBoolVect(pedvars), peds)
        pedestalAvg = np.mean(lowVarPed, axis=0)
        #

        nsamples = event.r1.tel[telid].num_samples if datatype != 'MC' else event.dl0.tel[telid].num_samples
        #data_ped = data - np.atleast_3d(ped / nsamples)

        #data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
        params = {"integrator": "nb_peak_integration",
                  "integration_window": [7, 3],
                  "integration_sigamp": [2, 4],
                  "integration_lwt": 0}

        data = data if datatype == 'MC' else np.array([data])  # Test LG functionality
        data_peds = data-pedestalAvg/nsamples
        integration, window, peakpos = integrators.simple_integration(data_peds, params)
        integrationData, windowD, peakposD = integrators.simple_integration(data, params)

        #plt.subplot(2, 3, 1)
        displayType[0].image = logVect(pedvars)
        #plt.subplot(2, 3, 2)
        displayType[1].image = peds

        #plt.subplot(2, 3, 3)
        displayType[2].image = pedsCut

        plt.subplot(2, 3, 4)
        plt.hist(lowVarPed,bins=np.arange(0, 40, 1))
        plt.xlim([0,30])
        plt.ylim([0,250])
        #plt.subplot(2, 3, 5)
        displayType[3].image = integration[0]
        #plt.subplot(2, 3, 6)
        displayType[4].image = integrationData[0]

        plt.show()
        var = input("Press enter for next event")
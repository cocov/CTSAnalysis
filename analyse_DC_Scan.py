#!/usr/bin/env python3
import matplotlib.pyplot as plt
from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.camera import CameraGeometry
from ctapipe.io.camera import find_neighbor_pixels
from astropy import units as u

from ctapipe import visualization
from ctapipe.calib import pedestals
from ctapipe.calib.camera import integrators
import numpy as np
from ctapipe.io import zfits
from matplotlib.ticker import LogFormatter
# Get CTS
from cts import cameratestsetup as cts


availableSector = [1]
availableBoard = [0,1,2,3,4,6,7,8,9]
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
        pix_x.append(-100.)
        pix_y.append(-100.)
        pix_id.append(pix.ID)
        pix_badid.append(False)

pix_badid= np.array(pix_badid)
# pixels = list(self.cts.pixel_to_led['DC'].keys())
# pixels.sort()
#pix_x = np.array([pix.center[0] for pix in pixels])
#pix_y = np.array([pix.center[1] for pix in pixels])
#pix_id = np.array([pix.ID for pix in pixels])
neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones((1296)) * 400., neighbors_pix, 'hexagonal')

datatype = 'DATA'

if datatype == 'MC':
    inputfile_reader = hessio_event_source(filename)
else:
    inputfile_reader = zfits.zfits_event_source(url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.36.fits.fz"
                                                , data_type='r1', max_events=500)
formatter = LogFormatter(10, labelOnlyBase=False)

plt.figure(0)
displayType = []
plt.subplot(2, 3, 1)
displayType.append(visualization.CameraDisplay(geom, title='Pedestal Variation log', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 2)
displayType.append(visualization.CameraDisplay(geom, title='Pedestals', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 3)
displayType.append(visualization.CameraDisplay(geom, title='Pedestal With Low var', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()
plt.subplot(2, 3, 5)
displayType.append(visualization.CameraDisplay(geom, title='Data ped subtracted', norm='lin', cmap='coolwarm'))
#v = np.linspace(0, 300, 300, endpoint=True)
displayType[-1].add_colorbar()

plt.subplot(2, 3, 6)
displayType.append(
    visualization.CameraDisplay(geom, title='Data without pedestal subtraction', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()

logVect = np.vectorize(np.log10)


def cutoff(x):
    y = 1.
    if x <1000 : y = 0.3#np.log10(x) < -1000.: y = 0.
    return y


def cutoffBool(x):
    y = True
    if x<1000 : y = False #np.log10(x) < -1000.: y = False
    return y



cutoffVect = np.vectorize(cutoff)
cutoffBoolVect = np.vectorize(cutoffBool)
plt.ion()
# "/data/datasets/CameramyCamera_1_000.fits.fz"
min_evtcounter = 0
evt_counter = 0
for event in inputfile_reader:
    evt_counter +=1
    if evt_counter < min_evtcounter : continue
    for telid in (event.r1.tels_with_data if datatype != 'MC' else event.dl0.tels_with_data):
        data = np.array(list(event.r1.tel[telid].adc_samples.values() if datatype != 'MC' else event.dl0.tel[
            telid].adc_samples.values()))
        if datatype == 'MC' and data.shape[1] != 1296: continue
        # pedestal subtraction
        peds, pedvars = pedestals.calc_pedestals_from_traces(data, 0, 50)
        #print (np.max(peds))
        # select only low variation pedestals
        pedsCut = np.multiply(cutoffVect(peds), peds)
        pedsCut2 = np.multiply(cutoffVect(peds), peds)
        lowVarPed = np.extract(cutoffBoolVect(peds), peds)

        pedestalAvg = np.mean(lowVarPed, axis=0)
        #
        print (pedestalAvg,data[245])
        nsamples = event.r1.tel[telid].num_samples if datatype != 'MC' else event.dl0.tel[telid].num_samples
        # data_ped = data - np.atleast_3d(ped / nsamples)

        # data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
        params = {"integrator": "nb_peak_integration",
                  "integration_window": [50, 1],
                  "integration_sigamp": [2, 4],
                  "integration_lwt": 0}

        data = data if datatype == 'MC' else np.array([data])  # Test LG functionality
        data_peds = data - pedestalAvg

        integration, window, peakpos = integrators.simple_integration(data_peds, params)
        integrationData, windowD, peakposD = integrators.simple_integration(data, params)

        displayType[0].image = logVect(pedvars)
        displayType[1].image = peds
        displayType[2].image = pedsCut
        plt.subplot(2, 3, 4)
        plt.hist(lowVarPed, bins=np.arange(1800, 2300, 1))
        plt.xlim([1800, 2300])
        plt.ylim([0, 200])

        plt.xscale('log')
        plt.xscale('log')
        test = np.ones((1296),dtype='int')*25
        print('orig',integrationData[0][245])
        print(np.min(np.extract(pix_badid, integrationData[0])))
        pedsubValues = np.where(pix_badid,integrationData[0],np.min(np.extract(pix_badid, integrationData[0])))
        print('filered',pedsubValues[245])

        displayType[3].image = integration[0]
        displayType[4].image = pedsubValues

        plt.show()
        var = input("Press enter for next event")


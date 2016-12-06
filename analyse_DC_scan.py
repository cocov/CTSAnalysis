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
import matplotlib.animation as animation
import time


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
    pix_x.append(pix.center[0])
    pix_y.append(pix.center[1])
    pix_id.append(pix.ID)
    if pix.sector in availableSector and pix.fadc in availableBoard:
        pix_badid.append(True)
        #pixels.append(pix)
    else :
        #pix_x.append(-100.)
        #pix_y.append(-100.)
        pix_badid.append(False)

pix_badid= np.array(pix_badid)
neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones((1296)) * 400., neighbors_pix, 'hexagonal')


def find_led(data):
    pix = [np.argmax(data)]
    #neigh = geom.neighbors[pix[-1]]
    #pix+=neigh
    mask = np.zeros_like(np.zeros(1296), dtype='bool')
    mask[np.array(pix)]=True
    return mask

def generate_neighbourg_masks():
    mask = np.zeros_like(np.zeros((1296,1296)), dtype='bool')
    for i in range(1296):
        neigh = [i]
        neigh += geom.neighbors[i]
        mask[i][np.array(neigh)]=True
    return mask

def generate_hollow_neighbourg_masks():
    mask = np.zeros_like(np.zeros((1296,1296)), dtype='bool')
    for i in range(1296):
        neigh = []
        neigh += geom.neighbors[i]
        mask[i][np.array(neigh)]=True
    return mask

def sum_cluster(data,masks):
    data_tmp = np.array(data)
    data_tmp = data_tmp.reshape(1,1296)
    data_tmp = np.repeat(data_tmp, 1296, axis=0)
    return np.sum(np.ma.masked_array(data_tmp, mask=masks, fill_value=0),axis=0)


datatype = 'DATA'

plotting = True
if datatype == 'MC':
    inputfile_reader = hessio_event_source(filename)
else:
    inputfile_reader = zfits.zfits_event_source(url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.36.fits.fz"
                                                , data_type='r1', max_events=2130)
formatter = LogFormatter(10, labelOnlyBase=False)
if plotting:
    plt.figure(0)
    displayType = []
    plt.subplot(1, 1, 1)
    displayType.append(visualization.CameraDisplay(geom, title='Integrated ADC over 200ns, pedestal subtracted', norm='lin', cmap='coolwarm'))
    #displayType[-1].add_colorbar()
    '''
    plt.subplot(1, 2, 2)
    displayType.append(visualization.CameraDisplay(geom, title='Pedestals Variation', norm='lin', cmap='coolwarm'))
    displayType[-1].add_colorbar()
    '''
'''
plt.subplot(2, 2, 3)
displayType.append(visualization.CameraDisplay(geom, title='Data', norm='lin', cmap='coolwarm'))
displayType[-1].add_colorbar()
'''
plt.ion()

neigh_mask = generate_neighbourg_masks()
hollow_neigh_mask = generate_hollow_neighbourg_masks()


# "/data/datasets/CameramyCamera_1_000.fits.fz"
min_evtcounter = 3383-3018
best_pix = -1
h,h1=[],[]
evt_counter = 0
cnt = 0
for event in inputfile_reader:
    evt_counter +=1
    if evt_counter < min_evtcounter : continue
    for telid in (event.r1.tels_with_data if datatype != 'MC' else event.dl0.tels_with_data):
        data = np.array(list(event.r1.tel[telid].adc_samples.values() if datatype != 'MC' else event.dl0.tel[
            telid].adc_samples.values()))
        if datatype == 'MC' and data.shape[1] != 1296: continue
        # pedestal subtraction
        peds, pedvars = pedestals.calc_pedestals_from_traces(data, 0, 50)
        nsamples = event.r1.tel[telid].num_samples if datatype != 'MC' else event.dl0.tel[telid].num_samples
        # data_ped = data - np.atleast_3d(ped / nsamples)

        # data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
        params = {"integrator": "nb_peak_integration",
                  "integration_window": [50, 0],
                  "integration_sigamp": [2, 4],
                  "integration_lwt": 0}

        data = data if datatype == 'MC' else np.array([data])  # Test LG functionality
        integration, window, peakpos = integrators.simple_integration(data, params)
        integration[0]=integration[0]/50.
        #subtract median
        integration_sub = integration[0] - np.ones(1296)*np.median(np.extract(pix_badid, integration[0]))
        integration = np.where(pix_badid,integration[0],np.min(np.extract(pix_badid, integration[0])))
        integration_sub = np.where(pix_badid,integration_sub,np.min(np.extract(pix_badid, integration_sub)))
        '''
        displayType[0].image = integration
        mask = find_led(displayType[0].image)
        displayType[0].highlight_pixels(mask, linewidth=3)
        displayType[1].image = integration_sub
        mask = find_led(displayType[1].image)
        displayType[1].highlight_pixels(mask, linewidth=3)

        displayType[2].image = sum_cluster(integration_sub, neigh_mask)
        mask = find_led(displayType[2].image)
        displayType[2].highlight_pixels(mask, linewidth=3)
        plt.subplot(2, 2, 4)
        plt.hist(peds, bins=np.arange(0, 2300, 1))
        plt.xlim([1800, 2300])
        plt.ylim([0, 200])
        plt.yscale('log')
        '''
        #plt.xscale('log')
        bestPix = np.argmax(sum_cluster(integration, neigh_mask))
        badpix = np.argmin(integration)
        #h.append(np.divide(sum_cluster(integration, hollow_neigh_mask),sum_cluster(integration, neigh_mask))[bestPix])

        mypix = geom.neighbors[bestPix]
        sum_neigh = 0.
        num_neigh = 0
        for p in mypix:
            if not pix_badid[p]: continue
            sum_neigh+=integration_sub[p]
            num_neigh+=1
        sum_neigh = sum_neigh/num_neigh
        ratio = sum_neigh / (sum_neigh + integration_sub[bestPix])
        h1.append(ratio)
        if bestPix == best_pix or integration_sub[bestPix]<100 or sum_neigh<0.2: continue
        h.append(ratio)
        best_pix = bestPix
        val = np.zeros(1296)
        val[best_pix]=integration[best_pix]
        #print (integration[best_pix])
        if plotting :
            #displayType[0].image = val
            displayType[0].image = integration_sub
            mask1 = find_led(displayType[0].image)
            displayType[0].highlight_pixels(mask1, linewidth=1)
            plt.show()
            cnt+=1
            plt.savefig('plots/plot_'+str(cnt).zfill(3)+'.png')
            var = input("Press enter for next event")





ratios = np.array(h)
ratios1 = np.array(h1)
plt.hist(ratios1, bins=100)
plt.hist(ratios, bins=100)
#!/usr/bin/env python3
import matplotlib.pyplot as plt
from utils.geometry import generate_geometry
from ctapipe import visualization
from ctapipe.calib import pedestals
from ctapipe.calib.camera import integrators
import numpy as np
from ctapipe.io import zfits
from cts import cameratestsetup as cts


plotting = True
# Define Geometry
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)
geom,good_pixels = generate_geometry(cts,availableBoard={1:[0,1,2,3,4,6,7,8,9],2:[],3:[]})
good_pixels_mask = np.repeat(good_pixels, 50).reshape(good_pixels.shape[0], 1, 50)


plt.ion()

availableSector = [1]
availableBoard = [0,1,2,3,4,6,7,8,9]
# Define cts
cts = cts.CTS('/data/software/CTS/config/cts_config_0.cfg',
              '/data/software/CTS/config/camera_config.cfg',
              angle=0., connected=False)


def find_led(data):
    pix = [np.argmax(data)]
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



inputfile_reader = zfits.zfits_event_source(url="/data/datasets/CTA/CameraDigicam@localhost.localdomain_0_000.36.fits.fz"
                                                , data_type='r1', max_events=2130)
if plotting:
    plt.figure(0)
    displayType = []
    plt.subplot(1, 1, 1)
    displayType.append(visualization.CameraDisplay(geom, title='Integrated ADC over 200ns, pedestal subtracted', norm='lin', cmap='coolwarm'))

neigh_mask = generate_neighbourg_masks()
hollow_neigh_mask = generate_hollow_neighbourg_masks()


min_evtcounter = 3383-3018
best_pix = -1
h,h1=[],[]
evt_counter = 0
cnt = 0
for event in inputfile_reader:
    evt_counter +=1
    if evt_counter < min_evtcounter : continue
    for telid in event.r1.tels_with_data :
        data = np.array(list(event.r1.tel[telid].adc_samples.values() ))
        # pedestal subtraction
        peds, pedvars = pedestals.calc_pedestals_from_traces(data, 0, 50)
        nsamples = event.r1.tel[telid].num_samples
        # data_ped = np.array([data_ped[0], data_ped[0]])  # Test LG functionality
        params = {"integrator": "nb_peak_integration",
                  "integration_window": [50, 0],
                  "integration_sigamp": [2, 4],
                  "integration_lwt": 0}

        data = np.array([data])
        integration, window, peakpos = integrators.simple_integration(data, params)
        integration[0]=integration[0]/50.
        integration_sub = integration[0] - np.ones(1296)*np.median(np.extract(good_pixels, integration[0]))
        integration = np.where(good_pixels, integration[0], np.min(np.extract(good_pixels, integration[0])))
        integration_sub = np.where(good_pixels, integration_sub, np.min(np.extract(good_pixels, integration_sub)))
        bestPix = np.argmax(sum_cluster(integration, neigh_mask))
        badpix = np.argmin(integration)
        mypix = geom.neighbors[bestPix]
        sum_neigh = 0.
        num_neigh = 0
        for p in mypix:
            if not good_pixels[p]: continue
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
        if plotting :
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
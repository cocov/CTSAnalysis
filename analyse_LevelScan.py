#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.io import zfits
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry
from utils.fitting import gaussian_residual,spe_peaks_in_event_list
from utils.plots import pickable_visu
from utils.fitting import multi_gaussian_residual_with0
import sys
from ctapipe import visualization
from utils.histogram import histogram
from ctapipe.calib.camera import integrators

from optparse import OptionParser

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

# Setup configuration
parser.add_option("--cts_sector", dest="cts_sector",
                  help="Sector covered by CTS", default=1,type=int)
parser.add_option("-l", "--scan_level", dest="scan_level",
                  help="list of scans DC level, separated by ',', if only three argument, min,max,step", default="50,260,10")
parser.add_option("-e", "--events_per_level", dest="events_per_level",
                  help="number of events per level", default=3500,type=int)

parser.add_option("-s", "--use_saved_histo", dest="use_saved_histo",
                  help="load the histograms from file", default=False)



# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="list of string differing in the file name, sperated by ','", default='87,91' )
parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/LevelScan/20161130/")
parser.add_option( "--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")
parser.add_option( "--calibration_filename", dest="calibration_filename",
                  help="calibration file name", default="calib_spe.npz")
parser.add_option( "--calibration_directory", dest="calibration_directory",
                  help="calibration file directory", default="/data/datasets/DarkRun/")
parser.add_option( "--saved_histo_directory", dest="saved_histo_directory",
                  help="directory of histo file", default='/data/datasets/CTA/LevelScan/20161130/')
parser.add_option( "--saved_histo_filename", dest="saved_histo_filename",
                  help="directory of histo file", default='mpes.npz')

# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')
options.scan_level = [int(level) for level in options.scan_level.split(',')]

if len(options.scan_level)==3:
    options.scan_level=np.arange(options.scan_level[0],options.scan_level[1]+options.scan_level[2],options.scan_level[2])

# Define Geometry
sector_to_angle = {1:0.,2:120.,3:240.} #TODO check and put it in cts
cts = cts.CTS('/data/software/CTS/config/cts_config_%d.cfg'%(sector_to_angle[options.cts_sector]),
              '/data/software/CTS/config/camera_config.cfg',
              angle=sector_to_angle[options.cts_sector], connected=False)

geom,good_pixels = generate_geometry(cts)

plt.ion()

# Get calibration objects

calib_file = np.load(options.calibration_directory+options.calibration_filename)

def remap_conf_dict(config):
    new_conf = []
    for i, pix in enumerate(config[list(config.keys())[0]]):
        new_conf.append({})
    for key in list(config.keys()):
        for i, pix in enumerate(config[key]):
            if np.isfinite(pix[0]):
                new_conf[i][key] = pix[0]
            else:
                new_conf[i][key] = 0.
    return new_conf


# Prepare the mpe histograms
mpes = histogram(bin_center_min=-10., bin_center_max=4095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)))
peaks = histogram(bin_center_min=-10., bin_center_max=4095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)))
n_batch, batch_num, max_evt = 10, 0, 1000000
level = 0
batch_peakpos = np.ones(options.scan_level.shape+(1296,1))*np.nan
batch_integration = np.empty(options.scan_level.shape+(1296,1))*np.nan
evt_num=0
first_evt= True
first_evt_num = 0
if not options.use_saved_histo:
    # Loop over the files
    for file in options.file_list:
        # Get the file
        _url = options.directory+options.file_basename%(file)
        inputfile_reader = zfits.zfits_event_source( url= _url
                                                     ,data_type='r1'
                                                     , max_events=100000)
        print('--|> Moving to file %s'%(_url))
        # Loop over event in this file
        for event in inputfile_reader:
            for telid in event.r1.tels_with_data:
                if first_evt:
                    first_evt_num = event.r1.tel[telid].eventNumber
                    first_evt = False
                evt_num = event.r1.tel[telid].eventNumber-first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    print('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                if (event.r1.event_id) % 100 == 0:
                    print("Progress {:2.1%}".format(
                        (evt_num - level * options.events_per_level) / options.events_per_level), end="\r")

                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                data = data -  calib_file['baseline'][:,0].reshape(data.shape[0],1)
                data=data.reshape((1,)+data.shape)
                # now integrate
                params = {"integrator": "nb_peak_integration",
                          "integration_window": [50, 1],
                          "integration_sigamp": [2, 4],
                          "integration_lwt": 0}
                integration, window, peakpos = integrators.simple_integration(data,params)
                mpes.fill(integration[0],indices=(level,))
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.io import zfits
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry
from utils.histogram import histogram
from ctapipe.calib.camera import integrators
from utils.plots import pickable_visu_mpe
from utils.pdf import mpe_gaussian_distribution
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
                  help="list of scans DC level, separated by ',', if only three argument, min,max,step", default="50,250,10")
parser.add_option("-e", "--events_per_level", dest="events_per_level",
                  help="number of events per level", default=3500,type=int)
parser.add_option("-s", "--use_saved_histo", dest="use_saved_histo",action="store_true",
                  help="load the histograms from file", default=False)
parser.add_option("-p", "--perform_fit", dest="perform_fit",action="store_false",
                  help="perform fit of mpe", default=True)

# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="list of string differing in the file name, sperated by ','", default='87,88,89,90,91' )
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
                  help="name of histo file", default='mpes.npz')
parser.add_option( "--saved_fit_filename", dest="saved_fit_filename",
                  help="name of fit file", default='fits_mpes.npz')

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

# Leave the hand
plt.ion()

# Get calibration objects
calib_file = np.load(options.calibration_directory+options.calibration_filename)

# Prepare the mpe histograms
mpes = histogram(bin_center_min=-100., bin_center_max=3095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Peak ADC',ylabel='$\mathrm{N_{entries}}$',label='MPE')
peaks = histogram(bin_center_min=-1., bin_center_max=51., bin_width=1., data_shape=(options.scan_level.shape+(1296,)),
                  xlabel='Peak maximum position [4ns]', ylabel='$\mathrm{N_{entries}}$', label='MPE')

# Few counters
level,evt_num,first_evt,first_evt_num = 0,0,True,0

# Where do we take the data from
if not options.use_saved_histo:
    # Loop over the files
    for file in options.file_list:
        # Get the file
        _url = options.directory+options.file_basename%(file)
        inputfile_reader = zfits.zfits_event_source( url= _url,data_type='r1', max_events=100000)
        if options.verbose: print('--|> Moving to file %s'%(_url))
        # Loop over event in this file
        for event in inputfile_reader:
            for telid in event.r1.tels_with_data:
                if first_evt:
                    first_evt_num = event.r1.tel[telid].eventNumber
                    first_evt = False
                evt_num = event.r1.tel[telid].eventNumber-first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    if options.verbose: print('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                if options.verbose and (event.r1.event_id) % 100 == 0:
                    print("Progress {:2.1%}".format(
                        (evt_num - level * options.events_per_level) / options.events_per_level), end="\r")
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # subtract the pedestals
                data = data -  calib_file['baseline'][:,0].reshape(data.shape[0],1)
                # put in proper format
                data=data.reshape((1,)+data.shape)
                # integration parameter
                params = {"integrator": "nb_peak_integration","integration_window": [8, 4],
                          "integration_sigamp": [2, 4],"integration_lwt": 0}
                # now integrate
                integration, window, peakpos = integrators.simple_integration(data,params)
                # and fill the histos
                mpes.fill(integration[0],indices=(level,))
                peaks.fill(np.argmax(data[0],axis=1),indices=(level,))
    # Save the MPE histos in a file
    mpes._compute_errors()
    peaks._compute_errors()
    if options.verbose : print('--|> Save the data in %s' % (options.saved_histo_directory+options.saved_histo_filename))
    np.savez_compressed(options.saved_histo_directory+options.saved_histo_filename, mpes=mpes.data ,mpes_bin_centers = mpes.bin_centers ,
                        peaks=peaks.data, peaks_bin_centers=peaks.bin_centers
                        )
else :
    if options.verbose: print('--|> Recover data from %s' % (options.saved_histo_directory+options.saved_histo_filename))
    file = np.load(options.saved_histo_directory+options.saved_histo_filename)
    mpes = histogram(data=file['mpes'],bin_centers=file['mpes_bin_centers'])
    peaks = histogram(data=file['peaks'],bin_centers=file['peaks_bin_centers'])
    peaks.xlabel = 'sample [$\mathrm{4 ns^{1}}$]'
    peaks.ylabel = '$\mathrm{N_{trigger}}$'
    mpes.xlabel = 'Integrated ADC in sample [4-12]'
    mpes.ylabel = '$\mathrm{N_{trigger}}$'


# Fit them
if options.perform_fit:
    def p0_first_func(x,xrange,config):
        p = []
        p.append(5.3)
        p.append(0.7)
        p.append(0.7)
        p.append(0.)
        # how many peaks
        npeaks= 20#int((x[np.where(x != 0)[0][-1]]-x[np.where(x != 0)[0][0]])//p[0])
        for i in range(npeaks):
            p.append(100.)
        return p

    def bound_func(x,xrange,config):
        return None

    def slice_func(x,xrange,config):
        return [np.where(x != 0)[0][0],np.where(x != 0)[0][-1],1]

    # Perform the actual fit
    mpes.fit(mpe_gaussian_distribution,p0_first_func, slice_func, bound_func)
    # Save the parameters
    if options.verbose: print('--|> Save the fit result in %s' % (options.saved_histo_directory + options.saved_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_fit_filename, mpes_fit_results=mpes.fit_result)
else :
    if options.verbose: print('--|> Load the fit result from %s' % (options.saved_histo_directory + options.saved_fit_filename))
    h = np.load(options.saved_histo_directory + options.saved_fit_filename)
    mpes.fit_result = h['mpes_fit_results']
    mpes.fit_function = mpe_gaussian_distribution

# Plot them




def show_level(level,hist):
    def slice_fun(x, **kwargs):
        return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_mpe([hist], ax[1], fig, slice_fun, {'level':level}, geom, title='', norm='lin',
                                     cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    peak = peaks.data[level]
    peak = peaks.find_bin(np.argmax(peak, axis=1))
    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = peak
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(516)
    plt.show()


show_level(3,mpes)
show_level(3,peaks)
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.io import zfits
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry
from optparse import OptionParser
from utils.fitting import gaussian,spe_peaks_in_event_list
from utils.plots import pickable_visu
from utils.fitting import multi_gaussian_residual_with0
from utils.fitting import multi_gaussian_with0
import sys
from ctapipe import visualization
from utils.histogram import histogram
import peakutils
from scipy import signal
from utils.peakdetect import peakdetect

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

# Setup configuration
parser.add_option("--cts_sector", dest="cts_sector",
                  help="Sector covered by CTS", default=1,type=int)
parser.add_option("-a", "--use_saved_histo_adcs", dest="use_saved_histo_adcs",action="store_true",
                  help="load the histograms from file", default=False)
parser.add_option("-s", "--use_saved_histo_spe", dest="use_saved_histo_spe",action="store_true",
                  help="load the histograms from file", default=False)

parser.add_option("-c", "--perform_adc_fit", dest="perform_adc_fit",action="store_false",
                  help="perform fit of adcs", default=True)

parser.add_option("-e", "--perform_spe_fit", dest="perform_spe_fit",action="store_false",
                  help="perform fit of spes", default=True)

# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="input filenames separated by ','", default='86' )
parser.add_option( "--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")
parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161130/")


parser.add_option( "--calibration_filename", dest="calibration_filename",
                  help="calibration file name", default="darkrun_calib.npz")
parser.add_option( "--calibration_directory", dest="calibration_directory",
                  help="calibration file directory", default="/data/datasets/CTA/DarkRun/20161130/")

parser.add_option( "--saved_histo_directory", dest="saved_histo_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161130/')
parser.add_option( "--saved_adc_histo_filename", dest="saved_adc_histo_filename",
                  help="name of histo file", default='darkrun_adc_hist.npz')
parser.add_option( "--saved_spe_histo_filename", dest="saved_spe_histo_filename",
                  help="name of histo file", default='darkrun_adc_hist.npz')

parser.add_option( "--saved_adc_fit_filename", dest="saved_adc_fit_filename",
                  help="name of adc fit file", default='darkrun_adc_fit.npz')
parser.add_option( "--saved_spe_fit_filename", dest="saved_spe_fit_filename",
                  help="name of spe fit file", default='darkrun_spe_fit.npz')

# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')


# Define Geometry
sector_to_angle = {1:0.,2:120.,3:240.} #TODO check and put it in cts
cts = cts.CTS('/data/software/CTS/config/cts_config_%d.cfg'%(sector_to_angle[options.cts_sector]),
              '/data/software/CTS/config/camera_config.cfg',
              angle=sector_to_angle[options.cts_sector], connected=False)
geom,good_pixels = generate_geometry(cts)
# Leave the hand
plt.ion()


adcs = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))
spes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))


#peaks_index = peakutils.indexes(y, threshold, min_dist)

if not options.use_saved_histo_adcs:
    # Reading the file
    n_evt,n_batch,batch_num,max_evt=0,300,0,1e8
    batch = None

    print('--|> Treating the batch #%d of %d events' % (batch_num, n_batch))
    for file in options.file_list:
        # Open the file
        _url = options.directory+options.file_basename%(file)
        inputfile_reader = zfits.zfits_event_source(
            url=_url
            , data_type='r1', max_events=100000)

        if options.verbose: print('--|> Moving to file %s'%(_url))
        # Loop over event in this file
        for event in inputfile_reader:
            n_evt += 1
            if n_evt > max_evt: break
            if (n_evt - n_batch * batch_num) % 10 == 0: print(
                "Progress {:2.1%}".format(float(n_evt - batch_num * n_batch) / n_batch), end="\r")
            for telid in event.r1.tels_with_data:
                if n_evt % n_batch == 0:
                    print('--|> Treating the batch #%d of %d events' % (batch_num, n_batch))
                    # Update adc histo
                    adcs.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2]))
                    # Reset the batch
                    batch = None
                    batch_num += 1
                    print('--|> Reading  the batch #%d of %d events' % (batch_num, n_batch))
                # Get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # Append the data to the batch
                if type(batch).__name__ != 'ndarray':
                    batch = data.reshape(data.shape[0], 1, data.shape[1])
                else:
                    batch = np.append(batch, data.reshape(data.shape[0], 1, data.shape[1]), axis=1)

            '''
            for telid in event.r1.tels_with_data:
                if options.verbose and (event.r1.event_id) % 100 == 0:
                    print("Progress {:2.1%}".format(event.r1.event_id/10000), end="\r")
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # fill with a batch of n_sample
                adcs.fill_with_batch(data)
            '''

    if options.verbose : print('--|> Save the data in %s' % (options.saved_histo_directory+options.saved_adc_histo_filename))
    np.savez_compressed(options.saved_histo_directory+options.saved_adc_histo_filename,
                        adcs=adcs.data, adcs_bin_centers=adcs.bin_centers)
else:
    if options.verbose: print(
        '--|> Recover data from %s' % (options.saved_histo_directory + options.saved_adc_histo_filename))
    file = np.load(options.saved_histo_directory + options.saved_adc_histo_filename)
    adcs = histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))


if options.perform_adc_fit:

    def slice_func(y, x, *args, **kwargs):
        if np.where(y != 0)[0].shape[0] == 0: return [0, 1, 1]
        xmin = np.max(np.argmax(y)-10,0)
        xmax = np.argmax(y)+2
        return [xmin,xmax,1]


    def p0_func(x,xrange,*args,**kwargs):
        norm = np.sum(x)
        mean = xrange[np.argmax(x)]
        sigma = 0.9
        return [norm,mean,sigma]

    def bound_func(x,xrange,*args,**kwargs):
        min_norm, max_norm = 1., np.inf
        min_mean, max_mean = xrange[np.argmax(x) - 2], xrange[np.argmax(x) + 2]
        min_sigma, max_sigma = 1e-6, 1e6
        return ([min_norm, min_mean, min_sigma], [max_norm, max_mean, max_sigma])


    print('--|> Compute baseline and sigma_e from ADC distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = adcs.fit( gaussian , p0_func, slice_func, bound_func)

    #adcs.predef_fit(type='Gauss',slice_func=slice_func,initials = [10000,2000,0.7])
    if options.verbose: print(
        '--|> Save the data in %s' % (options.saved_histo_directory + options.saved_adc_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_adc_fit_filename, adcs_fit_results=adcs.fit_result)
else:
    if options.verbose: print(
        '--|> Recover data from %s' % (options.saved_histo_directory + options.saved_adc_fit_filename))
    file = np.load(options.saved_histo_directory + options.saved_adc_fit_filename)
    adcs.fit_result = np.copy(file['adcs_fit_results'])
    adcs.fit_function = gaussian




if not options.use_saved_histo_spe:
    # Reading the file
    n_evt, n_batch, batch_num, max_evt = 0, 300, 0, 1e8
    batch = None

    print('--|> Treating the batch #%d of %d events' % (batch_num, n_batch))
    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % (file)
        inputfile_reader = zfits.zfits_event_source(
            url=_url
            , data_type='r1', max_events=100000)

        if options.verbose: print('--|> Moving to file %s' % (_url))
        # Loop over event in this file
        for event in inputfile_reader:
            n_evt += 1
            if n_evt > max_evt: break
            if (n_evt - n_batch * batch_num) % 10 == 0: print(
                "Progress {:2.1%}".format(float(n_evt - batch_num * n_batch) / n_batch), end="\r")
            for telid in event.r1.tels_with_data:
                if n_evt % n_batch == 0:
                    print('--|> Treating the batch #%d of %d events' % (batch_num, n_batch))
                    # Update adc histo
                    spes.fill_with_batch(
                        spe_peaks_in_event_list(batch, adcs.fit_result[:, 1, 0], adcs.fit_result[:, 2, 0]) )
                    # Reset the batch
                    batch = None
                    batch_num += 1
                    print('--|> Reading  the batch #%d of %d events' % (batch_num, n_batch))
                # Get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # Append the data to the batch
                if type(batch).__name__ != 'ndarray':
                    batch = data.reshape(data.shape[0], 1, data.shape[1])
                else:
                    batch = np.append(batch, data.reshape(data.shape[0], 1, data.shape[1]), axis=1)




    if options.verbose: print(
        '--|> Save the data in %s' % (options.saved_histo_directory + options.saved_spe_histo_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_spe_histo_filename,
                        spes=adcs.data, spes_bin_centers=spes.bin_centers)
else:
    if options.verbose: print(
        '--|> Recover data from %s' % (options.saved_histo_directory + options.saved_spe_histo_filename))
    file = np.load(options.saved_histo_directory + options.saved_spe_histo_filename)
    spes = histogram(data=np.copy(file['spes']), bin_centers=np.copy(file['spes_bin_centers']))




def display(hists=[adcs,spes],pix_init=700):
    def slice_func(x,*args,**kwargs):
        return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu(hists, ax[1], fig, slice_func, True,'log', geom, title='', norm='lin',
                                 cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    peak = hists[0].fit_result[:,2,0]
    peak[np.isnan(peak)]=0
    peak[peak<0.3]=0.3
    peak[peak>1.3]=1.3

    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = peak
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(pix_init)
    plt.show()


display()

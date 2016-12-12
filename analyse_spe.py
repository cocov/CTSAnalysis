#!/usr/bin/env python3
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
from ctapipe import visualization
from cts import cameratestsetup as cts

from data_treatement import adc_hist
from utils.fitting import gaussian
from utils.fitting import multi_gaussian_with0
from utils.geometry import generate_geometry
from utils.histogram import histogram
from utils.plots import pickable_visu

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

# Setup configuration
parser.add_option("--cts_sector", dest="cts_sector",
                  help="Sector covered by CTS", default=1, type=int)
parser.add_option("-a", "--use_saved_histo_adcs", dest="use_saved_histo_adcs", action="store_true",
                  help="load the histograms from file", default=False)
parser.add_option("-s", "--use_saved_histo_spe", dest="use_saved_histo_spe", action="store_true",
                  help="load the histograms from file", default=False)

parser.add_option("-c", "--perform_adc_fit", dest="perform_adc_fit", action="store_false",
                  help="perform fit of adcs", default=True)

parser.add_option("-e", "--perform_spe_fit", dest="perform_spe_fit", action="store_false",
                  help="perform fit of spes", default=True)

# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="input filenames separated by ','", default='86')
parser.add_option("--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")
parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161130/")

parser.add_option("--calibration_filename", dest="calibration_filename",
                  help="calibration file name", default="darkrun_calib.npz")
parser.add_option("--calibration_directory", dest="calibration_directory",
                  help="calibration file directory", default="/data/datasets/CTA/DarkRun/20161130/")

parser.add_option("--saved_histo_directory", dest="saved_histo_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161130/')
parser.add_option("--saved_adc_histo_filename", dest="saved_adc_histo_filename",
                  help="name of histo file", default='darkrun_adc_hist.npz')
parser.add_option("--saved_spe_histo_filename", dest="saved_spe_histo_filename",
                  help="name of histo file", default='darkrun_spe_hist.npz')

parser.add_option("--saved_adc_fit_filename", dest="saved_adc_fit_filename",
                  help="name of adc fit file", default='darkrun_adc_fit.npz')
parser.add_option("--saved_spe_fit_filename", dest="saved_spe_fit_filename",
                  help="name of spe fit file", default='darkrun_spe_fit.npz')

parser.add_option("-n", "--n_evt_per_batch", dest="n_evt_per_batch",
                  help="number of events per batch", default=300, type=int)
parser.add_option("--evt_max", dest="evt_max",
                  help="maximal number of events", default=2e5, type=int)

# Arrange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')

# Define Geometry
sector_to_angle = {1: 0., 2: 120., 3: 240.}
cts = cts.CTS('/data/software/CTS/config/cts_config_%d.cfg' % (sector_to_angle[options.cts_sector]),
              '/data/software/CTS/config/camera_config.cfg',
              angle=sector_to_angle[options.cts_sector], connected=False)
geom, good_pixels = generate_geometry(cts)

# Leave the hand
plt.ion()

# Define the histograms
adcs = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))
spes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))

# Get the adcs
if not options.use_saved_histo_adcs:
    # Fill the adcs hist from data
    adc_hist.run(adcs, options, 'ADC')
else:
    if options.verbose:
        print('--|> Recover data from %s' % (options.saved_histo_directory + options.saved_adc_histo_filename))
    file = np.load(options.saved_histo_directory + options.saved_adc_histo_filename)
    adcs = histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))

if options.perform_adc_fit:
    # noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal,PyShadowingNames
    def slice_func(y, x, *args, **kwargs):
        if np.where(y != 0)[0].shape[0] == 0:
            return [0, 1, 1]
        xmin = np.max(np.argmax(y) - 10, 0)
        xmax = np.argmax(y) + 2
        return [xmin, xmax, 1]


    # noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
    def p0_func(x, xrange, *args, **kwargs):
        norm = np.sum(x)
        mean = xrange[np.argmax(x)]
        sigma = 0.9
        return [norm, mean, sigma]


    # noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
    def bound_func(x, xrange, *args, **kwargs):
        min_norm, max_norm = 1., np.inf
        min_mean, max_mean = xrange[np.argmax(x) - 2], xrange[np.argmax(x) + 2]
        min_sigma, max_sigma = 1e-6, 1e6
        return [min_norm, min_mean, min_sigma], [max_norm, max_mean, max_sigma]


    print('--|> Compute baseline and sigma_e from ADC distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = adcs.fit(gaussian, p0_func, slice_func, bound_func)

    # adcs.predef_fit(type='Gauss',slice_func=slice_func,initials = [10000,2000,0.7])
    if options.verbose:
        print('--|> Save the data in %s' % (options.saved_histo_directory + options.saved_adc_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_adc_fit_filename,
                        adcs_fit_results=adcs.fit_result)
else:
    if options.verbose:
        print('--|> Recover data from %s' % (options.saved_histo_directory + options.saved_adc_fit_filename))
    file = np.load(options.saved_histo_directory + options.saved_adc_fit_filename)
    adcs.fit_result = np.copy(file['adcs_fit_results'])
    adcs.fit_function = gaussian

if not options.use_saved_histo_spe:
    # Fill the adcs hist from data
    adc_hist.run(spes, options, 'SPE')
else:
    if options.verbose:
        print('--|> Recover data from %s' % (options.saved_histo_directory + options.saved_spe_histo_filename))
    file = np.load(options.saved_histo_directory + options.saved_spe_histo_filename)
    spes = histogram(data=np.copy(file['spes']), bin_centers=np.copy(file['spes_bin_centers']))


# noinspection PyUnusedLocal
def func_multi_all(p, x, config):
    p_new = [0.] * 12
    p_new[0] = 0.
    p_new[1] = p[0]
    p_new[2] = p[1]
    p_new[3] = p[2]
    p_new[4] = p[3]
    p_new[5] = p[4]
    p_new[6] = p[5]
    p_new[7] = p[6]
    p_new[8] = p[7]
    p_new[9] = 1.
    p_new[10] = p[8]
    p_new[11] = p[9]
    return multi_gaussian_with0(p_new, x)


if options.perform_spe_fit:
    # noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
    def p0_func(*args, config=None, **kwargs):
        # print([config[2][0] ,0.7 , 5.6, 10000.,1000.,100. , config[1][0] ,0. , 100. ,10.])
        return [config[2][0], 0.7, 5.6, 10000., 1000., 100., config[1][0], 0., 100., 10.]
        # return [0.7 , 5.6, 10000.,1000.,100. , 0. , 100. ,10.]


    # noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
    def bound_func(x, *args, config=None, **kwargs):
        param_min = [config[2][0] * 0.1, 0.01, 0., 100., 1., 0., config[1][0] - 3 * config[1][1], -10., 0., 0.]
        param_max = [config[2][0] * 10., 5., 100., np.inf, np.inf, np.inf, config[1][0] + 3 * config[1][1], 10., np.inf,
                     np.inf]
        # param_min = [0.01, 0. , 100.   , 1.    , 0.  ,-10., 0.    ,0.]
        # param_max = [5. , 100., np.inf, np.inf, np.inf,10. , np.inf,np.inf]
        return param_min, param_max


    # noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
    def slice_func(x, *args, **kwargs):
        if np.where(x != 0)[0].shape[0] == 0:
            return [0, 1, 1]
        return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


    print('--|> Compute Gain, sigma_e, sigma_i from SPE distributions')
    # Fit the baseline and sigma e of all pixels
    spes.fit(func_multi_all, p0_func, slice_func, bound_func, config=adcs.fit_result)
    if options.verbose:
        print('--|> Save the data in %s' % (options.saved_histo_directory + options.saved_spe_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_spe_fit_filename,
                        spes_fit_results=spes.fit_result)
else:
    if options.verbose:
        print('--|> Recover data from %s' % (options.saved_histo_directory + options.saved_spe_fit_filename))
    file = np.load(options.saved_histo_directory + options.saved_spe_fit_filename)
    spes.fit_result = np.copy(file['spes_fit_results'])
    spes.fit_function = func_multi_all


def display(hists=None, pix_init=700):
    if hists is None:
        hists = [adcs, spes]

    # noinspection PyShadowingNames,PyShadowingNames,PyUnusedLocal,PyUnusedLocal
    def slice_func(x, *args, **kwargs):
        return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]

    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu(hists, ax[1], fig, slice_func, [True, True], 'log', geom, title='', norm='lin',
                                 cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    peak = hists[0].fit_result[:, 2, 0]
    peak[np.isnan(peak)] = 0
    peak[peak < 0.3] = 0.3
    peak[peak > 1.3] = 1.3

    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = peak
    # noinspection PyProtectedMember
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(pix_init)
    plt.show()


display()


def display_var(hist, title='Gain [ADC/p.e.]', index_var=1, limit_min=0., limit_max=10., bin_width=0.2):
    f, ax = plt.subplots(1, 2, figsize=(20, 7))
    plt.subplot(1, 2, 1)
    vis_gain = visualization.CameraDisplay(geom, title='', norm='lin', cmap='viridis')
    vis_gain.add_colorbar()
    vis_gain.colorbar.set_label(title)
    h = np.copy(hist.fit_result[:, index_var, 0])
    h_err = np.copy(hist.fit_result[:, index_var, 1])
    h[np.isnan(h_err)] = limit_min
    h[h < limit_min] = limit_min
    h[h > limit_max] = limit_max
    vis_gain.image = h
    # plt.subplot(1,2,2)
    hh, bin_tmp = np.histogram(h, bins=np.arange(limit_min - bin_width / 2, limit_max + 1.5 * bin_width, bin_width))
    hh_hist = histogram(data=hh.reshape(1, hh.shape[0]),
                        bin_centers=np.arange(limit_min, limit_max + bin_width, bin_width), xlabel=title,
                        ylabel='$\mathrm{N_{pixel}/%.2f}$' % bin_width, label='All pixels')
    hh_hist.show(which_hist=(0,), axis=ax[1], show_fit=False)
    plt.show()


display_var(spes, title='Gain [ADC/p.e.]', index_var=1, limit_min=2., limit_max=6., bin_width=0.2)
display_var(spes, title='$\sigma_i$ [ADC]', index_var=0, limit_min=0., limit_max=2., bin_width=0.05)
display_var(adcs, title='$\sigma_e$ [ADC]', index_var=2, limit_min=0., limit_max=2., bin_width=0.05)
display_var(adcs, title='Baseline [ADC]', index_var=1, limit_min=1950., limit_max=2050., bin_width=10.)

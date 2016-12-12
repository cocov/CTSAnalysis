#!/usr/bin/env python3
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry,generate_geometry_0
from utils.plots import pickable_visu_mpe
from utils.pdf import mpe_distribution_general
from optparse import OptionParser
from utils.histogram import histogram
import peakutils
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from data_treatement import mpe_hist

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

# Setup configuration
parser.add_option("--cts_sector", dest="cts_sector",
                  help="Sector covered by CTS", default=1,type=int)

parser.add_option("-l", "--scan_level", dest="scan_level",
                  help="list of scans DC level, separated by ',', if only three argument, min,max,step", default="50,80,10")

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
                  help="name of histo file", default='mpes_few.npz')

parser.add_option( "--saved_fit_filename", dest="saved_fit_filename",
                  help="name of fit file", default='fits_mpes_few.npz')

parser.add_option( "--saved_spe_fit_filename", dest="saved_spe_fit_filename",
                  help="name of spe fit file", default='darkrun_spe_fit.npz')

parser.add_option( "--dark_calibration_directory", dest="dark_calibration_directory",
                  help="darkrun calibration file directory", default="/data/datasets/CTA/DarkRun/20161130/")

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
#geom,good_pixels = generate_geometry(cts)
geom = generate_geometry_0()

# Leave the hand
plt.ion()

# Prepare the mpe histograms
mpes = histogram(bin_center_min=-100., bin_center_max=3095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Integrated ADC',ylabel='$\mathrm{N_{entries}}$',label='MPE')
mpes_peaks = histogram(bin_center_min=-100., bin_center_max=3095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Peak ADC',ylabel='$\mathrm{N_{entries}}$',label='MPE')
peaks = histogram(bin_center_min=-1., bin_center_max=51., bin_width=1., data_shape=(options.scan_level.shape+(1296,)),
                  xlabel='Peak maximum position [4ns]', ylabel='$\mathrm{N_{entries}}$', label='MPE')


# Where do we take the data from
if not options.use_saved_histo:
    # Loop over the files
    mpe_hist.run([mpes,mpes_peaks,peaks], options)
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
    # recover previous fit
    if options.verbose: print(
        '--|> Recover data from %s' % (options.dark_calibration_directory + options.saved_spe_fit_filename))
    file = np.load(options.dark_calibration_directory + options.saved_spe_fit_filename)
    spes_fit_result = np.copy(file['spes_fit_results'])


    def p0_func(y, x, config=None, *args, **kwargs):

        threshold = 0.005
        min_dist = 4
        peaks_index = peakutils.indexes(y, threshold, min_dist)

        if len(peaks_index) == 0:
            return [np.nan] * 7
        amplitude = np.sum(y)
        offset, gain = np.polynomial.polynomial.polyfit(np.arange(0, peaks_index.shape[-1], 1), x[peaks_index], deg=1,
                                                        w=(np.sqrt(y[peaks_index])))
        sigma_start = np.zeros(peaks_index.shape[-1])
        for i in range(peaks_index.shape[-1]):

            start = max(int(peaks_index[i] - gain // 2), 0)  ## Modif to be checked
            end = min(int(peaks_index[i] + gain // 2), len(x))  ## Modif to be checked
            if start == end and end < len(x) - 2:
                end += 1
            elif start == end:
                start -= 1

            # print(start,end,y[start:end])
            if i == 0:
                mu = -np.log(np.sum(y[start:end]) / np.sum(y))

            temp = np.average(x[start:end], weights=y[start:end])
            sigma_start[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))

        bounds = [[0., 0.], [np.inf, np.inf]]
        sigma_n = lambda x, y, n: np.sqrt(x ** 2 + n * y ** 2)
        sigma, sigma_error = curve_fit(sigma_n, np.arange(0, peaks_index.shape[-1], 1), sigma_start, bounds=bounds)
        sigma = sigma / gain

        mu_xt = np.mean(y) / mu / gain - 1

        # print([mu, mu_xt, gain, offset, sigma[0], sigma[1]], amplitude)

        if config:

            if 'baseline' in config:
                offset = config[6,0]

            if 'gain' in config:
                gain = config[2,0]

            return [mu, mu_xt, gain, offset, amplitude]

        # print (gain)
        # print (mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude)

        return [mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude]


    def bound_func(y, x, config=None, *args, **kwargs):
        if config:
            param_min = [0., 0., 0., 0., 0.]
            param_max = [np.mean(y), 1, np.inf, np.inf, np.sum(y) + np.sqrt(np.sum(y))]

        else:

            param_min = [0., 0., 0., 0., 0., 0., 0.]
            param_max = [np.mean(y), 1, np.inf, np.inf, np.inf, np.inf, np.sum(y) + np.sqrt(np.sum(y))]

        return (param_min, param_max)


    def slice_func(y, x, config=None, *args, **kwargs):
        if np.where(y != 0)[0].shape[0] == 0: return [0, 1, 1]
        return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]


    def function(p , x ,*args,config=None,**kwargs):
        p_new = [p[0],p[1],p[2],config[6,0],config[0,0],p[3],p[4]]
        return mpe_distribution_general(p_new,x)


    # Perform the actual fit
    mpes.fit(function,p0_func, slice_func, bound_func,config=spes_fit_result, limited_indices=[(0,700,),(1,700,),(2,700,),(3,700,)])

    # Save the parameters
    if options.verbose: print('--|> Save the fit result in %s' % (options.saved_histo_directory + options.saved_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_fit_filename, mpes_fit_results=mpes.fit_result)
else :
    if options.verbose: print('--|> Load the fit result from %s' % (options.saved_histo_directory + options.saved_fit_filename))
    h = np.load(options.saved_histo_directory + options.saved_fit_filename)
    mpes.fit_result = h['mpes_fit_results']
    mpes.fit_function = mpe_distribution_general


# Plot them
def slice_fun(x, **kwargs):
    return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


def show_level(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_mpe([hist], ax[1], fig, slice_fun, level,True, geom, title='', norm='lin',
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
    vis_baseline.on_pixel_clicked(700)
    plt.show()



show_level(3,mpes)

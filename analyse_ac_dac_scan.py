#!/usr/bin/env python3
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry,generate_geometry_0
from utils.plots import pickable_visu_mpe,pickable_visu_led_mu
from utils.pdf import mpe_distribution_general
from optparse import OptionParser
from utils.histogram import histogram
import peakutils
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from data_treatement import mpe_hist,synch_hist
from utils.plots import pickable_visu

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

parser.add_option("-t", "--use_saved_histo_peak", dest="use_saved_histo_peak",action="store_true",
                  help="load the peak histograms from file", default=False)

parser.add_option("-p", "--perform_fit", dest="perform_fit",action="store_false",
                  help="perform fit of mpe", default=True)

# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="list of string differing in the file name, sperated by ','", default='87,88,89,90,91' )

parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161130/")

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

parser.add_option( "--saved_histo_peak_filename", dest="saved_histo_peak_filename",
                  help="name of histo file", default='peaks.npz')

parser.add_option( "--saved_fit_filename", dest="saved_fit_filename",
                  help="name of fit file", default='fits_mpes_few.npz')

parser.add_option( "--saved_spe_fit_filename", dest="saved_spe_fit_filename",
                  help="name of spe fit file", default='darkrun_spe_fit.npz')

parser.add_option( "--saved_adc_fit_filename", dest="saved_adc_fit_filename",
                  help="name of adc fit file", default='darkrun_adc_fit.npz')

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
mpes = histogram(bin_center_min=1950.*8, bin_center_max=4095.*8, bin_width=8.,
                 data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Integrated ADC',ylabel='$\mathrm{N_{entries}}$',label='MPE')
mpes_peaks = histogram(bin_center_min=1950., bin_center_max=4095., bin_width=1.,
                       data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Peak ADC',ylabel='$\mathrm{N_{entries}}$',label='MPE')

peaks = histogram(bin_center_min=0.5, bin_center_max=51.5, bin_width=1.,
                  data_shape=((1296,)),
                  xlabel='Peak maximum position [4ns]', ylabel='$\mathrm{N_{entries}}$', label='peak position')


# Where do we take the data from
if not options.use_saved_histo_peak:
    # Loop over the files
    synch_hist.run(peaks, options)
else :
    if options.verbose:
        print('--|> Recover data from %s' % (options.saved_histo_directory+options.saved_histo_peak_filename))
    file = np.load(options.saved_histo_directory+options.saved_histo_peak_filename)
    peaks = histogram(data=file['peaks'],bin_centers=file['peaks_bin_centers'],xlabel = 'sample [$\mathrm{4 ns^{1}}$]',
                      ylabel = '$\mathrm{N_{trigger}}$',label='synchrone peak position')


def display(hists, pix_init=700):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu(hists, ax[1], fig, None, [False], 'linear', geom, title='', norm='lin',
                                 cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    peak = hists[0].data

    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))

    vis_baseline.image = np.argmax(peak,axis=1)
    # noinspection PyProtectedMember
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(pix_init)
    plt.show()


#display([peaks])

# Where do we take the data from
if not options.use_saved_histo:
    # Loop over the files
    mpe_hist.run([mpes,mpes_peaks], options, peak_positions = peaks.data)
else :
    if options.verbose: print('--|> Recover data from %s' % (options.saved_histo_directory+options.saved_histo_filename))
    file = np.load(options.saved_histo_directory+options.saved_histo_filename)
    mpes = histogram(data=file['mpes'],bin_centers=file['mpes_bin_centers'],xlabel = 'Integrated ADC in sample [4-12]',
                     ylabel='$\mathrm{N_{trigger}}$',label='MPE from integration')
    mpes_peaks = histogram(data   = file['mpes_peaks'], bin_centers=file['mpes_peaks_bin_centers'] ,
                          xlabel = 'Peak ADC',
                          ylabel = '$\mathrm{N_{trigger}}$',label='MPE from peak value')

# Fit them

if options.perform_fit:
    # recover previous fit
    if options.verbose: print(
        '--|> Recover fit results from %s' % (options.dark_calibration_directory + options.saved_spe_fit_filename))
    file = np.load(options.dark_calibration_directory + options.saved_spe_fit_filename)
    spes_fit_result = np.copy(file['spes_fit_results'])
    if options.verbose: print(
        '--|> Recover fit results from %s' % (options.dark_calibration_directory + options.saved_adc_fit_filename))
    file = np.load(options.dark_calibration_directory + options.saved_adc_fit_filename)
    adcs_fit_result = np.copy(file['adcs_fit_results'])


    def p0_func(y, x, *args, config=None, **kwargs):
        threshold = 0.05
        min_dist = 15
        peaks_index = peakutils.indexes(y, threshold, min_dist)

        if len(peaks_index) == 0:
            return [np.nan] * 7
        amplitude = np.sum(y)

        offset,gain = 0.,1.
        if type(config).__name__=='ndarray':
            offset_tmp = config[1]
            gain_tmp = config[5]
            sig_tmp = config[2]
            if np.any(np.isnan(offset_tmp)) or np.any(np.isnan(sig_tmp)) or np.any(np.isnan(gain_tmp)): return [1., 1.,1. , 1., 1.,1.,1.]
            lin_func = lambda v, off, g : off + g * v
            popt, pcov = curve_fit(lin_func, np.arange(0, peaks_index.shape[-1], 1),
                                x[peaks_index],p0=[offset_tmp[0],gain_tmp[0]],
                                sigma=(np.sqrt(y[peaks_index])),
                                bounds = ([offset_tmp[0]-3*sig_tmp[0],gain_tmp[0]-5*gain_tmp[1]],[offset_tmp[0]+3*sig_tmp[0],gain_tmp[0]+5*gain_tmp[1]]))
            offset, gain = popt[0],popt[1]
            ## if parameters are very different
            if offset/offset_tmp[0]<0.5 or offset/offset_tmp[0]>2. :
                return [1., 1.,1. , 1., 1.,1.,1.]
            if gain/gain_tmp[0]<0.5 or gain/gain_tmp[0]>2. :
                return [1., 1.,1. , 1., 1.,1.,1.]

        else:

            offset, gain = np.polynomial.polynomial.polyfit(np.arange(0, peaks_index.shape[-1], 1), x[peaks_index],
                                                            deg=1,
                                                            w=(np.sqrt(y[peaks_index])))

        sigma_start = np.zeros(peaks_index.shape[-1])
        for i in range(peaks_index.shape[-1]):
            start = max(int(peaks_index[i] - gain // 2), 0)  ## Modif to be checked
            end = min(int(peaks_index[i] + gain // 2), len(x))  ## Modif to be checked
            if start == end and end < len(x) - 2:
                end += 1
            elif start == end:
                start -= 1

            if i == 0:
                mu = -np.log(np.sum(y[start:end]) / np.sum(y))
            temp = np.average(x[start:end], weights=y[start:end])
            sigma_start[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))

        bounds = [[0., 0.], [np.inf, np.inf]]
        sigma_n = lambda x, y, n: np.sqrt(x ** 2 + n * y ** 2)
        sigma, sigma_error = curve_fit(sigma_n, np.arange(0, peaks_index.shape[-1], 1), sigma_start, bounds=bounds)
        sigma = sigma / gain

        mu_xt = np.mean(y) / mu / gain - 1

        if mu_xt < 0.: mu_xt = 0.

        return [mu, mu_xt, gain, offset, sigma[0], sigma[1], amplitude]


    def bound_func(y, x, *args, config=None, **kwargs):
        offset = config[1]
        gain = config[5]
        sig = config[2]
        if np.any(np.isnan(offset)) or np.any(np.isnan(sig)) or np.any(np.isnan(gain)): return [-np.inf]*7,[np.inf]*7
        if type(config).__name__ == 'ndarray':
            param_min = [0., 0.,gain[0]-10*gain[1] , offset[0]-3*sig[0], 1.e-2,1.e-2,0.]
            param_max = [np.inf, 1, gain[0]+10*gain[1], offset[0]+3*sig[0],10., 10.,np.inf]
        else:
            param_min = [0., 0., 0., -np.inf, 0., 0., 0.]
            param_max = [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf]

        return param_min, param_max


    def slice_func(y, x, *args, config=None, **kwargs):
        if np.where(y != 0)[0].shape[0] == 0: return [0, 1, 1]
        return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]

    prev_fit = np.append(
                       np.repeat(adcs_fit_result.reshape((1,) + adcs_fit_result.shape), mpes_peaks.data.shape[0],
                                 axis=0),
                       np.repeat(spes_fit_result.reshape((1,) + spes_fit_result.shape), mpes_peaks.data.shape[0],
                                 axis=0),
                       axis=2)
    # Perform the actual fit
    mpes_peaks.fit(mpe_distribution_general, p0_func, slice_func, bound_func,
                   config=prev_fit)
                   #,
                   #limited_indices=[(level, i,) for i in range(mpes_peaks.data.shape[1])])
    #print(options.scan_level)
    '''
    if False == True:
        for level in range(len(options.scan_level)):
            if level == 0:
                ## Take from spe fit
                for pix in range(mpes_peaks[level].shape[0]):
                    if abs(np.nanmean(mpes_peaks[level,pix])-np.nanmean(adcs[pix]))<0.1:
                        fit_result = [[0.,0.], [0.,0.], [prev_fit[level,pix,5]], [prev_fit[level,pix,1]],
                                      [prev_fit[level,pix,2]], [0.01,0.], [1000,0.]]
                    else:
                        mpes_peaks[level]._axis_fit(pix, func , p0_func(self.data[indices],self.bin_centers,config=None),
                                      slice=slice_func(self.data[indices[0]],self.bin_centers,config=None),
                                      bounds = bound_func(self.data[indices[0]],self.bin_centers,config=None))
                    ## Check if the mu is different from DC mu
                    ## if it is... then do not fit and just return the equivalent
                    ## else run the fit with input DC
                    print('spe input')
            else:
                ## Take from previous
                for pix in range(mpes_peaks[level].shape[0]):
                    ## Check wether mpes_peaks.fit_result[level-1][pix][0]> 10
                    ## If yes, fix them all except mu..
                    ## and modify the fit result to get the same shape
                    print('prev input')

'''

    # Save the parameters
    if options.verbose: print('--|> Save the fit result in %s' % (options.saved_histo_directory + options.saved_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_fit_filename, mpes_fit_results=mpes_peaks.fit_result)
else :
    if options.verbose: print('--|> Load the fit result from %s' % (options.saved_histo_directory + options.saved_fit_filename))
    h = np.load(options.saved_histo_directory + options.saved_fit_filename)
    mpes_peaks.fit_result = h['mpes_fit_results']
    mpes_peaks.fit_function = mpe_distribution_general


# Plot them
def slice_fun(x, **kwargs):
    if np.where(x != 0)[0][0]== np.where(x != 0)[0][-1]:return [0,1,1]
    return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


def show_level(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_mpe([hist], ax[1], fig, slice_fun, level,True, geom, title='', norm='lin',
                                     cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    val = hist.fit_result[3,:,2,0]
    val[np.isnan(val)]=0
    val[val<1.]=1.
    val[val>10.]=10.
    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = val
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(700)
    plt.show()



show_level(0,mpes_peaks)



def show_mu(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_led_mu([hist], ax[1], fig, slice_fun, level,True, geom, title='', norm='lin',
                                     cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    val = hist.fit_result[level,:,2,0]
    val[np.isnan(val)]=0
    val[val<1.]=1.
    val[val>10.]=10.
    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = val
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(700)
    plt.show()


from matplotlib.colors import LogNorm

def display_fitparam(hist,param_ind,pix,param_label,range=[0.9,1.1]):
    fig, ax = plt.subplots(1,2)
    param = hist.fit_result[:, :, param_ind, 0]
    param_err = hist.fit_result[:, :, param_ind, 1]
    param_ratio = np.divide(param, param_err[0])
    param_ratio[np.isnan(param_ratio)]=0.
    print(param_ratio.shape)
    plt.subplot(1,2,1)
    plt.errorbar(np.arange(50, 260, 10), param_ratio[:,pix], yerr=param_err[:,pix], fmt='ok')
    plt.ylim(range)
    plt.ylabel(param_label)
    plt.xlabel('AC LED DAC')
    #plt.axes().xaxis.get_label().set_ha('right')
    #plt.axes().xaxis.get_label().set_position((1, 0))
    #plt.axes().yaxis.get_label().set_ha('right')
    #plt.axes().yaxis.get_label().set_position((0, 1))
    plt.subplot(1,2,2)
    xaxis = np.repeat(np.arange(50, 260, 10).reshape((1,)+np.arange(50, 260, 10).shape),param.shape[1],axis=0).reshape(np.prod([param_ratio.shape]))
    y_axis = param_ratio.reshape(np.prod([param_ratio.shape]))
    plt.hist2d(xaxis,y_axis,bins=20,range=[[50, 250], range],norm=LogNorm())
    plt.colorbar()
    plt.show()

def display_fitparam_err(hist,param_ind,pix,param_label,range=[0.9,1.1]):
    fig, ax = plt.subplots(1,2)
    param = hist.fit_result[:, :, param_ind, 0]
    param_err = hist.fit_result[:, :, param_ind, 1]
    param_ratio = np.divide(param, param[np.nanargmin(param_err,axis=0)])
    param_ratio[np.isnan(param_ratio)]=0.
    print(param_ratio.shape)
    plt.subplot(1,2,1)
    plt.plot(np.arange(50, 260, 10), param_err[:,pix], color='k')
    plt.ylim(range)
    plt.ylabel(param_label)
    plt.xlabel('AC LED DAC')
    #plt.axes().xaxis.get_label().set_ha('right')
    #plt.axes().xaxis.get_label().set_position((1, 0))
    #plt.axes().yaxis.get_label().set_ha('right')
    #plt.axes().yaxis.get_label().set_position((0, 1))
    plt.subplot(1,2,2)
    xaxis = np.repeat(np.arange(50, 260, 10).reshape((1,)+np.arange(50, 260, 10).shape),param.shape[1],axis=0).reshape(np.prod([param_ratio.shape]))
    y_axis = param_ratio.reshape(np.prod([param_ratio.shape]))
    plt.hist2d(xaxis,y_axis,bins=20,range=[[50, 250], range],norm=LogNorm())
    plt.colorbar()
    plt.show()

#display_fitparam_err(mpes_peaks,2,700,'Error Gain',[0.9,1.1])
#show_mu(0,mpes_peaks)
#display_fitparam(mpes_peaks,1,700,'$\mu_{XT}$',[0.5,2.]) #<N(p.e.)>@DAC=x
display_fitparam(mpes_peaks,2,700,'Gain',[0.9,1.1]) #<N(p.e.)>@DAC=x
#!/usr/bin/env python3
from utils.geometry import generate_geometry,generate_geometry_0
from utils.plots import pickable_visu_mpe,pickable_visu_led_mu
from utils.pdf import mpe_distribution_general,mpe_distribution_general_sh
from optparse import OptionParser
from utils.histogram import histogram
import peakutils
from matplotlib import pyplot as plt
import numpy as np
from utils.plots import pickable_visu
from specta_fit import fit_low_light,fit_hv_off,fit_dark,fit_high_light

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

# Setup configuration
parser.add_option("--cts_sector", dest="cts_sector",
                  help="Sector covered by CTS", default=3,type=int)

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
                  help="input directory", default="/data/20161130/")

parser.add_option( "--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")

parser.add_option( "--calibration_filename", dest="calibration_filename",
                  help="calibration file name", default="calib_spe.npz")

parser.add_option( "--calibration_directory", dest="calibration_directory",
                  help="calibration file directory", default="data/DarkRun/")

parser.add_option( "--saved_histo_directory", dest="saved_histo_directory",
                  help="directory of histo file", default='data/LevelScan/20161130/')

parser.add_option( "--saved_histo_filename", dest="saved_histo_filename",
                  help="name of histo file", default='mpes_few.npz')

parser.add_option( "--saved_histo_peak_filename", dest="saved_histo_peak_filename",
                  help="name of histo file", default='peaks.npz')

parser.add_option( "--saved_fit_filename", dest="saved_fit_filename",
                  help="name of fit file", default='fits_mpes_few_new.npz')

parser.add_option( "--saved_spe_fit_filename", dest="saved_spe_fit_filename",
                  help="name of spe fit file", default='darkrun_spe_fit.npz')

parser.add_option( "--saved_adc_fit_filename", dest="saved_adc_fit_filename",
                  help="name of adc fit file", default='darkrun_adc_fit.npz')

parser.add_option( "--dark_calibration_directory", dest="dark_calibration_directory",
                  help="darkrun calibration file directory", default="data/DarkRun/20161130/")
parser.add_option("--saved_adc_histo_filename", dest="saved_adc_histo_filename",
                  help="name of histo file", default='darkrun_adc_hist.npz')

# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')
options.scan_level = [int(level) for level in options.scan_level.split(',')]
if len(options.scan_level)==3:
    options.scan_level=np.arange(options.scan_level[0],options.scan_level[1]+options.scan_level[2],options.scan_level[2])

# Define Geometry
sector_to_angle = {1:120.,2:240.,3:0.} #TODO check and put it in cts
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
    if options.verbose:
        print('--|> Recover data from %s' % (options.dark_calibration_directory + options.saved_adc_histo_filename))
    file = np.load(options.dark_calibration_directory + options.saved_adc_histo_filename)
    adcs = histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))


    prev_fit = np.append( adcs_fit_result.reshape((1,) + adcs_fit_result.shape),
                          spes_fit_result.reshape((1,) + spes_fit_result.shape),axis=2)
    # reodred (this will disapear once dark fit is implemented properly)
    # amp0,baseline,sigma_e, sigma_e,sigma_i,gain,amp1,amp2,amp3,baseline,offset,amp4,amp5
    prev_fit[...,[0,1,2,3,4,5,6,7,8,9,10,11,12],:] = prev_fit[...,[12,11,5,1,2,4,0,10,3,6,7,8,9],:]
    prev_fit = np.delete(prev_fit,[8,9,10,11,12],axis=2)
    # fix the cross talk for now...
    prev_fit[...,1,:]=[0.08,10.]
    print(prev_fit.shape)
    #print(options.scan_level)

    #intialise the fit result
    tmp_shape = prev_fit.shape
    tmp_shape=mpes_peaks.data.shape[:1]+tmp_shape[1:]
    print(tmp_shape)
    mpes_peaks.fit_result = np.ones(tmp_shape)*np.nan

    for level in range(len(options.scan_level)):
        if options.verbose: print("################################# Level",level)
        if level == 0:
            ## Take from spe fit
            for pix in range(mpes_peaks.data[level].shape[0]):
                if np.any(np.isnan(mpes_peaks.data[level, pix])):
                    if options.verbose: print('----> Pix',pix,'abort')
                    continue
                elif abs(np.nanmean(mpes_peaks.data[level, pix]) - np.nanmean(adcs.data[pix])) < 0.1:
                    if options.verbose: print('----> Pix',pix,'as dark')
                    ## to be replaced by fit_dark
                    mpes_peaks.fit_result[level,pix] = [[2., 1.e3], [prev_fit[0, pix, 1]], [prev_fit[0, pix, 2]], [prev_fit[0, pix, 3]],
                                  [prev_fit[0, pix, 4]], [0.7, 10.], [1000, 1.e8],[prev_fit[0, pix, 7]]]
                else:
                    if options.verbose: print('----> Pix',pix,'low light')
                    mpes_peaks.fit_result[level, pix] = \
                        mpes_peaks._axis_fit((level,pix,),
                                                    fit_low_light.fit_func,
                                                    fit_low_light.p0_func(mpes_peaks.data[level,pix],
                                                                          mpes_peaks.bin_centers,
                                                                          config=prev_fit[0,pix]),
                                                    slice=fit_low_light.slice_func(mpes_peaks.data[level,pix],
                                                                                   mpes_peaks.bin_centers,
                                                                                   config=prev_fit[0,pix]),
                                                    bounds=fit_low_light.bounds_func(mpes_peaks.data[level,pix],
                                                                                     mpes_peaks.bin_centers,
                                                                                     config=prev_fit[0,pix]),
                                                    # mu_xt,baseline,sigma_e,offset
                                                    fixed_param = np.array([[i for i in [1,3,4,7]],[prev_fit[0, pix, i , 0] for i in [1,3,4,7]]])
                                                    )
        else:
            ## Take from previous
            for pix in range(mpes_peaks.data[level].shape[0]):
                if np.any(np.isnan(mpes_peaks.data[level, pix])):
                    if options.verbose: print('----> Pix',pix,'abort')
                    continue

                elif abs(np.nanmean(mpes_peaks.data[level, pix]) - np.nanmean(adcs.data[pix])) < 0.1:
                    if options.verbose: print('----> Pix',pix,'as dark')
                    ## to be replaced by fit_dark
                    mpes_peaks.fit_result[level,pix] = [[2., 1.e3], [prev_fit[0, pix, 1]], [prev_fit[0, pix, 2]], [prev_fit[0, pix, 3]],
                                  [prev_fit[0, pix, 4]], [0.7, 10.], [1000, 1.e8],[prev_fit[0, pix, 7]]]

                elif mpes_peaks.fit_result[level-1,pix,0,0]<10.:
                    if options.verbose: print('----> Pix',pix,'low light')
                    mpes_peaks.fit_result[level, pix] = \
                        mpes_peaks._axis_fit((level,pix,),
                                                    fit_low_light.fit_func,
                                                    fit_low_light.p0_func(mpes_peaks.data[level, pix],
                                                                          mpes_peaks.bin_centers,
                                                                          config=mpes_peaks.fit_result[level-1, pix]),
                                                    slice=fit_low_light.slice_func(mpes_peaks.data[level, pix],
                                                                                   mpes_peaks.bin_centers,
                                                                                   config=mpes_peaks.fit_result[level-1, pix]),
                                                    bounds=fit_low_light.bounds_func(mpes_peaks.data[level, pix],
                                                                                     mpes_peaks.bin_centers,
                                                                                     config=mpes_peaks.fit_result[level-1, pix]),
                                                    # mu_xt,baseline,sigma_e,offset
                                                    fixed_param = np.array([[i for i in [1,3,4,7]],[mpes_peaks.fit_result[level-1, pix, i , 0] for i in [1,3,4,7]]])
                                                    )
                    print(mpes_peaks.fit_result[level, pix])

                else:
                    if options.verbose: print('----> Pix',pix,'high light')
                    mpes_peaks.fit_result[level, pix] = \
                        mpes_peaks._axis_fit((level,pix,),
                                                    fit_high_light.fit_func,
                                                    fit_high_light.p0_func(mpes_peaks.data[level, pix],
                                                                          mpes_peaks.bin_centers,
                                                                          config=mpes_peaks.fit_result[level - 1, pix]),
                                                    slice=fit_high_light.slice_func(mpes_peaks.data[level, pix],
                                                                                   mpes_peaks.bin_centers,
                                                                                   config=mpes_peaks.fit_result[
                                                                                       level - 1, pix]),
                                                    bounds=fit_high_light.bounds_func(mpes_peaks.data[level, pix],
                                                                                     mpes_peaks.bin_centers,
                                                                                     config=mpes_peaks.fit_result[
                                                                                         level - 1, pix]),
                                                    # fix all but mu and amplitude
                                                    fixed_param=np.array([[i for i in [1,2,3,4,5,7]],[mpes_peaks.fit_result[level-1, pix, i , 0] for i in [1,2,3,4,5,7]]])
                                                    )


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
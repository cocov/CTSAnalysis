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
                  help="Sector covered by CTS", default="1")
parser.add_option("-l", "--scan_level", dest="scan_level",
                  help="list of scans DC level, separated by ',', if only three argument, min,max,step", default="50,100,10")

parser.add_option("-s", "--use_saved_histo", dest="use_saved_histo",
                  help="load the histograms from file", default=False)



# File management
parser.add_option("-f", "--file_list", dest="file_list",
                  help="list of string differing in the file name, sperated by ','", default='87,88' )
parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/LevelScan/20161130/")
parser.add_option( "--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")
parser.add_option( "--calibration_filename", dest="calibration_filename",
                  help="calibration file name", default="calib_spe.npz")
parser.add_option( "--calibration_directory", dest="calibration_directory",
                  help="calibration file directory", default="/data/datasets/CTA/DarkRun/")
parser.add_option( "saved_histo_directory", dest="saved_histo_directory",
                  help="directory of histo file", default='/data/datasets/CTA/LevelScan/20161130/')
parser.add_option( "saved_histo_filename", dest="saved_histo_filename",
                  help="directory of histo file", default='mpes.npz')

# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')
options.scan_level = [int(level) for level in options.scan_level.split(',')]

if len(options.scan_level)==3:
    options.scan_level=np.arange(options.scan_level[0],options.scan_level[1],options.scan_level[2])

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

calib = remap_conf_dict(calib_file)

# Prepare the mpe histograms
mpes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)))
peakes = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(options.scan_level.shape+(1296,)))

n_batch, batch_num, max_evt = 10, 0, 100
batch_peakpos = np.ones(options.scan_level.shape+(1296,1))*np.nan
batch_integration = np.empty(options.scan_level.shape+(1296,1))*np.nan

if not options.use_saved_histo:
    # Loop over the files
    for file in options.file_list:
        # Get the file
        _url = options.directory+options.file_basename%(file)
        inputfile_reader = zfits.zfits_event_source( url= _url
                                                     ,data_type='r1'
                                                     , max_events=100000)
        print('--|> Will process %d events from %s'%(max_evt,_url))

        # Loop over event in this file
        for event in inputfile_reader:
            if event.event_id > max_evt: break
            for telid in event.r1.tels_with_data:
                data = event.r1.tel[telid].adc_samples.values()
                data = data -  calib_file['baseline'][:,0].reshape(data.shape[0],1)
                # now integrate
                integration, window, peakpos = integrators.full_integration(data)

                if np.all(np.isnan(batch_peakpos[0][:,0]):
                    batch_peakpos[:,1] = peakpos[0]
                else:
                    batch = np.append(batch,data.reshape(data.shape[0],1,data.shape[1]),axis = 1)

                if type(batch_spe).__name__!='ndarray':
                    batch = data.reshape(data.shape[0],1,data.shape[1])
                else:
                    batch = np.append(batch,data.reshape(data.shape[0],1,data.shape[1]),axis = 1)


    # Creating the histograms

    # Reading the file
    n_evt,n_batch,batch_num,max_evt=0,1000,0,30000

    batch = None

    print('--|> Reading  the batch #%d of %d events' % (batch_num, n_batch))

    for event in inputfile_reader:
        if event.r1.event_id > max_evt: break
        if (n_evt-n_batch*1000)%10==0:print("Progress {:2.1%}".format(float(n_evt - batch_num*n_batch) / n_batch), end="\r")
        for telid in event.r1.tels_with_data:
            print(event.r1.tel[telid].eventNumber, event.r1.event_id)
            if n_evt%n_batch == 0:
                print('--|> Treating the batch #%d of %d events'%( batch_num,n_batch))
                # Update adc histo

                adcs.fill_with_batch(batch.reshape(batch.shape[0],batch.shape[1]*batch.shape[2]))
                # Update the spe histo
                if calib_unknown:
                    spes.fill_with_batch(spe_peaks_in_event_list(batch, None, None, l=1))
                else:
                    variance = calib['sigma_e'][:,0]
                    variance[np.isnan(variance)]=0
                    baseline = calib['baseline'][:,0]
                    baseline[np.isnan(baseline)]=0
                    if apply_calib:
                        spes.fill_with_batch(spe_peaks_in_event_list(batch, None, variance, l=1))
                    else:
                        spes.fill_with_batch(spe_peaks_in_event_list(batch, baseline, variance, l=1))

                # Reset the batch
                batch = None
                batch_num+=1
                print('--|> Reading  the batch #%d of %d events'%( batch_num,n_batch))

            # Get the data
            data = np.array(list(event.r1.tel[telid].adc_samples.values()))
            # Subtract baseline if calibration exists
            if 'baseline' in calib:
                pedestals = calib['baseline'][:,0].reshape(1296,1)
                pedestals[np.isnan(pedestals)]=0
                if apply_calib : data = data -pedestals
            # Append the data to the batch
            if type(batch).__name__!='ndarray':
                batch = data.reshape(data.shape[0],1,data.shape[1])
            else:
                batch = np.append(batch,data.reshape(data.shape[0],1,data.shape[1]),axis = 1)

    file_name = "histos_subtracted.npz" if apply_calib else "histos.npz"

    print('--|> Save in /data/datasets/DarkRun/%s' % (file_name))
    np.savez_compressed("/data/datasets/DarkRun/"+file_name, adcs=adcs.data ,adcs_bin_centers = adcs.bin_centers, spes = spes.data, spes_bin_centers = spes.bin_centers )
else:
    file_name = "histos_subtracted.npz" if apply_calib else "histos.npz"
    print('--|> Recover data from /data/datasets/DarkRun/%s' % (file_name))
    file = np.load("/data/datasets/DarkRun/"+file_name)
    adcs = histogram(data=file['adcs'],bin_centers=file['adcs_bin_centers'])
    spes = histogram(data=file['spes'],bin_centers=file['spes_bin_centers'])

fit_result = None

if calib_unknown:
    # Define the function getting the initial fit value, the parameters bound and the slice to fit
    def p0_func(x,config):
        norm = np.sum(x)
        mean = np.argmax(x)
        sigma = 0.9
        return [norm,mean,sigma]

    def bound_func(x,xrange,config):
        min_norm, max_norm = 1., np.inf
        min_mean, max_mean = xrange[np.argmax(x) - 2], xrange[np.argmax(x) + 2]
        min_sigma, max_sigma = 1e-6, 1e6
        return ([min_norm, min_mean, min_sigma], [max_norm, max_mean, max_sigma])

    def slice_func(x,config):
        return [max(np.argmax(x) -20,0),np.argmax(x) + 1,1]

    print('--|> Compute baseline and sigma_e from ADC distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = adcs.fit( gaussian_residual , p0_func, slice_func, bound_func)
    calib['baseline'] = fit_result[:, 1]
    calib['sigma_e'] = fit_result[:, 2]
    calib['norm'] = fit_result[:, 0]

    print('--|> Save in /data/datasets/DarkRun/calib.npz')
    np.savez_compressed("/data/datasets/DarkRun/calib.npz", sigma_e=calib['sigma_e'], baseline=calib['baseline'], norm=calib['norm'])


if perform_spe_fit:
    # Define the function getting the initial fit value, the parameters bound and the slice to fit
    def p0_func(x,config):
        return [1000000.   ,0.8, 0.1 , 5.6, 100000.,10000.,1000. , config['baseline']      , 0.,0.8      , 100. ,10.]

    def bound_func(x,xrange,config):
        param_min = [0.    ,0.01, 0.01, 0. , 10.   , 1.    , 0.    , config['baseline'] - 10.,-10.,0.8*0.1, 0.    ,0.]
        param_max = [np.inf,8. , 5. , 100., np.inf, np.inf, np.inf, config['baseline'] + 10.,10. ,0.8*10., np.inf,np.inf]
        return (param_min, param_max)

    def slice_func(x,config):
        if np.where(x != 0)[0].shape[0]==0: return[0,1,1]
        return [np.where(x != 0)[0][0]+3,np.where(x != 0)[0][-1],1]

    def remap_conf_dict(config):
        new_conf = []
        for i,pix in enumerate(config[list(config.keys())[0]]):
            new_conf.append({})
        for key in list(config.keys()):
            for i,pix in enumerate(config[key]):
                if np.isfinite(pix[0]):
                    new_conf[i][key]=pix[0]
                else:new_conf[i][key]=0.
        return new_conf

    print('--|> Compute Gain, sigma_e, sigma_i from SPE distributions')
    # Fit the baseline and sigma e of all pixels
    fit_result = spes.fit(multi_gaussian_residual_with0, p0_func, slice_func, bound_func,config=remap_conf_dict(calib))
    calib_spe ={}
    calib_spe['gain'] = fit_result[:, 3]
    calib_spe['sigma_e_spe'] = fit_result[:, 1]
    calib_spe['sigma_i'] = fit_result[:, 2]
    calib_spe['baseline_spe'] = fit_result[:, 7]
    calib_spe['sigma_e'] = calib['sigma_e']
    calib_spe['baseline'] = calib['baseline']
    calib_spe['norm'] = calib['norm']
    calib_spe['full_spe_fitres']=fit_result
    print('--|> Save in /data/datasets/DarkRun/calib_spe.npz')
    np.savez_compressed("/data/datasets/DarkRun/calib_spe.npz",
                        sigma_e_spe=calib_spe['sigma_e_spe'],
                        sigma_e=calib_spe['sigma_e'],
                        sigma_i=calib_spe['sigma_i'],
                        baseline_spe=calib_spe['baseline_spe'],
                        baseline=calib_spe['baseline'],
                        gain=calib_spe['gain'],
                        norm=calib_spe['norm'],
                        full_spe_fitres = calib_spe['full_spe_fitres'])

    print('--|> Recover baseline and sigma_e from /data/datasets/DarkRun/calib_spe.npz')
    calib = np.load("/data/datasets/DarkRun/calib_spe.npz")


## Plot the baseline extraction
def slice_fun(x,**kwargs):
    return [np.where(x != 0)[0][0],np.where(x != 0)[0][-1],1]

fig1, axs1 = plt.subplots(1, 2,figsize=(30, 10))
plt.subplot(1, 2 ,1)
vis_baseline = pickable_visu([adcs,spes],axs1[1],fig1,slice_fun,calib,apply_calib,geom, title='',norm='lin', cmap='viridis',allow_pick=True)
vis_baseline.add_colorbar()
vis_baseline.colorbar.set_label('Gain [ADC/p.e.]')
plt.subplot(1, 2 ,1)
h = np.copy(calib['gain'][:,0])
h_err = np.copy(calib['gain'][:,1])
h[np.isnan(h_err)]=2
h[h>20]=2

ba =calib['baseline_spe'][:,1]
h[ba>100]=2
h[h<3]=3
h[h>6]=6

vis_baseline.axes.xaxis.get_label().set_ha('right')
vis_baseline.axes.xaxis.get_label().set_position((1,0))
vis_baseline.axes.yaxis.get_label().set_ha('right')
vis_baseline.axes.yaxis.get_label().set_position((0,1))
vis_baseline.image = h
fig1.canvas.mpl_connect('pick_event', vis_baseline._on_pick )
vis_baseline.on_pixel_clicked(374)

plt.subplots(2, 2,figsize=(15, 18))

ax= plt.subplot(2, 2 ,1,xlabel='Gain [ADC/p.e.]',ylabel='$\mathrm{N_{pixel}}$')
hh = np.copy(calib['gain'][:,0])
hh_fiterr= np.copy(calib['gain'][:,1])
hh_fin=hh[np.isfinite(hh_fiterr)]
hist ,bin = np.histogram(hh_fin,bins=np.arange(-0.05,10.15,0.1))
gain = histogram( data=hist.reshape(1,hist.shape[0]),bin_centers=np.arange(0.,10.1,0.1))
plt.errorbar(x=gain.bin_centers,y=gain.data[0],yerr = gain.errors[0], fmt = 'o')
gain.predef_fit('Gauss',x_range=[5.,6.],initials=[200.,5.5,0.3])
ax.plot(gain.fit_axis,gain.fit_function(gain.fit_result[0][:,0],gain.fit_axis))
ax.text(0.6,0.9,'$\mu$=%1.3f $\pm$ %1.3f\n$\sigma$=%1.3f $\pm$ %1.3f' % (
        gain.fit_result[0][1][0], gain.fit_result[0][1][1],
        gain.fit_result[0][2][0], gain.fit_result[0][2][1]),
        fontsize=15, transform=ax.transAxes, va='top', )
ax.xaxis.get_label().set_ha('right')
ax.xaxis.get_label().set_position((1,0))
ax.yaxis.get_label().set_ha('right')
ax.yaxis.get_label().set_position((0,1))

axs2=plt.subplot(2, 2 ,2)
axs2.set_xlabel('$\sigma_{e}$')
axs2.set_ylabel('$\mathrm{N_{pixel}}$')
hh = np.copy(calib['sigma_e_spe'][:,0])
hh_fiterr= np.copy(calib['sigma_e_spe'][:,1])
hh_fin=hh[np.isfinite(hh_fiterr)]
hist ,bin = np.histogram(hh_fin,bins=np.arange(-0.05,2.15,0.1))
sigma_e = histogram( data=hist.reshape(1,hist.shape[0]),bin_centers=np.arange(0.,2.1,0.1))
plt.errorbar(x=sigma_e.bin_centers,y=sigma_e.data[0],yerr = sigma_e.errors[0], fmt = 'o')
axs2.xaxis.get_label().set_ha('right')
axs2.xaxis.get_label().set_position((1,0))
axs2.yaxis.get_label().set_ha('right')
axs2.yaxis.get_label().set_position((0,1))

axs3=plt.subplot(2, 2 ,3)
axs3.set_xlabel('$\sigma_{i}$')
axs3.set_ylabel('$\mathrm{N_{pixel}}$')
hh = np.copy(calib['sigma_i'][:,0])
hh_fiterr= np.copy(calib['sigma_i'][:,1])
hh_fin=hh[np.isfinite(hh_fiterr)]
hist ,bin = np.histogram(hh_fin,bins=np.arange(-0.05,2.15,0.1))
sigma_i = histogram( data=hist.reshape(1,hist.shape[0]),bin_centers=np.arange(0.,2.1,0.1))
plt.errorbar(x=sigma_i.bin_centers,y=sigma_i.data[0],yerr = sigma_i.errors[0], fmt = 'o')
axs3.xaxis.get_label().set_ha('right')
axs3.xaxis.get_label().set_position((1,0))
axs3.yaxis.get_label().set_ha('right')
axs3.yaxis.get_label().set_position((0,1))


axs4=plt.subplot(2, 2 ,4)
axs4.set_xlabel('Baseline')
axs4.set_ylabel('$\mathrm{N_{pixel}}$')
hh = np.copy(calib['baseline'][:,0])
hh_fiterr= np.copy(calib['baseline'][:,1])
hh_fin=hh[np.isfinite(hh_fiterr)]
hist ,bin = np.histogram(hh_fin,bins=np.arange(-0.5,4096.5,1.))
baseline = histogram( data=hist.reshape(1,hist.shape[0]),bin_centers=np.arange(0.,4096,1.))
plt.errorbar(x=baseline.bin_centers,y=baseline.data[0],yerr = baseline.errors[0], fmt = 'o')
axs4.xaxis.get_label().set_ha('right')
axs4.xaxis.get_label().set_position((1,0))
axs4.yaxis.get_label().set_ha('right')
axs4.yaxis.get_label().set_position((0,1))


plt.subplots(1, 1,figsize=(15, 10))
plt.subplot(1, 1 ,1)
vis_si = visualization.CameraDisplay(geom, title='',norm='lin', cmap='viridis')
vis_si.add_colorbar()
vis_si.colorbar.set_label('$\sigma_e+\sigma_i$')
h3 = np.copy(calib['sigma_i'][:,0])
h4 = np.copy(calib['sigma_e'][:,0])
h3 = np.add(h3,h4)
h_err = np.copy(calib['gain'][:,1])
h3[np.isnan(h_err)]=1.2
h3[h3>2.3]=1.2
ba =calib['baseline_spe'][:,1]
h3[ba>100]=1.2

vis_si.axes.xaxis.get_label().set_ha('right')
vis_si.axes.xaxis.get_label().set_position((1,0))
vis_si.axes.yaxis.get_label().set_ha('right')
vis_si.axes.yaxis.get_label().set_position((0,1))

vis_si.image = h3

plt.show()
from ctapipe import visualization
from utils.fitting import gaussian
from utils.fitting import get_poisson_err
from utils.fitting import multi_gaussian_residual_with0
from utils.fitting import multi_gaussian_with0
from numpy.linalg import inv
import numpy as np

class pickable_visu(visualization.CameraDisplay):
    def __init__(self,pickable_datas,extra_plot,figure,slice_func,config,apply_calib,*args, **kwargs):
        super(pickable_visu, self).__init__(*args, **kwargs)
        self.pickable_datas = pickable_datas
        self.extra_plot = extra_plot
        self.figure=figure
        self.slice_func = slice_func
        self.config=config
        self.apply_calib = apply_calib

    def on_pixel_clicked(self, pix_id):

        legend_handles = []
        self.extra_plot.cla()
        for i,pickable_data in enumerate(self.pickable_datas):
            col = 'r' if i==0 else 'b'
            slice = self.slice_func(pickable_data.data[pix_id])
            label = 'Raw ADC distribution'
            if i==1:
                slice[0]=np.argmax(pickable_data.data[pix_id])+3
                label = 'Peak ADC distribution'

            adcs = self.extra_plot.errorbar(pickable_data.bin_centers[slice[0]:slice[1]:slice[2]],
                                     pickable_data.data[pix_id][slice[0]:slice[1]:slice[2]],
                                     yerr=np.vectorize(get_poisson_err)(
                                         pickable_data.data[pix_id][slice[0]:slice[1]:slice[2]]), fmt='o'+col,label=label)
            legend_handles.append(adcs)


            if i ==0:
                self.extra_plot.set_yscale('log')
                self.extra_plot.set_ylim(1.e-1,np.max(pickable_data.data[pix_id][slice[0]:slice[1]:slice[2]]) * 2)

                self.extra_plot.set_xlabel('ADC')
                self.extra_plot.set_ylabel("N / ADC")


            if i == 1 :
                label = 'Fit to Peak ADC'
                x_fit = np.linspace(pickable_data.bin_centers[slice[0]], pickable_data.bin_centers[slice[1]], 200)
                fit, = self.extra_plot.plot(x_fit, multi_gaussian_with0(self.config['full_spe_fitres'][pix_id][:,0], x_fit),label=label)
                self.extra_plot.text(0.5, 0.9,
                                     'GAIN=%1.3f $\pm$ %1.3f\n$\sigma_e$=%1.3f $\pm$ %1.3f\n$\sigma_i$=%1.3f $\pm$ %1.3f\nbaseline=%1.3f $\pm$ %1.3f' % (
                                         self.config['full_spe_fitres'][pix_id][3][0], self.config['full_spe_fitres'][pix_id][3][1],
                                         self.config['full_spe_fitres'][pix_id][1][0], self.config['full_spe_fitres'][pix_id][1][1],
                                         self.config['full_spe_fitres'][pix_id][2][0], self.config['full_spe_fitres'][pix_id][2][1],
                                         self.config['full_spe_fitres'][pix_id][7][0], self.config['full_spe_fitres'][pix_id][7][1]),
                                     fontsize=20, transform=self.extra_plot.transAxes, va='top')
                legend_handles.append(fit)

        try:
            self.extra_plot.legend(handles=legend_handles, loc=3)
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')



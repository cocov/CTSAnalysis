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

        self.extra_plot.cla()
        for i,pickable_data in enumerate(self.pickable_datas):
            col = 'r' if i==0 else 'b'
            slice = self.slice_func(pickable_data.data[pix_id])
            if i==1:
                slice[0]=np.argmax(pickable_data.data[pix_id])+3
            self.extra_plot.errorbar(pickable_data.bin_centers[slice[0]:slice[1]:slice[2]],
                                     pickable_data.data[pix_id][slice[0]:slice[1]:slice[2]],
                                     yerr=np.vectorize(get_poisson_err)(
                                         pickable_data.data[pix_id][slice[0]:slice[1]:slice[2]]), fmt='o'+col)
            if i ==0:
                self.extra_plot.set_yscale('log')
                self.extra_plot.set_ylim(1.e-1,np.max(pickable_data.data[pix_id][slice[0]:slice[1]:slice[2]]) * 2)

            if i == 0 :
                '''

                if self.apply_calib:
                    self.extra_plot.plot(x_fit,
                                         gaussian(
                                             [self.config['norm'][pix_id][0], 0., self.config['sigma_e'][pix_id][0]],
                                             x_fit))
                else:
                    self.extra_plot.plot(x_fit, gaussian(
                        [self.config['norm'][pix_id][0], self.config['baseline'][pix_id][0],
                         self.config['sigma_e'][pix_id][0]], x_fit))
                '''
                self.extra_plot.text(0.5, 0.9,
                                     'baseline=%1.3f $\pm$ %1.3f\n$\sigma_e$=%1.3f $\pm$ %1.3f' % (
                                         self.config['baseline'][pix_id][0], self.config['baseline'][pix_id][1],
                                         self.config['sigma_e'][pix_id][0], self.config['sigma_e'][pix_id][1]),
                                     fontsize=20, transform=self.extra_plot.transAxes, va='top', )

            if i == 1 :
                def my_func(param,x,*args,**kwargs):
                    p_new = [0.,
                             param[0],
                             param[1],
                             param[2],
                             param[3],
                             param[4],
                             param[5],
                             param[6],
                             0.,
                             1.,
                             param[7]
                             ]
                    return multi_gaussian_residual_with0(p_new,x ,*args, **kwargs)
                def my_func2(param,x):
                    param = [0.,
                             param[0],
                             param[1],
                             param[2],
                             param[3],
                             param[4],
                             param[5],
                             param[6],
                             0.,
                             1.,
                             param[7]
                             ]
                    return multi_gaussian_with0(param,x )

                p0 = [self.config['norm'][pix_id][0],0.8,0.5 , 5.6, 10000., 1000., 100.,self.config['baseline'][pix_id][0],-1.,self.config['sigma_e'][pix_id][0],10.]
                param_min =  [self.config['norm'][pix_id][0]*0.01,0.08,0.0001 , 3., 10., 1., 0.,self.config['baseline'][pix_id][0]-10.,-10.,self.config['sigma_e'][pix_id][0]*0.1,0.]
                param_max =  [self.config['norm'][pix_id][0]*100.,8.,5. , 10., np.inf, np.inf, np.inf,self.config['baseline'][pix_id][0]+10.,10.,self.config['sigma_e'][pix_id][0]*10.,np.inf]

                p0 =  [0.8,0.0005 , 5.6, 10000., 1000., 100.,self.config['baseline'][pix_id][0],10.]
                param_min =  [0.08,0.0001 , 3., 10., 1., 0.,self.config['baseline'][pix_id][0]-10.,0.]
                param_max =  [8.,5. , 10., np.inf, np.inf, np.inf,self.config['baseline'][pix_id][0]+10.,np.inf]


                bound_param = (param_min,param_max)
                fit_result = pickable_data._axis_fit( idx=pix_id, func=my_func, p0=p0, slice=slice,bounds=bound_param)
                self.extra_plot.text(0.5, 0.7, 'GAIN=%1.3f $\pm$ %1.3f\n$\sigma_e$=%1.3f $\pm$ %1.3f\n$\sigma_i$=%1.3f $\pm$ %1.3f\nbaseline=%1.3f $\pm$ %1.3f' % (
                    fit_result[0][2][0], fit_result[0][2][1],
                    fit_result[0][0][0], fit_result[0][0][1],
                    fit_result[0][1][0], fit_result[0][1][1],
                    fit_result[0][6][0], fit_result[0][6][1]),
                                     fontsize=20, transform=self.extra_plot.transAxes, va='top')
                x_fit = np.linspace(pickable_data.bin_centers[slice[0]], pickable_data.bin_centers[slice[1]], 200)
                self.extra_plot.plot(x_fit, my_func2(fit_result[0][:,0], x_fit))
                ## TODO change the peak at 0....
                print(fit_result)
                #except Exception:
                #    print("Problem")

        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')



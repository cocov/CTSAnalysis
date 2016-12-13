from ctapipe import visualization
from utils.fitting import gaussian
from utils.fitting import get_poisson_err
from utils.fitting import multi_gaussian_residual_with0
from utils.fitting import multi_gaussian_with0
from numpy.linalg import inv
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Button

class pickable_visu(visualization.CameraDisplay):
    def __init__(self,pickable_datas,extra_plot,figure,slice_func,show_fit,axis_scale,*args, **kwargs):
        super(pickable_visu, self).__init__(*args, **kwargs)
        self.pickable_datas = pickable_datas
        self.extra_plot = extra_plot
        self.figure=figure
        self.slice_func = slice_func
        self.show_fit = show_fit
        self.axis_scale = axis_scale

    def on_pixel_clicked(self, pix_id):
        self.extra_plot.cla()
        colors = ['k','r','b']
        for i,pickable_data in enumerate(self.pickable_datas):
            slice = self.slice_func(pickable_data.data[pix_id]) if self.slice_func else [0,pickable_data.bin_centers.shape[0],1]
            init_func = pickable_data.fit_function
            if i==1:
                init_func = pickable_data.fit_function
                func = lambda p, x : init_func(p,x,self.pickable_datas[0].fit_result[pix_id])
                pickable_data.fit_function = func

            pickable_data.show(which_hist=(pix_id,), axis=self.extra_plot,
                               show_fit=self.show_fit[i], slice=slice,
                               scale= self.axis_scale,color=colors[i], setylim = i==0)
            if i==1:
                pickable_data.fit_function = init_func
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')


class pickable_visu_mpe(visualization.CameraDisplay):

    def __init__(self,pickable_datas,extra_plot,figure,slice_func,level,show_fit,*args, **kwargs):
        super(pickable_visu_mpe, self).__init__(*args, **kwargs)
        self.pickable_datas = pickable_datas
        self.extra_plot = extra_plot
        self.figure=figure
        self.slice_func = slice_func
        self.level=level
        self.show_fit = show_fit

    def on_pixel_clicked(self, pix_id):
        legend_handles = []
        self.pix_id = pix_id
        self.extra_plot.cla()
        axprev = plt.axes([0.7, 0.8, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.8, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.on_next_clicked)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(self.on_prev_clicked)
        for i,pickable_data in enumerate(self.pickable_datas):
            col = 'k' if i==0 else 'b'

            slice = self.slice_func(pickable_data.data[self.level,self.pix_id])
            pickable_data.show(which_hist=(self.level,self.pix_id,), axis=self.extra_plot, show_fit=self.show_fit, slice=slice)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')

    def on_next_clicked(self,event):
        self.level += 1
        self.extra_plot.cla()
        for i, pickable_data in enumerate(self.pickable_datas):
            slice = self.slice_func(pickable_data.data[self.level, self.pix_id])
            pickable_data.show(which_hist=(self.level, self.pix_id,), axis=self.extra_plot, show_fit=self.show_fit,
                               slice=slice)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')

    def on_prev_clicked(self,event):
        self.level -=1
        if self.level < 0 :
            self.level = 0
            return
        print('level',self.level)
        self.extra_plot.cla()
        for i,pickable_data in enumerate(self.pickable_datas):
            slice = self.slice_func(pickable_data.data[self.level, self.pix_id])
            pickable_data.show(which_hist=(self.level, self.pix_id,), axis=self.extra_plot, show_fit=self.show_fit, slice=slice)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')


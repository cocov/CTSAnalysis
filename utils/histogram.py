import numpy as np
from numpy.linalg import inv
from scipy import optimize
from utils.fitting import gaussian_residual,gaussian
import matplotlib.pyplot as plt

class histogram :
    """
    A simple class to hold histograms data and manipulate them
    """
    def __init__(self, data = np.zeros(0),data_shape = (0,), bin_centers = np.zeros(0), bin_center_min=0 , bin_center_max=1 , bin_width=1 , xlabel='x',ylabel='y',label='hist'):
        ## TODO add constructor with binedge or with np.histo directly
        if bin_centers.shape[0] == 0:
            self.bin_width = bin_width
            # generate the bin edge array
            self.bin_edges = np.arange(bin_center_min - self.bin_width / 2, bin_center_max + self.bin_width / 2 + 1, self.bin_width)
            # generate the bin center array
            self.bin_centers = np.arange(bin_center_min, bin_center_max + 1, self.bin_width)
        else:
            self.bin_centers = bin_centers
            # generate the bin edge array
            self.bin_width = self.bin_centers[1]-self.bin_centers[0]
            self.bin_edges = np.arange(self.bin_centers[0] - self.bin_width / 2,
                                       self.bin_centers[self.bin_centers.shape[0]-1] + self.bin_width / 2 + 1, self.bin_width)

        # generate empty data
        if data.shape[0] == 0:
            self.data = np.zeros(data_shape+(self.bin_centers.shape[0],))
            self.errors = np.zeros(data_shape+(self.bin_centers.shape[0],))
        else :
            self.data = data
            self._compute_errors()

        self.fit_result = None
        self.fit_function = None
        self.fit_axis = None
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label

    def fill_with_batch(self,batch,indices=None):
        """
        A function to transform a batch of data in histogram and add it to the existing one
        :param batch: a np.array with the n-1 same shape of data, and n dimension containing the arry to histogram
        :return:
        """
        data = None
        if not indices:
            data = self.data
        else:
            data = self.data[indices]

        hist = lambda x : np.histogram( x , bins = self.bin_edges , density = False)[0]
        if batch.dtype != 'object':
            # Get the new histogram
            new_hist = np.apply_along_axis(hist, len(data.shape)-1, batch)
            # Add it to the existing
            data = np.add(data,new_hist)

        else :
            for indices in np.ndindex(batch.shape):
                # Get the new histogram
                new_hist = hist(batch[indices])
                # Add it to the existing
                data[indices] = np.add(data[indices],new_hist)

        if not indices:
            self.data = data
        else:
            self.data[indices] = data
        self._compute_errors()

    def _axis_fit( self, idx, func, p0  , slice=None , bounds=None):
        fit_result = None
        print(p0, slice, self.data[idx][slice[0]:slice[1]:slice[2]])
        if self.data[idx][slice[0]:slice[1]:slice[2]].shape == 0:
            fit_result = (np.ones((len(p0), 2)) * np.nan).reshape((1,) + (len(p0), 2))
        else:
            if not slice: slice = [0, self.bin_centers.shape[0] - 1, 1]
            try:
                ## TODO add the chi2 to the fitresult

                residual = lambda p, x, y, y_err: self._residual(func, p, x, y, y_err)
                out = optimize.least_squares(residual, p0, args=(
                    self.bin_centers[slice[0]:slice[1]:slice[2]], self.data[idx][slice[0]:slice[1]:slice[2]],
                    self.errors[idx][slice[0]:slice[1]:slice[2]]), bounds=bounds)
                val = out.x

                try:
                    cov = np.sqrt(np.diag(inv(np.dot(out.jac.T, out.jac))))
                    fit_result = np.append(val.reshape(val.shape + (1,)), cov.reshape(cov.shape + (1,)), axis=1)
                except np.linalg.linalg.LinAlgError as inst:
                    print(inst)
                    fit_result = np.append(val.reshape(val.shape + (1,)), np.ones((len(p0), 1)) * np.nan, axis=1)
                    fit_result = fit_result.reshape((1,) + fit_result.shape)
            except Exception as inst:
                print(inst)
                fit_result = (np.ones((len(p0), 2)) * np.nan).reshape((1,) + (len(p0), 2))
        return fit_result

    def fit(self,func, p0_func, slice_func, bound_func, config = None , limited_indices = None):
        """
        An helper to fit histogram
        :param func:
        :return:
        """
        data_shape = list(self.data.shape)
        data_shape.pop()
        data_shape = tuple(data_shape)
        self.fit_function = func
        self.fit_result = None
        # perform the fit of the 1D array in the last dimension
        count = 0
        indices_list = np.ndindex(data_shape)
        if limited_indices:
            indices_list = limited_indices
        for indices in indices_list:
            if type(self.fit_result).__name__ != 'ndarray':
                self.fit_result = np.ones(data_shape+(len(p0_func(self.data[indices],self.bin_centers,config=None)),2))*np.nan
            print("Fit Progress {:2.1%}".format(count/np.prod(data_shape)), end="\r")
            count+=1
            fit_res = None
            if type(config).__name__!='ndarray':
                fit_res = self._axis_fit( indices , func , p0_func(self.data[indices],self.bin_centers,config=None),
                                      slice=slice_func(self.data[indices[0]],self.bin_centers,config=None),
                                      bounds = bound_func(self.data[indices[0]],self.bin_centers,config=None))

            else:
                fit_res = self._axis_fit(indices, func,
                                         p0_func(self.data[indices], self.bin_centers, config=config[indices[0]]),
                                         slice=slice_func(self.data[indices[0]], self.bin_centers,
                                                          config=config[indices[0]]),
                                         bounds=bound_func(self.data[indices[0]], self.bin_centers,
                                                           config=config[indices[0]]))
            self.fit_result[indices]=fit_res



    '''
    def find_bin(self,x):
        #(x-self.bin_edge[0])/self.bin_width
        return (np.abs(self.bin_centers-x)).argmin()
    '''
    def find_bin(self,x):
        #TODO test that it gives good value
        return (x-self.bin_edges[0])//self.bin_width


    def fill(self,value,indices=None):
        '''
        Update the histogram array with an array of values
        :param value:
        :param indices:
        :return: void
        '''
        # change the value array to an array of histogram index to be modified
        hist_indicies = ((value - self.bin_edges[0]) // self.bin_width).astype(int)
        # treat overflow and underflow
        hist_indicies[hist_indicies>self.data.shape[-1]-1]=self.data.shape[-1]-1
        hist_indicies[hist_indicies<0]=0
        # get the corresponding indices multiplet
        dim_indices   = tuple( [np.indices(value.shape)[i].reshape(np.prod(value.shape)) for i in range(np.indices(value.shape).shape[0])], )
        dim_indices  += (hist_indicies.reshape(np.prod(value.shape)),)
        if value[...,0].shape == self.data[...,0].shape or not indices:
            self.data[dim_indices]+=1
        else:
            self.data[indices][dim_indices]+=1


    def fill_with_batch2(self,batch,indices=None):
        """
        A function to transform a batch of data in histogram and add it to the existing one
        :param batch: a np.array with the n-1 same shape of data, and n dimension containing the arry to histogram
        :return:
        """
        if not indices:
            data = self.data
        else:
            data = self.data[indices]

        hist = lambda x : np.histogram( x , bins = self.bin_edges , density = False)[0]
        if batch.dtype != 'object':
            # Get the new histogram
            new_hist = np.apply_along_axis(hist, len(data.shape)-1, batch)
            # Add it to the existing
            data = np.add(data,new_hist)
        else :
            for indices in np.ndindex(batch.shape):
                # Get the new histogram
                new_hist = hist(batch[indices])
                # Add it to the existing
                data[indices] = np.add(data[indices],new_hist)
        self._compute_errors()


    def _compute_errors(self):
        self.errors = np.sqrt(self.data)
        self.errors[self.errors==0.]=1.


    def predef_fit(self,type = 'Gauss' ,x_range=None, initials = None,bounds = None, config= None, slice_func=None):
        if type == 'Gauss':
            p0_func = None
            if not initials:
                p0_func = lambda x , xrange , config : [np.sum(x), xrange[np.argmax(x)], np.std(x)] #TODO modify
            else :
                p0_func = lambda x,  xrange , config: initials
            if not slice_func:
                if not x_range:
                    x_range = [self.bin_edges[0], self.bin_edges[-1]]
                    slice_func = lambda x,xrange , config: [0, self.bin_centers.shape[0], 1]
                else:
                    slice_func = lambda x, xrange, config: [self.find_bin(x_range[0]), self.find_bin(x_range[1]), 1]
            bound_func = None
            if not bounds:
                bound_func = lambda x , xrange , config: ([0.,-np.inf,1e-9],[np.inf,np.inf,np.inf])
            else:
                bound_func = lambda x , xrange , config : bounds

            data_shape = list(self.data.shape)
            data_shape.pop()
            data_shape = tuple(data_shape)
            config_array = None
            if not config:
                print('assign config array')
                config_array = np.zeros(data_shape)
            else :
                config_array = config

            self.fit(gaussian, p0_func=p0_func, slice_func=slice_func, bound_func=bound_func, config=config_array)
            # TODO self.fit_text

    def _residual(self,function, p , x , y , y_err):
        return (y - function(p, x)) / y_err

    def show(self, which_hist= None , axis=None ,show_fit=False, slice = None):

        if not which_hist:
            which_hist=(0,)*len(self.data[...,0].shape)

        if not slice:
            slice=[0,self.bin_centers.shape[0],1]
        x_text = np.min(self.bin_centers[slice[0]:slice[1]:slice[2]])
        y_text = 0.8 *(np.max(self.data[which_hist][slice[0]:slice[1]:slice[2]])+ self.errors[which_hist + (np.argmax(self.data[which_hist][slice[0]:slice[1]:slice[2]]),)])

        text_fit_result = ''
        precision = int(3)

        if show_fit:

            print(self.fit_result.shape)
            print(self.fit_result.shape[-2])
            print(which_hist)
            print (self.fit_result[which_hist][:,0])
            for i in range(self.fit_result[which_hist].shape[-2]):
                print()
                text_fit_result += 'p' +str(i) +  ' : ' + str(np.round(self.fit_result[which_hist+(i,0,)],precision))
                text_fit_result += ' $\pm$ ' + str(np.round(self.fit_result[which_hist+(i,1,)],precision))
                text_fit_result += '\n'

        ax=axis
        if not axis :
            plt.figure()
            ax=plt
        #plt.step(self.bin_centers, self.data[which_hist], where='mid', label='hist')
        ax.errorbar(self.bin_centers[slice[0]:slice[1]:slice[2]], self.data[which_hist][slice[0]:slice[1]:slice[2]], yerr=self.errors[which_hist][slice[0]:slice[1]:slice[2]],
                    fmt = 'o',label='MPE, level %d'%(which_hist[0]))
        if show_fit:
            ax.plot(self.bin_centers[slice[0]:slice[1]:slice[2]], self.fit_function(self.fit_result[which_hist][:,0], self.bin_centers[slice[0]:slice[1]:slice[2]]), label='fit')
            ax.text(x_text, y_text, text_fit_result)
        if not axis :
            ax.xlabel(self.xlabel)
            ax.ylabel(self.ylabel)
        else :

            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)
            ax.xaxis.get_label().set_ha('right')
            ax.xaxis.get_label().set_position((1, 0))
            ax.yaxis.get_label().set_ha('right')
            ax.yaxis.get_label().set_position((0, 1))
        #ax.xaxis.get_label().set_ha('right')
        #ax.xaxis.get_label().set_position((1,0))
        #ax.yaxis.get_label().set_ha('right')
        #ax.yaxis.get_label().set_position((0,1))
        if not axis :ax.ylim(0, (np.max(self.data[which_hist][slice[0]:slice[1]:slice[2]])+ self.errors[which_hist + (np.argmax(self.data[which_hist][slice[0]:slice[1]:slice[2]]),)])*1.3)
        else : ax.set_ylim(0, (np.max(self.data[which_hist][slice[0]:slice[1]:slice[2]])+ self.errors[which_hist + (np.argmax(self.data[which_hist][slice[0]:slice[1]:slice[2]]),)])*1.3)
        ax.legend(loc='best')


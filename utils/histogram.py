import numpy as np
from numpy.linalg import inv
from scipy import optimize
from utils.fitting import gaussian_residual,gaussian
import matplotlib.pyplot as plt

class histogram :
    """
    A simple class to hold histograms data and manipulate them
    """
    def __init__(self, data = np.zeros(0),data_shape = (0,), bin_centers = np.zeros(0), bin_center_min=0 , bin_center_max=1 , bin_width=1 ):
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
            self.data = np.zeros((data_shape,self.bin_centers.shape[0]))
            self.errors = np.zeros((data_shape,self.bin_centers.shape[0]))
        else :
            self.data = data
            self._compute_errors()

        self.fit_result = None
        self.fit_function = None
        self.fit_axis = None

    def fill_with_batch(self,batch):
        """
        A function to transform a batch of data in histogram and add it to the existing one
        :param batch: a np.array with the n-1 same shape of data, and n dimension containing the arry to histogram
        :return:
        """
        hist = lambda x : np.histogram( x , bins = self.bin_edges , density = False)[0]
        if batch.dtype != 'object':
            # Get the new histogram
            new_hist = np.apply_along_axis(hist, len(self.data.shape)-1, batch)
            # Add it to the existing
            self.data = np.add(self.data,new_hist)
        else :
            for indices in np.ndindex(batch.shape):
                # Get the new histogram
                new_hist = hist(batch[indices])
                # Add it to the existing
                self.data[indices] = np.add(self.data[indices],new_hist)
        self._compute_errors()

    def _axis_fit( self, idx, func, p0  , slice=None , bounds=None):
        if self.data[idx][slice[0]:slice[1]:slice[2]].shape == 0:
            return (np.ones((len(p0), 2)) * np.nan).reshape((1,) + (len(p0), 2))
        if not slice: slice = [0, self.bin_centers.shape[0] - 1, 1]
        try:
            ## TODO add the chi2 to the fitresult
            out = optimize.least_squares(func, p0, args=(
                self.bin_centers[slice[0]:slice[1]:slice[2]], self.data[idx][slice[0]:slice[1]:slice[2]]),
                                         bounds=bounds)
            val = out.x

            try:
                cov = np.sqrt(np.diag(inv(np.dot(out.jac.T, out.jac))))
            except np.linalg.linalg.LinAlgError:
                fit_rest = np.append(val.reshape(val.shape + (1,)), np.ones((len(p0), 1)) * np.nan, axis=1)
                fit_rest = fit_rest.reshape((1,) + fit_rest.shape)
                return fit_rest
            fit_res = np.append(val.reshape(val.shape + (1,)), cov.reshape(cov.shape + (1,)), axis=1)
            return fit_res.reshape((1,) + fit_res.shape)
        except :
            return (np.ones((len(p0),2)) * np.nan).reshape((1,)+(len(p0),2))


    def fit(self,func, p0_func, slice_func, bound_func, config = None):
        """
        An helper to fit histogram
        :param func:
        :return:
        """
        data_shape = list(self.data.shape)
        data_shape.pop()
        data_shape = tuple(data_shape)
        fit_results = None
        # perform the fit of the 1D array in the last dimension
        for indices in np.ndindex(data_shape):
            fit_res = self._axis_fit( indices , func, p0_func(self.data[indices],config=config[indices[0]]),
                                      slice=slice_func(self.data[indices[0]],config=config[indices[0]]),
                                      bounds = bound_func(self.data[indices[0]],self.bin_centers,config=config[indices[0]]))
            if type(fit_results).__name__!='ndarray':
                fit_results = fit_res
            else:
                fit_results = np.append(fit_results,fit_res,axis=0)
        return fit_results

    def find_bin(self,x):
        return (np.abs(self.bin_centers-x)).argmin()

    def _compute_errors(self):
        self.errors = np.sqrt(self.data)
        self.errors[self.errors==0.]=1.


    def predef_fit(self,type = 'Gauss' ,x_range=None, initials = None,bounds = None, config= None):

        if type == 'Gauss':
            p0_func = None
            if not initials:
                p0_func = lambda x , *args, **kwargs : [np.sum(x), np.mean(x), np.std(x)]
            else :
                p0_func = lambda x, *args, **kwargs: initials
            slice_func = None
            if not x_range:
                x_range=[self.bin_edges[0],self.bin_edges[-1]]
                slice_func = lambda x , *args, **kwargs : [0, self.bin_centers.shape[0], 1]
            else:
                slice_func = lambda x , *args, **kwargs : [self.find_bin(x_range[0]), self.find_bin(x_range[1]), 1]
            bound_func = None
            if not bounds:
                bound_func = lambda x , *args, **kwargs : ([0.,-np.inf,1e-9],[np.inf,np.inf,np.inf])
            else:
                bound_func = lambda x , *args, **kwargs : bounds

            data_shape = list(self.data.shape)
            data_shape.pop()
            data_shape = tuple(data_shape)

            config_array = None
            if not config:
                config_array = np.zeros(data_shape)
            else :
                config_array = config

            self.fit_result = self.fit(gaussian_residual, p0_func=p0_func, slice_func=slice_func, bound_func=bound_func, config=config_array)
            self.fit_function = gaussian
            self.fit_axis = np.linspace(x_range[0],x_range[1],(x_range[1]-x_range[0])/self.bin_width*10.)
            # TODO self.fit_text

    def _residual(self,function, p , x , y , y_err):
        return (y - function(p, x)) / y_err

    def show(self, which_hist=0 ,show_fit=False):

        print(self.bin_centers.shape)
        print(self.data.shape)

        x_text = np.min(self.bin_centers[which_hist])
        y_text = 0.8*(np.max(self.data[which_hist])+ self.errors[which_hist, np.argmax(self.data[which_hist])])

        text_fit_result = ''
        precision = int(3)

        if show_fit:

            for i in range(self.fit_result[which_hist].shape[0]):
                text_fit_result += 'p' +str(i) +  ' : ' + str(round(self.fit_result[which_hist,i,0],precision))
                text_fit_result += ' $\pm$ ' + str(round(self.fit_result[which_hist,i,1],precision))
                text_fit_result += '\n'


        plt.figure()
        plt.step(self.bin_centers, self.data[which_hist], where='mid', label='hist')
        plt.errorbar(self.bin_centers, self.data[which_hist], yerr=self.errors[which_hist])
        if show_fit:
            plt.plot(self.bin_centers, self.fit_function(self.fit_result[which_hist,:,0], self.bin_centers), label='fit')
            plt.text(x_text, y_text, text_fit_result, withdash=True)
        plt.xlabel('bin')
        plt.ylabel('count')
        plt.ylim((0, np.max(self.data[which_hist])+ self.errors[which_hist, np.argmax(self.data[which_hist])]))
        plt.legend(loc='best')
import numpy as np
from numpy.linalg import inv
from scipy import optimize

class histogram :
    """
    A simple class to hold histograms data and manipulate them
    """

    def __init__(self, data = np.zeros(0),data_shape = (0,), bin_centers = np.zeros(0), bin_center_min=0 , bin_center_max=1 , bin_width=1 ):
        if bin_centers.shape[0] == 0:
            # generate the bin edge array
            self.bin_edges = np.arange(bin_center_min - bin_width / 2, bin_center_max + bin_width / 2 + 1, bin_width)
            # generate the bin center array
            self.bin_centers = np.arange(bin_center_min, bin_center_max + 1, bin_width)
        else:
            self.bin_centers = bin_centers
            # generate the bin edge array
            bin_width = self.bin_centers[1]-self.bin_centers[0]
            self.bin_edges = np.arange(self.bin_centers[0] - bin_width / 2,
                                       self.bin_centers[self.bin_centers.shape[0]-1] + bin_width / 2 + 1, bin_width)

        # generate empty data
        if data.shape[0] == 0:
            self.data = np.zeros((data_shape,self.bin_centers.shape[0]))
        else :
            self.data = data

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

    def _axis_fit( self, idx, func, p0  , slice=None , bounds=None):
        try:
            if not slice: slice = [0, self.bin_centers.shape[0] - 1, 1]
            out = optimize.least_squares(func, p0, args=(
                self.bin_centers[slice[0]:slice[1]:slice[2]], self.data[idx][slice[0]:slice[1]:slice[2]]),
                                         bounds=bounds)
            val = out.x
            try:
                cov = np.sqrt(np.diag(inv(np.dot(out.jac.T, out.jac))))
            except np.linalg.linalg.LinAlgError as inst:
                fit_rest = np.append(val.reshape(val.shape + (1,)), np.ones((len(p0), 1)) * np.nan, axis=1)
                fit_rest = fit_rest.reshape((1,) + fit_rest.shape)
                return fit_rest
            fit_res = np.append(val.reshape(val.shape + (1,)), cov.reshape(cov.shape + (1,)), axis=1)
            return fit_res.reshape((1,) + fit_res.shape)
        except:
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
                                      slice=slice_func(self.data[indices],config=config[indices[0]]),
                                      bounds = bound_func(self.data[indices],self.bin_centers,config=config[indices[0]]))
            if type(fit_results).__name__!='ndarray':
                fit_results = fit_res
            else:
                fit_results = np.append(fit_results,fit_res,axis=0)
        return fit_results
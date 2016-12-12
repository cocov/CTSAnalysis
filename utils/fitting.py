import numpy as np
from scipy import optimize
from utils.peakdetect import peakdetect

def get_poisson_err(x):
    """
    Compute poisson error
    :param x: bin value
    :return: error
    """
    return np.sqrt(float(x)) if x > 0 else  1.

def errfunc_hist(fit_func, p, x, y ):
    """
    Compute residual taking into account poisson_err

    :param fit_func: the estimator
    :param p: the parameters of the estimator
    :param x: x
    :param y: y
    :return: the residual
    """
    y_err=np.sqrt(y)
    y_err[y_err == 0] = 1
    return (y - fit_func(p, x)) / y_err

def gaussian( p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    return p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[1]) ** 2 / (2. * p[2] ** 2))

def gaussian_residual( p, x, y ):
    """
    The redisidual for the gaussian
    :param p:
    :param x:
    :param y:
    :return:
    """
    return errfunc_hist(gaussian, p, x, y)



def multi_gaussian( p, x):
    """
    Multiple gaussian fit for the SPE
    :param p:
    :param x:
    :return:
    """
    """
    gaus1 = p[0] / (p[1]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[3]) ** 2 / (2. * (p[1]**2)))
    gaus2 = p[4] / (np.sqrt(p[1]**2+(2*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]+p[3])) ** 2 / (2. * (p[1]**2+(2*p[2])**2)))
    gaus3 = p[5] / (np.sqrt(p[1]**2+(3*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*2+p[3])) ** 2 / (2. * (p[1]**2+(3*p[2])**2)))
    gaus4 = p[6] / (np.sqrt(p[1]**2+(4*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*3+p[3])) ** 2 / (2. * (p[1]**2+(4*p[2])**2)))
    """
    """
    gaus1 = p[0] / (p[1]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[8]) ** 2 / (2. * (p[1]**2)))
    gaus2 = p[4] / (np.sqrt(p[1]**2+(2*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*1+p[8])) ** 2 / (2. * (p[1]**2+(2*p[2])**2)))
    gaus3 = p[5] / (np.sqrt(p[1]**2+(3*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*2+p[8])) ** 2 / (2. * (p[1]**2+(3*p[2])**2)))
    gaus4 = p[6] / (np.sqrt(p[1]**2+(4*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*3+p[8])) ** 2 / (2. * (p[1]**2+(4*p[2])**2)))
    gaus5 = p[7] / (np.sqrt(p[1]**2+(4*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*4+p[8])) ** 2 / (2. * (p[1]**2+(5*p[2])**2)))
    """
    # gaussian at 1
    gaus1 = p[0] / (p[1]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[8]) ** 2 / (2. * (p[1]**2)))
    gaus2 = p[4] / (np.sqrt(p[1]**2+2*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*1+p[8])) ** 2 / (2. * (p[1]**2+2*(p[2])**2)))
    gaus3 = p[5] / (np.sqrt(p[1]**2+3*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*2+p[8])) ** 2 / (2. * (p[1]**2+3*(p[2])**2)))
    gaus4 = p[6] / (np.sqrt(p[1]**2+4*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*3+p[8])) ** 2 / (2. * (p[1]**2+4*(p[2])**2)))
    gaus5 = p[7] / (np.sqrt(p[1]**2+5*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*4+p[8])) ** 2 / (2. * (p[1]**2+5*(p[2])**2)))

    return gaus1+gaus2+gaus3+gaus4+gaus5



def multi_gaussian_with0( p, x):
    """
    Multiple gaussian fit for the SPE
    :param p:
    :param x:
    :return:
    """
    gaus0 = p[0] / (p[9]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[7])) ** 2 / (2. * (p[9]**2)))
    gaus1 = p[4] / (np.sqrt(p[1]**2+1*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]+p[7]+p[8])) ** 2 / (2. * (p[1]**2+1*(p[2])**2)))
    gaus2 = p[5] / (np.sqrt(p[1]**2+2*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*2+p[7]+p[8])) ** 2 / (2. * (p[1]**2+2*(p[2])**2)))
    gaus3 = p[6] / (np.sqrt(p[1]**2+3*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*3+p[7]+[p[8]])) ** 2 / (2. * (p[1]**2+3*(p[2])**2)))
    gaus4 = p[10] / (np.sqrt(p[1]**2+4*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*4+p[7]+[p[8]])) ** 2 / (2. * (p[1]**2+4*(p[2])**2)))
    gaus5 = p[11] / (np.sqrt(p[1]**2+5*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*5+p[7]+[p[8]])) ** 2 / (2. * (p[1]**2+5*(p[2])**2)))

    return gaus0+gaus1+gaus2+gaus3+gaus4+gaus5



def multi_gaussian_residual( p, x, y):
    return errfunc_hist(multi_gaussian, p, x, y)


def multi_gaussian_residual_with0( p, x, y):
    if p[11]>p[10] or p[10]>p[6] or p[6]>p[5] or p[5]>p[4] or p[4]>p[0]: return 1.e8
    #if p[1]>p[1]: return 1.e8
    return errfunc_hist(multi_gaussian_with0, p, x, y)


# some usefull functions
def fit_baseline(data):
    """
    baseline fit
    :param data: pixel adcs
    :return: [sigma,mu]
    """
    if np.isnan(data[0]): return [0.,0.]
    if np.std(data)>200: return [0.,0.]
    mode = scipy.stats.mode(data).mode[0]
    hist , bins = np.histogram(data,bins=np.arange(mode-10.5, mode+2.5, 1),density=False)
    out = optimize.least_squares(gaussian_residual, [data.shape[0], mode, 0.8], args=(np.arange(mode-10., mode+2., 1), hist),
                                 bounds=([0., mode-10., 1.e-6], [1.e8, mode+1., 1.e5]))
    return [out.x[1],out.x[2]]

# some usefull functions
def fit_baseline_hist(data):
    """
    baseline fit
    :param data: pixel adcs
    :return: [sigma,mu]
    """
    if np.isnan(data[0]): return [0.,0.]
    if np.std(data)>200: return [0.,0.]
    mode = scipy.stats.mode(data).mode[0]
    hist , bins = np.histogram(data,bins=np.arange(mode-10.5, mode+2.5, 1),density=False)
    out = optimize.least_squares(gaussian_residual, [data.shape[0], mode, 0.8], args=(np.arange(mode-10., mode+2., 1), hist),
                                 bounds=([0., mode-10., 1.e-6], [1.e8, mode+1., 1.e5]))
    return [out.x[1],out.x[2]]

# some usefull functions

def fit_multigaussian(data):
    """
    baseline fit
    :param data: pixel adcs
    :return: [sigma,mu]
    """
    hist , bins = np.histogram(data,bins=np.arange(5.5, 41.5, 1),density=False)
    out = optimize.least_squares(multi_gaussian_residual, [1000.,0.9,0.9,5.6,100.,10.,1.,5.6], args=(np.arange(6,41, 1), hist),
                                 bounds=([1000.,1e-7,1e-7,4.,0.,0.,0.,4.],
                                         [np.inf, 10., 10, 7., np.inf, np.inf, np.inf,8.]))
    return out


def cleaning_peaks_correct( data, baseline, threshold=2., l=1):
    peaks = peakdetect(data, lookahead=l)[0]
    newpeaks = []
    for peak in peaks:
        if peak[0]<2 or peak[0]>47:continue
        if peak[1]<threshold + baseline: continue
        best_position = 0
        if max(data[peak[0]],data[peak[0]-1])==data[peak[0]]:
            best_position = peak[0]
        else :best_position = peak[0]-1
        if max(data[best_position],data[peak[0]+1])==data[peak[0]+1]:
            best_position = peak[0]+1
        newpeaks.append(data[best_position]-baseline)
    return newpeaks

def cleaning_peaks( datain ,baseline,sigma):
    data = datain
    data=data-baseline
    data[data<2*sigma]=0
    peaks = peakdetect(data, lookahead=2)[0]
    new_peaks = []
    for pp in peaks:
        p = pp[0]
        if p < 3 or p > 47: continue
        maxval = 0.
        for i in range(3):
            val = np.sum(data[p - i:p - i + 3:1])
            if val > maxval: maxval = val
        if maxval < 3 * (2.5 * sigma): continue
        new_peaks +=[data[p - 1 + np.argmax(data[p - 1:p + 1:1])]+baseline] #[[p - 1 + np.argmax(data[p - 1:p + 1:1]), data[p - 1 + np.argmax(data[p - 1:p + 1:1])]]]
    return new_peaks

def spe_peaks_in_event_list( data, baseline,sigma ):
    peaks = []
    for pix in np.ndindex(data.shape[0]):
        print("Progress {:2.1%}".format(pix[0] / data.shape[0]), end="\r")
        peaks.append([])
        for evt in np.ndindex(data.shape[1]):
            peaks[-1]+=cleaning_peaks(data[pix][evt],baseline[pix], sigma[pix])
    return np.array(peaks,dtype=object)


import numpy as np

def get_poisson_err(x):
    return np.sqrt(float(x)) if x > 0 else  1.

def errfunc_hist(fit_func, p, x, y ):
    y_err = np.vectorize(get_poisson_err)( y )
    return (y - fit_func(p, x)) / y_err

def gaussian( p, x):
    return p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[1]) ** 2 / (2. * p[2] ** 2))

def gaussian_residual( p, x, y):
    return errfunc_hist(gaussian, p, x, y)



def multi_gaussian( p, x):
    norm1=p[0]
    sigma_e = p[1]
    sigma_1 = p[2]
    g = p[3]
    norm2,norm3,norm4 = p[4],p[5],p[6]
    mu1 = p[7]

    gaus1 = p[0] / (p[1]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[7]) ** 2 / (2. * (p[1]**2)))
    gaus2 = p[4] / (np.sqrt(p[1]**2+(2*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]+p[7])) ** 2 / (2. * (p[1]**2+(2*p[2])**2)))
    gaus3 = p[5] / (np.sqrt(p[1]**2+(3*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*2+p[7])) ** 2 / (2. * (p[1]**2+(3*p[2])**2)))
    gaus4 = p[6] / (np.sqrt(p[1]**2+(4*p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*3+p[7])) ** 2 / (2. * (p[1]**2+(4*p[2])**2)))
    return gaus1+gaus2+gaus3+gaus4



def multi_gaussian_residual( p, x, y):
    return errfunc_hist(multi_gaussian, p, x, y)

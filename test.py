import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import optimize

def poissonErr(x):
    return np.sqrt(float(x)) if x > 0 else  1.

def errfunc_hist(fit_func, p, x, y ):
    y_err = np.vectorize(poissonErr)(y)
    return (y - fit_func(p, x)) / y_err

gaussian = lambda p, x: p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[1]) ** 2 / (2. * p[2] ** 2))
gauss_residual = lambda p, x, y: errfunc_hist(gaussian, p, x, y)

plt.ion()
gaus = stats.norm
loc, scale = 10, 1
size = 20000
y = np.append(gaus.rvs( loc, scale, size=size),gaus.rvs( 15, 0.1, size=100000))
y = y[y<12]
x = np.linspace(0, 12, 20)

param = gaus.fit(y)
print (param)

pdf_fitted = gaus.pdf(x, *param)
plt.plot(np.linspace(0, 12, 20), pdf_fitted, color='r')

# plot the histogram
plt.hist(y, normed=True, bins=30)

plt.show()


hist , bins = np.histogram(y,bins=np.linspace(-0.5, 12.5, 21),density=True)
print(x.shape, hist.shape)
out = optimize.least_squares(gauss_residual, [10000.,8., 1.], args=(x,hist),
                             bounds=([0., 0.,1.e-6], [np.inf,100., 1.e5]))
print(out.x)
plt.plot(np.linspace(0, 12, 20),gaussian(out.x,x) , color='g')


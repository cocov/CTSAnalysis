import math
from scipy.special import factorial
import matplotlib.pyplot as plt
import numpy as np


def poisson(k, mu):

    return mu ** k * np.exp(-mu) / math.factorial(k)

def gaussian(x, sigma, mean, amplitude=1):

    return amplitude / np.sqrt(2 * sigma ** 2 * math.pi) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

def gaussian_sum(param, x):

    temp = np.zeros(x.shape)

    param = param.reshape(int(len(param)/3), 3)

    for i in range(param.shape[0]):

        temp += gaussian(x, param[i, 2], param[i, 1], amplitude=param[i, 0])

    return temp

def gaussian_sum_1gain(param, x):

    temp = np.zeros(x.shape)

    baseline = param[0]
    gain = param[1]
    sigma_e = param[2]
    sigma_1 = param[3]
    amplitudes = param[4:len(param)+1]

    n_peaks = len(amplitudes)

    for i in range(n_peaks):
        sigma = np.sqrt(sigma_e**2 + i*sigma_1**2)
        temp += gaussian(x, sigma, baseline + i*gain, amplitude=amplitudes[i])

    return temp

def generalized_poisson(k, mu, mu_xt, amplitude=1):
    if mu_xt < 0 or mu < 0 or k < 0:

        if isinstance(k, int):
            return 0
        else:
            return np.zeros(len(k))

    else:

        #return amplitude * mu * (mu + k * mu_xt) ** (k - 1) * np.exp(-mu - k * mu_xt) / factorial(k)
        return np.exp(np.log(amplitude) + np.log(mu) + np.log(mu + k * mu_xt)*(k - 1) + (-mu - k * mu_xt) - np.sum([np.log(i) for i in range(k)]))

def erlang_compound(x, mu, mu_xt):
    temp = 0
    mu_xt = (mu_xt)
    n = 15
    for k in range(n):

        if k == 0:
            temp += poisson(k, mu)
        else:
            temp += mu_xt ** k * x ** (k - 1) * np.exp(-mu_xt * x) / math.factorial(k - 1) * poisson(k, mu)

    return temp


def mpe_gaussian_distribution( p , x):
    p[0] = gain
    p[1] = sigma_e
    p[2] = sigma_1
    p[3] = offset
    amplitude = []
    for i in range(4,len(p)-4):
        amplitude.append(p[i])
    n_peak = len(amplitude)
    temp = np.zeros(len(x))
    x = x - offset
    amplitude = np.array(amplitude)
    x = x - offset
    for n in range(int(n_peak)):
        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2) * gain
        temp += amplitude[n] * gaussian(x, sigma_n, n * gain)
    return temp

def mpe_distribution_general(p, x, config=None):

    mu, mu_xt, gain, offset, sigma_e, sigma_1, amplitude = p

    #print(p)

    temp = np.zeros(x.shape)
    x = x - offset
    n_peak = 40
    for n in range(0, n_peak, 1):

        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2) * gain

        temp += generalized_poisson(n, mu, mu_xt) * gaussian(x, sigma_n, n * gain)

    return temp * amplitude


def mpe_distribution_general_sh(p, x, config=None):

    mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude , offset= p

    #print(p)

    temp = np.zeros(x.shape)
    x = x - baseline
    n_peak = 15
    for n in range(0, n_peak, 1):

        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2) #* gain

        temp += generalized_poisson(n, mu, mu_xt) * gaussian(x, sigma_n, n * gain + (offset if n!=0 else 0))

    return temp * amplitude



if __name__ == '__main__':
    x = np.arange(0, 200, 1)
    n_peak = 20
    gain = 5.6
    mu = 5.
    mu_xt = 0.08
    sigma_e = 0.09
    sigma_1 = 0.1
    type_list = ['generalized_poisson', 'poisson', 'erlang_compound']
    offset = 10.
    plt.figure()

    for type in type_list:
        y = mpe_distribution(x, n_peak, gain, mu, mu_xt, sigma_e, sigma_e, offset, type)

        plt.plot(x, y, label=type)

    plt.xlabel('ADC count [ADC]')
    plt.ylabel('P(ADC)')
    plt.legend(loc='best')
    plt.show()

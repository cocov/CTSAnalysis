import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

plt.ion()


def gaussian( p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    return p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-p[1]) ** 2 / (2. * p[2] ** 2))


filename_pulse_shape = 'pulse_SST-1M_AfterPreampLowGain.dat'
time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
amplitudes = amplitudes / min(amplitudes)
interpolated_pulseshape = scipy.interpolate.interp1d(time_steps, amplitudes, kind='cubic',
                                                          bounds_error=False, fill_value=0.)

'''
#time = np.arange(9.5-4*2,9.5+4*3,0.01)
time = np.arange(0,200,0.01)
print(time)
shape = interpolated_pulseshape(time)
shape[shape<1e-1]=-1.
plt.subplots(2,2)
plt.subplot(2,2,1)
plt.plot(time,shape)
plt.subplot(2,2,2)
shape1 = shape #np.extract(shape>0,shape)
print(shape1.shape,np.arange(0.0,1.02,0.01).shape)
shape_hist,bins = np.histogram(shape1,bins=np.arange(0.0,1.02,0.01),density=True)

shape_full = np.zeros(np.arange(-0.505,1.535,0.01).shape)
shape_full[int(0.5/0.01):int(0.5/0.01)+101:1]=shape_hist
print(bins.shape,np.arange(-0.005,1.005,0.01).shape,shape_hist.shape)
plt.plot(np.arange(-0.005,1.005,0.01),shape_hist)
plt.plot(np.arange(-0.505,1.535,0.01),shape_full)

time1 = np.arange(-0.505,0.515,0.01)
gauss = gaussian([1.,0.,0.7/5.6],time1)

plt.plot(time1,gauss)
'''


def create_pdf_DC( n=0 , gain=5.6, sigma_e = 0.7, th=0. , freq=1.e6):
    plt.subplots(1,3,figsize=(25,7))
    time = np.arange(0, 1e9/freq, 0.001)
    shape = interpolated_pulseshape(time)*gain
    shapeXT = interpolated_pulseshape(time)*gain*2
    shape[shape<th]=-2
    shapeXT[shape<th]=-2
    window = time[np.where(shape>th)[0][-1]]-time[np.where(shape>th)[0][0]]
    windowXT = time[np.where(shapeXT>th)[0][-1]]-time[np.where(shapeXT>th)[0][0]]
    proba1 = freq*window*1e-9
    proba2 = proba1**2
    proba3 = proba2*proba1
    proba4 = proba2**2
    print(window)
    plt.subplot(1,3,1)
    plt.plot(time,shape)
    #
    shape1 = shape
    binwidth = gain/100
    shape_full, bins = np.histogram(shape1, bins=np.arange(0., gain+3*binwidth, binwidth), density=True)
    time1 = np.arange(-binwidth/2, gain+1.5*binwidth, binwidth)
    print(time1.shape,shape_full.shape,)
    shape_full2 = np.convolve(shape_full, shape_full)
    time2= np.arange(2*time1[0], 2*time1[-1]+0.5*binwidth, binwidth)
    print(time2.shape,shape_full2.shape,)
    shape_full3 = np.convolve(shape_full2, shape_full)
    time3= np.arange(3*time1[0], 3*time1[-1]+0.5*binwidth, binwidth)
    shape_full4 = np.convolve(shape_full3, shape_full)
    time4= np.arange(4*time1[0], 4*time1[-1]+0.5*binwidth, binwidth)


    plt.subplot(1,3,2)
    plt.semilogy()

    plt.plot(time1, shape_full)
    plt.plot(time2, shape_full2)
    plt.plot(time3, shape_full3)
    plt.plot(time4, shape_full4)

    time0 = np.arange(-5*sigma_e-0.5*binwidth, 5*sigma_e+2.5*binwidth, binwidth)
    gauss = gaussian([(1-(proba1+proba2+proba3+proba4)), 0., sigma_e], time0)

    plt.plot(time0, gauss)
    conv = np.convolve(shape_full, gauss)
    conv = conv/np.sum(conv)*proba1
    conv1 = np.convolve(shape_full2, gauss)
    conv1 = conv1/np.sum(conv1)*proba2
    conv2 = np.convolve(shape_full3, gauss)
    conv2 = conv2/np.sum(conv2)*proba3
    conv3 = np.convolve(shape_full4, gauss)
    conv3 = conv3/np.sum(conv3)*proba4
    plt.subplot(1,3,3)
    plt.semilogy()

    t = np.arange(time0[0]-time1[0], time1[-1]+time0[-1]+binwidth, binwidth)
    t1 = np.arange(time0[0]-time2[0], time2[-1]+time0[-1]+2*binwidth, binwidth)
    t2 = np.arange(time0[0]-time3[0], time3[-1]+time0[-1]+3*binwidth, binwidth)
    t3 = np.arange(time0[0]-time4[0], time4[-1]+time0[-1]+4*binwidth, binwidth)
    print(t1.shape,conv1.shape,shape_full2.shape,gauss.shape)
    plt.plot(time0,gauss)
    plt.plot(t,conv)
    plt.plot(t1,conv1)
    plt.plot(t2,conv2)
    plt.plot(t3,conv3)
    tall=np.arange(time0[0]-time4[0],time4[-1]+time0[-1]+4*binwidth,binwidth)
    val = []
    for i,ti in enumerate(t3):
        val+=[0.]
        val[-1]+=gauss[i]
        if ti in t:
            val[-1]+=conv[i]
        if ti in t1:
            val[-1]+=conv1[i]
        if ti in t2:
            val[-1]+=conv2[i]
        if ti in t3:
            val[-1]+=conv3[i]

    plt.plot(t3, val)

    #plt.plot(conv2)
    #plt.plot(conv3)
    #plt.ylim(1e-6,1.)
    return conv


#plt.subplot(2,2,3)
conv = create_pdf_DC(th=0.)
#conv1 = create_pdf_DC(th=0.5)
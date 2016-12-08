from utils.histogram import histogram
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
h = histogram(bin_center_min=-10., bin_center_max=10., bin_width=1., data_shape=((3,10,)))
h1 = histogram(bin_center_min=-10., bin_center_max=10., bin_width=1., data_shape=((3,10,)))

data = np.arange(-10.,3*10).reshape(3,10)
data1 = np.arange(0.,10).reshape(10)


print(data)
print(h1.data.shape)

h1.fill(data1,indices=(0,))
h.fill(data)
h1._compute_errors()
h._compute_errors()

for i in range(h.data.shape[1]):
    h.show(which_hist=(0,i,))
    h.show(which_hist=(1,i,))
    h.show(which_hist=(2,i,))
    plt.show()
    input('bla')
#pri
    # nt(h.data)
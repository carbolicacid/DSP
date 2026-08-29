#1.Using function signal_fold() and signal_add(), write a code that will decompose any real signal into its even and odd counterparts. Plot the component.

import numpy as np
import matplotlib.pyplot as plt

#Sample signal x[n] for testing
n = np.arange(-2, 5)
x = np.array([1, 2, 3, 4, 3, 2, 1])

n0 = np.arange(-2, 5)
x0 = np.array([1, 2, 3, 4, 3, 2, 1])

def signal_add(x1, n1, x2, n2):
    
    min_n = min(np.min(n1), np.min(n2))
    max_n = max(np.max(n1), np.max(n2))
    n = np.arange(min_n, max_n + 1)
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))
    y1[np.where(np.isin(n,n1))] = x1
    y2[np.where(np.isin(n,n2))] = x2
    y = y1 + y2
    return y, n

def signal_fold(x, n):
    x1 = np.fliplr([x])[0]
    n1 = -np.fliplr([n])[0]
    return x1, n1

#decomposing into even odd signal

x_fold,n_fold = signal_fold(x, n)
xe, ne = signal_add(x, n, x_fold, n_fold)
xe = 0.5*xe

xo, no = signal_add(x, n, -x_fold, n_fold)
xo = 0.5*xo

y, m = signal_add(x, n, x0, n0)

plt.subplot(221)
plt.stem(n, x)
plt.xticks(n)
plt.yticks(x)
plt.title('Original Signal')
plt.grid()

plt.subplot(222)
plt.stem(ne, xe)
plt.xticks(ne)
plt.yticks(xe)
plt.title('Even Signal')
plt.grid()

plt.subplot(223)
plt.stem(no, xo)
plt.xticks(no)
plt.yticks(xo)
plt.title('Odd Signal')
plt.grid()

plt.subplot(224)
plt.stem(m, y)
plt.xticks(m)
plt.yticks(y)
plt.title('Added Signal')
plt.grid()
plt.tight_layout()





#2. If x[n] = {1, 1, 2, 3, ↑5, 8, 13, 21, 34, 55}, using the functions signal fold() and signal shift(), show i) x[−n − 3] and ii) x[−n + 4].

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,1,2,3,5,8,13,21,34,55])
n = np.arange(-4, 6)

def signal_fold(x, n):
    x1 = np.fliplr([x])[0]
    n1 = -np.fliplr([n])[0]
    return x1, n1

def signal_shift(x, n1, n0):
    n = n1 + n0
    y = x
    return y, n


#i) x[−n − 3]

a , b = signal_fold(x,n)
n0 = 3
c, d = signal_shift(a, b, -n0)
plt.subplot(121)
plt.stem(d, c)
plt.xticks(d)
plt.yticks(c)
plt.title('Signal: x[−n − 3]')
plt.grid()

#ii) x[−n + 4]   

e, f = signal_fold(x,n)
n1 = 4
g, h = signal_shift(e, f, n1)
plt.subplot(122)
plt.stem(h, g)
plt.xticks(h)
plt.yticks(g)
plt.title('Signal: x[−n + 4]')
plt.grid()
plt.tight_layout()





#3. Are time shifting and time reversal operations commutative? Verify this by using the Problem 2.

#Ans: No. Time shifting and time reversal aren't same as both the graphs aren't same in output.





#4. Verify the Associative Property of Convolution using convolution sum function.

import numpy as np
import matplotlib.pyplot as plt

def convulaton_sum(x, nx, h, nh, h1, nh1):
    kmin = np.min(nx) + np.min(nh) + np.min(nh1)
    kmax = np.max(nx) + np.max(nh) + np.max(nh1)
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(np.convolve(x, h), h1)
    return y, k


def convulaton_sum1(x, nx, h, nh):
    kmin = np.min(nx) + np.min(nh) 
    kmax = np.max(nx) + np.max(nh) 
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(x, h)
    return y, k

nx = np.arange(-5, 6)
x = 5 - np.abs(nx)
h = (nx==0)
nh = nx
h1 = (nx==0)
nh1 = nx


y1, n1 = convulaton_sum(x, nx, h, nh, h1, nh1)
plt.subplot(121)
plt.stem(n1, y1)
plt.title('y[n] = x[n]*h[n]*h1[n]')
plt.xticks(n1)
plt.yticks(y1)
plt.grid()


y0, n0 = convulaton_sum1(h, nh, x, nx)
y2, n2 = convulaton_sum1(y0, n0, h1, nh1)
plt.subplot(122)
plt.stem(n2, y2)
plt.title('y[n] = x[n]* (h[n]*h1[n])')
plt.xticks(n2)
plt.yticks(y2)
plt.grid()
plt.tight_layout()





#5. Consider an audio signal x[n] = cos 0.2πn + 0.5 cos 0.6πn. If the signal is reflected from a barrier and reaches the listeners, they will experience the signal as: y[n] = x[n] + 0.1x[n − 20]. Generate 200 samples of y[n] and show its autocorrelation. Can you find the delay between x[n] and y[n]?

import numpy as np
import matplotlib.pyplot as plt

def signal_fold(x, n):
    y = np.fliplr([x])[0]
    m = -np.fliplr([n])[0]
    return y, m

def signal_add(x1, n1, x2, n2):
    min_n = min(np.min(n1), np.min(n2))
    max_n = max(np.max(n1), np.max(n2))
    n = np.arange(min_n, max_n + 1)
    
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))
    
    y1[np.where(np.isin(n, n1))] = x1
    y2[np.where(np.isin(n, n2))] = x2
    
    y = y1 + y2
    return y, n

def signal_shift(x, n1, n0):
    n = n1 + n0
    y = x
    return y, n

def convolution_sum(x, nx, h, nh):
    kmin = np.min(nx) + np.min(nh)
    kmax = np.max(nx) + np.max(nh)
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(x, h)
    return y, k

n_samples = np.arange(0, 200)

x_audio = np.cos(0.2 * np.pi * n_samples) + 0.5 * np.cos(0.6 * np.pi * n_samples)

x_delayed, n_delayed = signal_shift(x_audio, n_samples, 20)

scaled_delayed = 0.1 * x_delayed
y_audio, ny_audio = signal_add(x_audio, n_samples, scaled_delayed, n_delayed)

N_y = len(y_audio)
ryy = np.correlate(y_audio, y_audio, mode='full')
lags = np.arange(-N_y + 1, N_y)

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(ny_audio[:200], y_audio[:200])

plt.subplot(2, 1, 2)
plt.plot(lags, ryy)

plt.tight_layout()
plt.show()





#6.  Given two signals, x[n] = {↑1, 2, 1, 1} and y[n] = {3, ↑5, 8, 13, 21}. Using these two signals show that cross-correlation is non-commutative. Show the corresponding cross-correlation output figures.

import numpy as np
import matplotlib.pyplot as plt
def signal_fold(x, n):
    y = np.fliplr([x])[0]
    m = -np.fliplr([n])[0]
    return y, m

def signal_add(x1, n1, x2, n2):
    min_n = min(np.min(n1), np.min(n2))
    max_n = max(np.max(n1), np.max(n2))
    n = np.arange(min_n, max_n + 1)
    
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))
    
    y1[np.where(np.isin(n, n1))] = x1
    y2[np.where(np.isin(n, n2))] = x2
    
    y = y1 + y2
    return y, n

def signal_shift(x, n1, n0):
    n = n1 + n0
    y = x
    return y, n

def convolution_sum(x, nx, h, nh):
    kmin = np.min(nx) + np.min(nh)
    kmax = np.max(nx) + np.max(nh)
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(x, h)
    return y, k

x6 = np.array([1, 2, 1, 1])
y6 = np.array([3, 5, 8, 13, 21])

r_xy = np.correlate(x6, y6, mode='full')
r_yx = np.correlate(y6, x6, mode='full')

lags_xy = np.arange(-len(y6) + 1, len(x6))
lags_yx = np.arange(-len(x6) + 1, len(y6))

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(lags_xy, r_xy)

plt.subplot(2, 1, 2)
plt.stem(lags_yx, r_yx)

plt.tight_layout()
plt.show()





#7. Solve all the exercise problems of the textbook from page 115 to page 117.
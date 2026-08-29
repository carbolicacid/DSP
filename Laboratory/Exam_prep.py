# 2.1

import numpy as np
import matplotlib.pyplot as plt

Fo = 100 # Signal Frequency in Hz
To = 1 / Fo # Signal Period in seconds
Fs = 800 # Sampling Frequency > Nyquist rate
Ts = 1 / Fs # Sampling Period in seconds
t = np.linspace(0, 3 * To, 1000)
xc = 10 * np.cos(2 * np.pi * Fo * t)
t1 = np.arange(0, 3 * To + Ts, Ts)
xs = 10 * np.cos(2 * np.pi * Fo * t1)

plt.subplot(211)
plt.plot(t, xc, color='b')
plt.stem(t1, xs, linefmt='r')

plt.subplot(212)
plt.stem(t1, xs, linefmt='r-')
plt.tight_layout()
plt.show()
  

#2.2                       Aliasing in Time Domain

import numpy as np
import matplotlib.pyplot as plt

Fo = 100 # Signal Frequency in Hz
To = 1 / Fo # Signal Period in seconds
Fs = 200 # Sampling Frequency = Nyquist rate
Ts = 1 / Fs # Sampling Period in seconds
t = np.linspace(0, 3 * To, 1000)
xc = 10 * np.cos(2 * np.pi * Fo * t)
t1 = np.arange(0, 3 * To + Ts, Ts)
xs = 10 * np.cos(2 * np.pi * Fo * t1)

plt.subplot(211)
plt.plot(t, xc, color='b')
plt.stem(t1, xs, linefmt='r')

plt.subplot(212)
plt.stem(t1, xs, linefmt='r-')
plt.grid()
plt.tight_layout()
plt.show()


#2.3

import numpy as np
import matplotlib.pyplot as plt

Fo = 100 # Original Signal Frequency in Hz
To = 1 / Fo # Original Signal Period in seconds
Fo1 = 50 # Aliased Signal Frequency in Hz
To1 = 1 / Fo1 # Aliased signal period in seconds
Fs = 150 # Sampling Frequency < Nyquist rate
Ts = 1 / Fs # Sampling Period in seconds

t = np.linspace(0, 3 * To + Ts, 1000)
xc = 10 * np.cos(2 * np.pi * Fo * t) # Continuous signal
t1 = np.arange(0, 3 * To + Ts, Ts)
xs = 10 * np.cos(2 * np.pi * Fo * t1)
t2 = np.linspace(0, 3 * To + Ts, 1000)
xc2 = 10 * np.cos(2 * np.pi * Fo1 * t2) # Aliased continuous signal

plt.plot(t, xc, 'b')
plt.plot(t2, xc2, 'g--')
plt.stem(t1, xs, linefmt='r', markerfmt='ro')
plt.grid()
plt.tight_layout()



#2.4                         Quantization
 


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

t = np.linspace(0, 1, 100)
x = signal.sawtooth(2 * np.pi * 5 * t) # Generate the input signal

b = [1, 2, 4, 8] # Number of bits
DR = np.max(x) - np.min(x) # Dynamic range

for i in range(len(b)):
    L = 2**b[i] # Quantization level
    q = DR / L # Quantization step size
    y = np.sign(x) * q * np.floor((abs(x) / q) + (1 / 2))
    
    plt.subplot(2, 2, i + 1)
    plt.plot(t, x, 'b', label='Original')
    plt.plot(t, y, 'r', label='Quantized')
    plt.grid()

plt.tight_layout()
plt.show()

#2.5

import numpy as np
import matplotlib.pyplot as plt

num_bits = [1, 2, 3, 4]
amplitude_max = 2
frequency = 2
sampling_rate = 500
duration = 1

t = np.linspace(0, duration, sampling_rate * duration)
analog_sig = amplitude_max * np.sin(2 * np.pi * frequency * t)

for i in range(0, len(num_bits), 1):
    num_levels = 2**num_bits[i]
    quantization_step = (2 * amplitude_max) / num_levels
    
    quantized_sig = np.round((analog_sig + amplitude_max) / quantization_step)
    quantized_sig_approx = (quantized_sig * quantization_step) - amplitude_max
    quantization_error = analog_sig - quantized_sig_approx
    
    plt.subplot(2, 2, i + 1)
    plt.plot(t, analog_sig, 'y', linewidth=4)
    plt.plot(t, quantized_sig_approx, 'b', linewidth=1)
    plt.plot(t, quantization_error, 'r', linewidth=1)
    plt.grid()

plt.tight_layout()
plt.show()


#2.6                     Signal Reconstruction


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
y = np.sinc(x)

plt.plot(x, y)

plt.grid()
plt.show()


#2.7

import numpy as np
import matplotlib.pyplot as plt

Fo = 100 # Signal frequency in Hz
To = 1 / Fo # Signal period
Fs = 800 # Sampling frequency in Hz
Ts = 1 / Fs # Sampling period

tc = np.arange(0, 3 * To, To / 100)
xc = 5 * np.cos(200 * np.pi * tc)

ts = np.arange(0, 3 * To + Ts, Ts)
xs = 5 * np.cos(200 * np.pi * ts)
N = len(ts) # Number of samples

xr = np.zeros(len(tc))
sinc_fun = np.zeros((N, len(tc)))

for t_idx in range(len(tc)):
    for n_idx in range(N):
        sinc_val = np.sinc((tc[t_idx] - n_idx * Ts) / Ts)
        sinc_fun[n_idx, t_idx] = sinc_val
        xr[t_idx] += xs[n_idx] * sinc_val

plt.figure()
for n_idx in range(N):
    plt.plot(tc, xs[n_idx] * sinc_fun[n_idx, :])
plt.stem(ts, xs, 'k', markerfmt='ko', basefmt=" ")
plt.grid()

plt.figure()
plt.plot(tc, xc, 'b-', linewidth=2,)
plt.stem(ts, xs, 'k', markerfmt='ko')
plt.plot(tc, xr, 'r--', linewidth=2)
plt.grid()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==============================================================================
# PROBLEM 1
# Continuous signal with frequencies 60 Hz, 50 Hz, and 75 Hz (Fo = 75 Hz).
# Sampled at Fs = 4*Fo, 2*Fo, and Fo for 3 cycles.
# ==============================================================================

Fo1 = 75
To1 = 1 / Fo1

t1 = np.linspace(0, 3 * To1, 1000)
xc1 = (10 * np.cos(120 * np.pi * t1) + 
       5 * np.sin(100 * np.pi * t1 + np.radians(300)) + 
       4 * np.sin(150 * np.pi * t1 + np.radians(45)))

sampling_rates1 = [4 * Fo1, 2 * Fo1, Fo1]

plt.figure(1)
for i, Fs in enumerate(sampling_rates1):
    Ts = 1 / Fs
    ts = np.arange(0, 3 * To1 , Ts)
    xs = (10 * np.cos(120 * np.pi * ts) + 
          5 * np.sin(100 * np.pi * ts + np.radians(300)) + 
          4 * np.sin(150 * np.pi * ts + np.radians(45)))
    
    plt.subplot(3, 1, i + 1)
    plt.plot(t1, xc1, 'b')
    plt.stem(ts, xs, 'r')
    plt.grid()

plt.tight_layout()


# ==============================================================================
# PROBLEM 2
# Signal x(t) converted to sequence x[n] for -10 <= n <= 10 using Fs = 500 Hz.
# Displays: i) original, ii) sampled, iii) discrete, iv) final output.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
Fo2 = 125
Fs2 = 500
Ts2 = 1 / Fs2

n2 = np.arange(-10, 11)
ts2 = n2 * Ts2
t2 = np.linspace(ts2[0], ts2[-1], 1000)

xc2 = 10 * np.cos(250 * np.pi * t2 + np.radians(60)) + 5 * np.sin(200 * np.pi * t2 + np.radians(75))
xn2 = 10 * np.cos(250 * np.pi * ts2 + np.radians(60)) + 5 * np.sin(200 * np.pi * ts2 + np.radians(75))

plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(t2, xc2, 'b')
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(ts2, xn2, 'r')
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(n2, xn2, 'g')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(t2, xc2, 'b')
plt.stem(ts2, xn2, 'r')
plt.grid()

plt.tight_layout()


# ==============================================================================
# PROBLEM 3
# Signal reconstruction using manually implemented Sinc Interpolation
# Sampled at Fs = 5*Fo and Fs = 1.5*Fo (Fo = 15 Hz).
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
Fo3 = 15
t3 = np.linspace(0, 2, 1000)
xc3 = np.sin(14 * np.pi * t3) + np.sin(18 * np.pi * t3) + np.sin(24 * np.pi * t3) + np.sin(30 * np.pi * t3)

sampling_rates3 = [5 * Fo3, 1.5 * Fo3]

plt.figure(3)
for i, Fs in enumerate(sampling_rates3):
    Ts = 1 / Fs
    ts = np.arange(0, 2 + Ts, Ts)
    xs = np.sin(14 * np.pi * ts) + np.sin(18 * np.pi * ts) + np.sin(24 * np.pi * ts) + np.sin(30 * np.pi * ts)
    
    N = len(ts)
    xr = np.zeros(len(t3))
    for t_idx in range(len(t3)):
        for n_idx in range(N):
            sinc_val = np.sinc((t3[t_idx] - n_idx * Ts) / Ts)
            xr[t_idx] += xs[n_idx] * sinc_val

    plt.subplot(2, 1, i + 1)
    plt.plot(t3, xc3, 'b')
    plt.stem(ts, xs, 'k')
    plt.plot(t3, xr, 'r--')
    plt.grid()

plt.tight_layout()


# ==============================================================================
# PROBLEM 4
# Signal reconstruction using scipy.signal.resample() (Built-in Sinc Interpolation)
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
Fo4 = 15
t_dense4 = np.linspace(0, 2, 1000)
xc4 = np.sin(14 * np.pi * t_dense4) + np.sin(18 * np.pi * t_dense4) + np.sin(24 * np.pi * t_dense4) + np.sin(30 * np.pi * t_dense4)

sampling_rates4 = [5 * Fo4, 1.5 * Fo4]

plt.figure(4)
for i, Fs in enumerate(sampling_rates4):
    Ts = 1 / Fs
    ts = np.arange(0, 2, Ts)
    xs = np.sin(14 * np.pi * ts) + np.sin(18 * np.pi * ts) + np.sin(24 * np.pi * ts) + np.sin(30 * np.pi * ts)
    
    xr = signal.resample(xs, len(t_dense4))
    
    plt.subplot(2, 1, i + 1)
    plt.plot(t_dense4, xc4, 'b')
    plt.stem(ts, xs, 'k')
    plt.plot(t_dense4, xr, 'r--')
    plt.grid()

plt.tight_layout()


# Display all figures
plt.show()






import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# SECTION 3.1: Plotting a Discrete Signal
# ==========================================

x = np.array([4, 6, 1, -1, -2, 1, 3, 2, 5, 4, -3])
n = np.arange(-5, 6)

plt.figure()
plt.stem(n, x)
plt.xticks(n)
plt.show()


# ==========================================
# SECTION 3.2.1: Unit Impulse Sequence
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def unit_impulse(n0, n1, n2):
    n = np.arange(n1, n2 + 1, 1)
    x = (n == n0)
    return x, n

n0 = 2
n1 = -5
n2 = 5

x, n = unit_impulse(n0, n1, n2)

plt.figure()
plt.stem(n, x)
plt.show()


# ==========================================
# SECTION 3.2.2: Unit Step Sequence
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def unit_step(n0, n1, n2):
    n = np.arange(n1, n2 + 1, 1)
    x = (n >= n0)
    return x, n

n0 = 2
n1 = -5
n2 = 8

x, n = unit_step(n0, n1, n2)

plt.figure()
plt.stem(n, x)
plt.xticks(n)
plt.show()


# ==========================================
# SECTION 3.2.3: Unit Ramp Signal
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def unit_ramp(n0, n1, n2):
    n = np.arange(n1, n2 + 1, 1)
    x = (n - n0) * (n >= n0)
    return x, n

n0 = 2
n1 = -5
n2 = 8

x, n = unit_ramp(n0, n1, n2)

plt.figure()
plt.stem(n, x)
plt.xticks(n)
plt.show()


# ==========================================
# SECTION 3.3.1: Amplitude Scaling
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
n = np.arange(-10, 11, 1)
A = [1, 2, 0.5, -3]
x = (n <= 0)

plt.figure()
for i, a in enumerate(A):
    y = a * x
    plt.subplot(2, 2, i + 1)
    plt.stem(n, y)
    # plt.stem(n, x)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.3.2: Amplitude Shifting (DC Offset)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
Fs = 100
n = np.arange(0, 40, 1)
x = np.sin(2 * np.pi * (5 / Fs) * n)

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n, x)

offset = [5, -5]
y1 = x + offset[0]
y2 = x + offset[1]

plt.subplot(3, 1, 2)
plt.stem(n, y1)

plt.subplot(3, 1, 3)
plt.stem(n, y2)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.3.3: Product of Two Signals
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def signal_mult(x1, n1, x2, n2):
    min_n = min(np.min(n1), np.min(n2))
    max_n = max(np.max(n1), np.max(n2))
    n = np.arange(min_n, max_n + 1)
    
    y1 = np.zeros(len(n))
    y2 = np.zeros(len(n))
    
    y1[np.where(np.isin(n, n1))] = x1
    y2[np.where(np.isin(n, n2))] = x2
    
    y = y1 * y2
    return y, n

n1 = np.arange(-2, 3)
n2 = np.arange(-3, 2)
x1 = np.array([1, -2, 3, -4, 5])
x2 = np.array([5, 4, -3, 2, 3])

y, n = signal_mult(x1, n1, x2, n2)

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n1, x1)
plt.xticks(n1)

plt.subplot(3, 1, 2)
plt.stem(n2, x2)
plt.xticks(n2)

plt.subplot(3, 1, 3)
plt.stem(n, y)
plt.xticks(n)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.3.4: Signal Addition
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
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

n1 = np.arange(-2, 3)
n2 = np.arange(-3, 2)
x1 = np.array([1, -2, 3, -4, 5])
x2 = np.array([5, 4, -3, 2, 3])

y, n = signal_add(x1, n1, x2, n2)

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n1, x1)
plt.xticks(n1)

plt.subplot(3, 1, 2)
plt.stem(n2, x2)
plt.xticks(n2)

plt.subplot(3, 1, 3)
plt.stem(n, y)
plt.xticks(n)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.3.5: Time Scaling (Downsampling & Upsampling)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def upsampling(x, L):
    N = len(x)
    y = np.zeros(L * N)
    y[::L] = x
    return y

def downsampling(x, M):
    return x[::M]

n = np.arange(-10, 10, 1)
Fs = 100
x = np.sin(2 * np.pi * (5 / Fs) * n)

M = 2
m_down = n[::M]
y_down = downsampling(x, M)

L = 2
y_up = upsampling(x, L)
m_up = np.arange(0, len(y_up))

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n, x)
plt.xticks(n)

plt.subplot(3, 1, 2)
plt.stem(m_down, y_down)
plt.xticks(m_down)

plt.subplot(3, 1, 3)
plt.stem(m_up, y_up)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.3.6: Time Shifting (Delay and Advance)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def signal_shift(x, n1, n0):
    n = n1 + n0
    y = x
    return y, n

n = np.arange(-10, 11)
x = (n >= 0)
k = 5

y1, n1 = signal_shift(x, n, k)
y2, n2 = signal_shift(x, n, -k)

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n, x)
plt.xticks(n)
plt.xlim(min(n2) - 1, max(n1) + 1)

plt.subplot(3, 1, 2)
plt.stem(n1, y1)
plt.xticks(n1)
plt.xlim(min(n2) - 1, max(n1) + 1)

plt.subplot(3, 1, 3)
plt.stem(n2, y2)
plt.xticks(n2)
plt.xlim(min(n2) - 1, max(n1) + 1)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.3.7: Time Reversal (Signal Folding)
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def signal_fold(x, n):
    y = np.fliplr([x])[0]
    m = -np.fliplr([n])[0]
    return y, m

n = np.arange(-10, 11)
x = n
y, m = signal_fold(x, n)

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(n, x)
plt.xticks(n)
plt.yticks(np.arange(-10, 11, 5))

plt.subplot(2, 1, 2)
plt.stem(m, y)
plt.xticks(m)
plt.yticks(np.arange(-10, 11, 5))

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.4: Convolution Sum
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
def convolution_sum(x, nx, h, nh):
    kmin = np.min(nx) + np.min(nh)
    kmax = np.max(nx) + np.max(nh)
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(x, h)
    return y, k

nx = np.arange(-5, 6)
x = 5 - np.abs(nx)
h = (nx == 0)
nh = nx

y, n = convolution_sum(x, nx, h, nh)

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(nx, x)
plt.xticks(nx)

plt.subplot(3, 1, 2)
plt.stem(nh, h)
plt.xticks(nh)

plt.subplot(3, 1, 3)
plt.stem(n, y)
plt.xticks(n)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.4.1: Commutative Property of Convolution
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
nx = np.arange(-5, 6)
x = np.array([2, -3, 4, -2, 1, 3, 1, -1, 4, -4, 3])
h = np.array([-3, -2, -1, 0, 1, 2, 1, 0, -1, -2, -3])
nh = np.arange(-3, 8)

y1, n1 = convolution_sum(x, nx, h, nh)
y2, n2 = convolution_sum(h, nh, x, nx)

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(n1, y1)
plt.xticks(n1)

plt.subplot(2, 1, 2)
plt.stem(n2, y2)
plt.xticks(n2)

plt.tight_layout()
plt.show()


# ==========================================
# SECTION 3.5: Autocorrelation and Cross-Correlation
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

def convolution_sum(x, nx, h, nh):
    kmin = np.min(nx) + np.min(nh)
    kmax = np.max(nx) + np.max(nh)
    k = np.arange(kmin, kmax + 1)
    y = np.convolve(x, h)
    return y, k

t = np.linspace(0, 1, 100)
o = np.arange(0, len(t))

x = np.sin(2 * np.pi * 5 * t)

w = 2.5 * np.random.randn(len(t))
y = x + w

w1 = np.flip([y])[0]
t1 = -np.flip([o])[0]

y1, n1 = convolution_sum(x, o, w1, t1)

plt.subplot(311)
plt.plot(t, x)
plt.grid()

plt.subplot(312)
plt.plot(t, y)
plt.grid()

plt.subplot(313)
plt.plot(n1, y1)
plt.grid()

plt.tight_layout()







import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PROBLEM 1: Even and Odd Sequence Decomposition
# ==========================================

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

n = np.arange(-3, 4)
x = np.array([1, 2, 4, 3, -1, 0, 5])

x_folded, n_folded = signal_fold(x, n)

sum_x_xf, n_e = signal_add(x, n, x_folded, n_folded)
x_even = 0.5 * sum_x_xf

neg_x_folded = -1 * x_folded
diff_x_xf, n_o = signal_add(x, n, neg_x_folded, n_folded)
x_odd = 0.5 * diff_x_xf

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n, x)

plt.subplot(3, 1, 2)
plt.stem(n_e, x_even)

plt.subplot(3, 1, 3)
plt.stem(n_o, x_odd)

plt.tight_layout()
plt.show()


# ==========================================
# PROBLEM 2: Sequence Folding and Shifting
# ==========================================

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

x2 = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
n2 = np.arange(0, len(x2))

x_f, n_f = signal_fold(x2, n2)

y_i, n_i = signal_shift(x_f, n_f, -3)

y_ii, n_ii = signal_shift(x_f, n_f, 4)

plt.figure()
plt.subplot(3, 1, 1)
plt.stem(n2, x2)

plt.subplot(3, 1, 2)
plt.stem(n_i, y_i)

plt.subplot(3, 1, 3)
plt.stem(n_ii, y_ii)

plt.tight_layout()
plt.show()


# ==========================================
# PROBLEM 3: Commutativity Verification (Shifting & Folding)
# ==========================================

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

x2 = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
n2 = np.arange(0, len(x2))

x_fold_first, n_fold_first = signal_fold(x2, n2)
y_pathA, n_pathA = signal_shift(x_fold_first, n_fold_first, 3)

y_shift_first, n_shift_first = signal_shift(x2, n2, 3)
y_pathB, n_pathB = signal_fold(y_shift_first, n_shift_first)

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(n_pathA, y_pathA)

plt.subplot(2, 1, 2)
plt.stem(n_pathB, y_pathB)

plt.tight_layout()
plt.show()


# ==========================================
# PROBLEM 4: Associative Property of Convolution
# ==========================================

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

x_seq = np.array([1, 2, 3, 4])
nx_seq = np.arange(0, 4)

h1 = np.array([1, -1, 2])
nh1 = np.arange(-1, 2)

h2 = np.array([0, 1, 1])
nh2 = np.arange(0, 3)

y_temp, ny_temp = convolution_sum(x_seq, nx_seq, h1, nh1)
y_left, ny_left = convolution_sum(y_temp, ny_temp, h2, nh2)

h_eq, nh_eq = convolution_sum(h1, nh1, h2, nh2)
y_right, ny_right = convolution_sum(x_seq, nx_seq, h_eq, nh_eq)

plt.figure()
plt.subplot(2, 1, 1)
plt.stem(ny_left, y_left)

plt.subplot(2, 1, 2)
plt.stem(ny_right, y_right)

plt.tight_layout()
plt.show()


# ==========================================
# PROBLEM 5: Echo Signal Generation & Autocorrelation
# ==========================================
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


# ==========================================
# PROBLEM 6: Non-commutativity of Cross-Correlation
# ==========================================

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






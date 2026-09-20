"""
EEE 3218 - Digital Signal Processing Lab (AUST)
================================================
Laboratory 4 : Transforms and Discrete-Time Systems
Laboratory 5 : Digital Filter Design (Part - 1)
Laboratory 6 : Digital Filter Design (Part - 2)

This script contains:
  1. The example codes exactly as given in the lab manual (organised into
     functions, one per example, so they can be run independently).
  2. Solutions to the Post-Laboratory Problems, written in the same style
     and using the same functions/tools that are taught in the manual
     (scipy.signal, numpy.fft, control.pzmap, lcapy, signal.firwin,
     signal.bilinear, signal.butter, signal.cheby2, etc.)

Textbook-only problems (asking to "solve exercise problems of the
textbook from page X to page Y") are not included since they require
the external textbook which is not part of this manual.

Requirements: numpy, scipy, matplotlib, control, lcapy
    pip install numpy scipy matplotlib control lcapy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import dft
import control as ct


# ==================================================================
#                       LABORATORY 4
#            Transforms and Discrete-Time Systems
# ==================================================================

# ---------------------------------------------------------------
# 4.2.1  Computation of DFT of a sequence using DFT matrix
# ---------------------------------------------------------------
def lab4_4_2_1():
    N = 4
    x = np.ones(N)
    W = dft(N)
    X = W @ x  # matrix multiplication
    MagX = np.abs(x)
    print(MagX)


# ---------------------------------------------------------------
# 4.2.2  Computation of Inverse DFT through DFT matrix
# ---------------------------------------------------------------
def lab4_4_2_2():
    N = 4
    X = [4, 0, 0, 0]
    W = dft(N)
    x = (W @ X) / N
    MagX = np.abs(x)
    print(MagX)


# ---------------------------------------------------------------
# 4.2.3  N-point DFT of a sequence and reconstruction from IDFT
# ---------------------------------------------------------------
def lab4_4_2_3():
    N = 5
    arr1 = np.array([1, 1, 0, 0, 1])
    arr2 = np.zeros(N - len(arr1))
    x = np.concatenate((arr1, arr2), axis=0)
    W = dft(N)
    X = W @ x  # matrix multiplication
    k = np.arange(N)
    MagX = np.abs(X)
    PhaseX = np.angle(X)

    plt.subplot(2, 1, 1)
    plt.stem(k, MagX)
    plt.title('Magnitude Spectrum')
    plt.xlabel('k')
    plt.ylabel('Magnitude')
    plt.grid(linestyle='--')

    plt.subplot(2, 1, 2)
    plt.stem(k, PhaseX)
    plt.title('Phase Spectrum')
    plt.xlabel('k')
    plt.ylabel('Phase (degrees)')
    plt.grid(linestyle='--')
    plt.tight_layout()

    x = (W @ X) / N
    x = np.real(x)
    n = np.arange(N)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(n, x)
    plt.title('Inverse DFT (Original Signal)')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(linestyle='--')
    plt.tight_layout()


# ---------------------------------------------------------------
# 4.3.1  N-point FFT of a sequence and reconstruction from IFFT
# ---------------------------------------------------------------
def lab4_4_3_1():
    N = 32

    arr1 = np.array([1, 1, 1, 1, 1])
    arr2 = np.zeros(N - len(arr1))
    x = np.concatenate((arr1, arr2), axis=0)

    X = np.fft.fft(x, N)
    k = np.arange(N)
    MagX = np.abs(X)
    PhaseX = np.angle(X)

    plt.subplot(2, 1, 1)
    plt.stem(k, MagX)
    plt.title('Magnitude Spectrum')
    plt.xlabel('k')
    plt.ylabel('Magnitude')
    plt.grid(linestyle='--')

    plt.subplot(2, 1, 2)
    plt.stem(k, PhaseX)
    plt.title('Phase Spectrum')
    plt.xlabel('k')
    plt.ylabel('Phase (degrees)')
    plt.ylabel('Amplitude')
    plt.grid(linestyle='--')

    plt.tight_layout()
    plt.show()

    x_reconstructed = np.fft.ifft(X, N)
    x_reconstructed = np.real(x_reconstructed)  # Take only the real part
    n = np.arange(N)  # Create the time axis for plotting

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(n, x_reconstructed)
    plt.title('Inverse DFT (Original Signal)')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 4.3.2  Computation of the frequency spectrum of a sinc function
# ---------------------------------------------------------------
def lab4_4_3_2():
    t0 = 0.2       # duration of sinc pulse in seconds
    ts = 8.3333e-4  # sampling period in seconds
    fs = 1 / ts    # sampling frequency in Hz

    t = np.arange(-t0 / 2, t0 / 2 + ts, ts)
    x = np.sinc(100 * t)
    # The argument 100*t is scaled correctly for the desired pulse width.
    N = 256  # Number of points for the DFT
    X = np.fft.fft(x, N)
    freq = np.fft.fftfreq(N, d=ts)
    # The fftfreq function is the most robust way to create the frequency axis.
    # It handles the zero-centered and positive-only ranges correctly.
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, x, 'b-', linewidth=2)
    plt.title('Sinc Pulse')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.stem(np.fft.fftshift(freq), np.abs(np.fft.fftshift(X)) / N)
    # np.fft.fftshift() centers the zero frequency
    # The magnitude is divided by N to scale it correctly
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.stem(np.fft.fftshift(freq), np.angle(np.fft.fftshift(X)) * 180 / np.pi)
    # np.angle() returns the phase in radians, so we convert it to degrees
    plt.title('Phase Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid(True)
    plt.tight_layout()


# ---------------------------------------------------------------
# 4.3.3  Convolution using DFT
# ---------------------------------------------------------------
def lab4_4_3_3():
    x = np.array([2, 1, 2, 1])       # signal x
    h = np.array([1, 2, 3, 4, 3])    # signal h
    conv_len = len(x) + len(h) - 1
    # The length of the linear convolution of two signals with lengths M and N
    # is M + N - 1. We must pad the signals to this length for the
    x_padded = np.pad(x, (0, conv_len - len(x)))
    h_padded = np.pad(h, (0, conv_len - len(h)))
    # Pad both signals with zeros so they have a length equal to conv_len.
    # This makes the circular convolution result identical to
    # the linear convolution.
    X = np.fft.fft(x_padded, conv_len)  # Perform DFT
    H = np.fft.fft(h_padded, conv_len)  # Perform DFT
    Y = X * H
    y = np.real(np.fft.ifft(Y, conv_len))
    # Perform IDFT to get the result in the Time Domain
    print("Signal x:", x)
    print("Signal h:", h)
    print("Padded signal x:", x_padded)
    print("Padded signal h:", h_padded)
    print("Length of convolution:", conv_len)
    print("Result of linear convolution (y):", y)

    y_verified = np.convolve(x, h)
    print("Result verified with np.convolve:", y_verified)
    # The result can also be verified using NumPy's built-in convolution function
    # This is for verification purposes and not part of the DFT method.


# ---------------------------------------------------------------
# 4.3.4  Modulation and demodulation of a message signal using FFT
# ---------------------------------------------------------------
def lab4_4_3_4():
    ### --- 1. Signal and System Parameters ---
    t0 = 0.2   # duration of the sinc pulse
    ts = 0.001  # sampling period
    fc = 250   # carrier frequency in Hz
    fs = 1 / ts  # sampling frequency in Hz
    t = np.arange(-t0 / 2, t0 / 2, ts)  # Time axis for the signals
    ### --- 2. Signal Generation and Modulation ---
    m = np.sinc(100 * t)                # The message signal
    c = np.cos(2 * np.pi * fc * t)      # The carrier signal,
    u = m * c   # DSB-SC modulation: a simple element-wise multiplication
    ### --- 3. Spectral Analysis using FFT ---
    N = 1024   # FFT bin size
    freq = np.fft.fftfreq(N, d=1 / fs)
    # Generate the correct frequency axis for plotting
    M_fft = np.fft.fft(m, N) / fs
    # Scale the N-point DFTs by the sampling frequency
    U_fft = np.fft.fft(u, N) / fs
    ### --- 4. Plotting: Message and Modulated Spectra ---
    # Plot the spectrum of the message signal
    plt.subplot(2, 1, 1)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(M_fft)), linewidth=2)
    plt.title('Spectrum of the Message Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    # Plot the spectrum of the modulated signal
    plt.subplot(2, 1, 2)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(U_fft)), linewidth=2)
    plt.title('Spectrum of the Modulated Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- continue the code after 'Modulation' part
    # --- 5. Demodulation and Filtering ---
    y = u * c   # Demodulation is multiplication by the carrier again
    Y_fft = np.fft.fft(y, N) / fs  # N-point DFT of the demodulated signal
    # Create a simple low-pass filter in the frequency domain
    fcut = 200  # Sets the cutoff frequency for the low-pass filter to 200 Hz.
    ncut = int(np.floor(fcut / (fs / N)))
    # Calculates the number of frequency bins corresponding to the fcut
    H_filter = np.zeros(N, dtype=np.float64)
    # Creates an array of zeros to serve as the frequency-domain filter.
    H_filter[0:ncut] = 2
    # Sets the gain for the positive frequencies up to the cutoff to 2.
    H_filter[N - ncut:N] = 2
    # Sets the gain for the corresponding negative frequencies to 2,
    # which is required for a symmetric filter response.
    U_filtered = Y_fft * H_filter
    # Apply the filter by multiplying in the frequency domain
    # --- 6. Plotting: Demodulated and Filtered Spectra ---
    # Plot the spectrum of the demodulated signal
    plt.subplot(2, 1, 1)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(Y_fft)))
    plt.title('Spectrum of Demodulated Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    # Plot the spectrum of the filtered signal
    plt.subplot(2, 1, 2)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(U_filtered)))
    plt.title('Spectrum of Filtered Demodulated Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # --- 7. Plotting: Reconstructed Time-Domain Signal ---
    # Use np.real() to remove any imaginary components due to rounding errors
    u_reconstructed = np.real(np.fft.ifft(U_filtered) * fs)
    # Plot the original message signal
    plt.subplot(2, 1, 1)
    plt.plot(t, m, linewidth=2)
    plt.title('Original Message Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    # Plot the reconstructed signal.
    plt.subplot(2, 1, 2)
    plt.plot(t, u_reconstructed[:len(t)])
    # Slice to match the original signal's length.
    plt.title('Reconstructed Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 4.4.1  Z-Transform of Unit Sample and Unit Step Sequences using lcapy
# ---------------------------------------------------------------
def lab4_4_4_1():
    from lcapy import n, delta, us
    x = delta(n)
    Xz = x.ZT()
    print(Xz)
    x1 = us(n)
    Yz = x1.ZT()
    print(Yz)


# ---------------------------------------------------------------
# 4.4.2  Z-Transform of x[n]=e^{jn}u[n], x[n]=cos(n), x[n]=(1/2)^n u[n]
# ---------------------------------------------------------------
def lab4_4_4_2():
    from lcapy import n, exp, cos, us
    x1 = exp(1j * n)
    x2 = cos(n)
    x3 = (1 / 2) ** n * us(n)
    X1 = x1.ZT()
    print(X1)
    X2 = x2.ZT()
    print(X2)
    X3 = x3.ZT()
    print(X3)


# ---------------------------------------------------------------
# 4.4.3  Inverse Z-Transform of X(z)=z^-1, X(z)=1/(1-z^-1) using lcapy
# ---------------------------------------------------------------
def lab4_4_4_3():
    from lcapy import z
    X = z ** (-1)
    x = X.IZT()
    print(x)
    Y = 1 / (1 - z ** (-1))
    y = Y.IZT()
    print(y)


# ---------------------------------------------------------------
# 4.4.4  Inverse Z-Transform using partial fraction expansion
# ---------------------------------------------------------------
def lab4_4_4_4():
    num = [1, 2, -1]         # Numerator coefficients
    den = [1, -1, 0.3561]    # Denominator coefficients
    r, p, k = signal.residuez(num, den)
    print("--- Partial Fraction Expansion Results ---")
    print("Residues (r):")
    print(r)
    print("\nPoles (p):")
    print(p)
    print("\nDirect terms (k):")
    print(k)
    print("-" * 40)

    num_rc, den_rc = signal.invresz(r, p, k)
    print("\nReconstructed Numerator:")
    print(num_rc)
    print("\nReconstructed Denominator:")
    print(den_rc)


# ---------------------------------------------------------------
# 4.5.1  Difference Equation Representation of a Discrete-Time LTI System
# ---------------------------------------------------------------
def lab4_4_5_1():
    n = np.arange(0, 4)
    x = np.ones(len(n))  # Input
    coeff_x = [1]
    coeff_y = [1, -1 / 2]
    y = signal.lfilter(coeff_x, coeff_y, x)

    plt.stem(y)
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.tight_layout()


# ---------------------------------------------------------------
# 4.5.2  Impulse Response of Discrete-Time System
# ---------------------------------------------------------------
def lab4_4_5_2():
    def unit_impulse(n0, n1, n2):
        n = np.arange(n1, n2 + 1, 1)
        x = (n == n0)
        return x, n
    n0 = 0
    n1 = 0
    n2 = 50
    x, n = unit_impulse(n0, n1, n2)  # Generation of unit sample signal
    x = x.astype(float)  # Convert from boolean to floating-point number
    coeff_x = [1]      # Define the system
    coeff_y = [1, -1]
    h = signal.lfilter(coeff_x, coeff_y, x)  # Obtaining the impulse response
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title('Impulse response (h[n])')


# ---------------------------------------------------------------
# 4.5.3  Pole-Zero Plot of Discrete-Time System
# ---------------------------------------------------------------
def lab4_4_5_3():
    num = [1, 2, -1]
    den = [1, -1, 0.3561]
    sys = ct.tf(num, den, dt=True)
    ct.pzmap(sys, title=False)


# ---------------------------------------------------------------
# 4.5.4  Frequency Response of a Discrete-Time System
# ---------------------------------------------------------------
def lab4_4_5_4():
    num = [1, -1.6180, 1]
    den = [1, -1.5371, 0.9025]
    Fs = 512
    N = 256
    w, h = signal.freqz(num, den, worN=N, fs=Fs)

    plt.subplot(2, 1, 1)
    magnitude_db = 20 * np.log10(np.abs(h))
    plt.plot(w, magnitude_db)
    plt.title('Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, linestyle='--')

    plt.subplot(2, 1, 2)
    phase_degrees = np.angle(h, deg=True)
    plt.plot(w, phase_degrees)
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid(True, linestyle='--')
    plt.tight_layout()


# ==================================================================
#           LABORATORY 4 : POST-LABORATORY PROBLEM SOLUTIONS
# ==================================================================

# 1. y[n]-4y[n-1]+4y[n-2] = x[n]-x[n-1]. Impulse response (i) by solving
#    the difference equation (via Z-transform / partial fractions, as in
#    4.4.4) and (ii) using lfilter (as in 4.5.2). Show output for
#    -20<=n<=25. Also show response to the given sinusoidal input.
def lab4_post_1():
    b = [1, -1]       # numerator (coefficients of x[n], x[n-1])
    a = [1, -4, 4]    # denominator (coefficients of y[n], y[n-1], y[n-2])

    n = np.arange(-20, 26)
    x = np.zeros_like(n, dtype=float)
    x[n == 0] = 1  # unit impulse delta[n]

    # (ii) impulse response using lfilter
    h_lfilter = signal.lfilter(b, a, x)

    # (i) impulse response by solving the difference equation using
    #     Z-transform / partial fraction expansion (same tool as 4.4.4)
    r, p, k = signal.residuez(b, a)
    h_analytic = np.zeros_like(n, dtype=complex)
    for i in range(len(n)):
        if n[i] >= 0:
            h_analytic[i] = np.sum(r * (p ** n[i]))
    h_analytic = np.real(h_analytic)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(n, h_analytic)
    plt.title('Impulse response h[n] (solved via difference equation)')
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.grid(linestyle='--')

    plt.subplot(2, 1, 2)
    plt.stem(n, h_lfilter)
    plt.title('Impulse response h[n] (using lfilter)')
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.grid(linestyle='--')
    plt.tight_layout()

    # Response to x[n] = (5 + 3cos(0.2*pi*n) + 4sin(0.6*pi*n)) u[n]
    xin = (5 + 3 * np.cos(0.2 * np.pi * n) + 4 * np.sin(0.6 * np.pi * n)) * (n >= 0)
    y = signal.lfilter(b, a, xin)
    plt.figure()
    plt.stem(n, y)
    plt.title('Output y[n] for x[n] = (5+3cos(0.2\u03c0n)+4sin(0.6\u03c0n))u[n]')
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.grid(linestyle='--')
    plt.tight_layout()
    # Note: the poles of this system are a double pole at z = 2 (outside
    # the unit circle), so the system is unstable and both h[n] and y[n]
    # grow without bound - this is the expected, correct result.


# 2. H(z) = (z^-1 + 0.5 z^-2) / (1 - (3/5) z^-1 + (2/25) z^-2). Frequency response.
def lab4_post_2():
    num = [0, 1, 0.5]
    den = [1, -3 / 5, 2 / 25]
    w, h = signal.freqz(num, den, worN=512)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(linestyle='--')

    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h, deg=True))
    plt.title('Phase Response')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Phase (degrees)')
    plt.grid(linestyle='--')
    plt.tight_layout()


# 3. H(z) = (z^3-2z^2+2z-1) / [(z-1)(z-0.5)(z-0.2)]. Pole-zero plot & stability.
def lab4_post_3():
    num = [1, -2, 2, -1]
    den = np.poly([1, 0.5, 0.2])  # (z-1)(z-0.5)(z-0.2)
    sys = ct.tf(num, den, dt=True)
    plt.figure()
    ct.pzmap(sys, title=False)

    poles = np.roots(den)
    print("Poles:", poles)
    print("System stable (all poles inside unit circle)?",
          bool(np.all(np.abs(poles) < 1)))


# 4. Partial fraction expansion of
#    X(z) = (2z^4+16z^3+44z^2+56z+32) / (3z^4+3z^3-15z^2+18z-12)
def lab4_post_4():
    num = [2, 16, 44, 56, 32]
    den = [3, 3, -15, 18, -12]
    r, p, k = signal.residuez(num, den)
    print("Residues (r):", r)
    print("Poles (p):", p)
    print("Direct terms (k):", k)

    sys = ct.tf(num, den, dt=True)
    plt.figure()
    ct.pzmap(sys, title=False)
    print("Causal & stable (all poles inside unit circle)?",
          bool(np.all(np.abs(p) < 1)))


# 5. x(t) = sinc^2(100t). Sample at 5 kHz and find the DFT spectrum.
def lab4_post_5():
    fs = 5000.0
    ts = 1 / fs
    t0 = 0.2
    t = np.arange(-t0 / 2, t0 / 2 + ts, ts)
    x = np.sinc(100 * t) ** 2
    N = 256
    X = np.fft.fft(x, N)
    freq = np.fft.fftfreq(N, d=ts)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title('x(t) = sinc^2(100t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.stem(np.fft.fftshift(freq), np.abs(np.fft.fftshift(X)) / N)
    plt.title('DFT Magnitude Spectrum (Fs = 5 kHz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()


# 6. x[n] = {1,1,2,3,5,8,13} (first sample at n=0). Autocorrelation via FFT.
def lab4_post_6():
    x = np.array([1, 1, 2, 3, 5, 8, 13])
    N = 2 * len(x) - 1
    X = np.fft.fft(x, N)
    Rxx = np.real(np.fft.ifft(X * np.conj(X), N))
    Rxx = np.fft.fftshift(Rxx)
    lag = np.arange(-(len(x) - 1), len(x))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(np.arange(len(x)), x)
    plt.title('x[n]')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(linestyle='--')

    plt.subplot(2, 1, 2)
    plt.stem(lag, Rxx)
    plt.title('Autocorrelation r_xx(l) using FFT')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(linestyle='--')
    plt.tight_layout()


# 7. FDM system (Figure 4.14). m1(t)=sinc(100t), m2(t)=sinc^2(100t) for
#    |t|<=0.1, fc1=250Hz, fc2=750Hz. Spectra of m1, m2 and output y(t).
def lab4_post_7():
    t0 = 0.2
    ts = 0.001
    fs = 1 / ts
    t = np.arange(-t0 / 2, t0 / 2, ts)
    m1 = np.sinc(100 * t)
    m2 = np.sinc(100 * t) ** 2
    fc1 = 250
    fc2 = 750
    c1 = np.cos(2 * np.pi * fc1 * t)
    c2 = np.cos(2 * np.pi * fc2 * t)
    y = m1 * c1 + m2 * c2   # FDM output y(t)

    N = 1024
    freq = np.fft.fftfreq(N, d=ts)
    M1_fft = np.fft.fft(m1, N) / fs
    M2_fft = np.fft.fft(m2, N) / fs
    Y_fft = np.fft.fft(y, N) / fs

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(M1_fft)))
    plt.title('Spectrum of m1(t)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(M2_fft)))
    plt.title('Spectrum of m2(t)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.fft.fftshift(freq), np.abs(np.fft.fftshift(Y_fft)))
    plt.title('Spectrum of FDM output y(t)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.tight_layout()


# Problems 8 and 9 of Laboratory 4 ask to "solve all exercise problems of
# the textbook" from specific page ranges. These require the external
# textbook (not included in the lab manual PDF) and are therefore not
# solved here.


# ==================================================================
#                       LABORATORY 5
#              Digital Filter Design (Part - 1)
# ==================================================================

# ---------------------------------------------------------------
# 5.5.1  First-Order IIR Filter  H(z) = 1/(1-z^-1)
# ---------------------------------------------------------------
def lab5_5_5_1():
    x = np.zeros(30)
    x[0] = 1  # Unit impulse delta[n]
    num = [1]
    den = [1, -1]   # H(z) = 1 / (1 - z^-1)
    fs = 100   # Sample frequency (for plotting frequency response)
    sys = ct.tf(num, den, dt=True)
    h = signal.lfilter(num, den, x)  # Obtain the impulse response
    w, H = signal.freqz(num, den)
    # w: normalized angular frequency (rad/sample), H: complex response

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)    # Impulse Response (Time Domain)
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title('Impulse Response (Unit Step)')
    plt.grid(linestyle=':')

    plt.subplot(2, 2, 2)    # Magnitude Response (Frequency Domain)
    plt.plot((w / np.pi) * fs / 2, 20 * np.log10(np.abs(H)))
    # Convert normalized frequency 'w' to Hz (w * fs / (2*pi))
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Magnitude Response (Low-Pass)')
    plt.grid(linestyle=':')

    plt.subplot(2, 2, 3)    # Pole-zero plot
    ct.pzmap(sys, title=False)

    plt.subplot(2, 2, 4)    # Phase Response (Frequency Domain)
    plt.plot((w / np.pi) * fs / 2, np.angle(H))
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Phase (Radians)')
    plt.title('Phase Response')
    plt.grid(linestyle=':')
    plt.tight_layout()


# ---------------------------------------------------------------
# 5.5.2  Three-Point Moving Average Filter
# ---------------------------------------------------------------
def lab5_5_5_2():
    x = np.zeros(30)
    x[0] = 1
    num = [1 / 3, 1 / 3, 1 / 3]
    den = [1]
    fs = 100
    h = signal.lfilter(num, den, x)
    w, H = signal.freqz(num, den)
    sys = ct.tf(num, den, dt=True)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title('Impulse Response (Unit Step)')
    plt.grid(linestyle=':')

    plt.subplot(2, 2, 2)
    plt.plot((w / np.pi) * fs / 2, 20 * np.log10(np.abs(H)))
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Magnitude Response (Low-Pass)')
    plt.grid(linestyle=':')

    plt.subplot(2, 2, 3)
    ct.pzmap(sys, title=False, marker_size=16)

    plt.subplot(2, 2, 4)
    plt.plot((w / np.pi) * fs / 2, np.angle(H))
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Phase (Radians)')
    plt.title('Phase Response')
    plt.grid(linestyle=':')
    plt.tight_layout()


# ---------------------------------------------------------------
# 5.5.3  Digital Resonator with two complex conjugate poles, r = 0.90
# ---------------------------------------------------------------
def lab5_5_5_3():
    x = np.zeros(30)
    x[0] = 1
    r = 0.9
    fs = 100   # Sampling frequency
    fc = 10    # Cutoff frequency
    w = 2 * np.pi * (fc / fs)
    num = [1]
    den = [1, -2 * r * np.cos(w), r ** 2]
    h = signal.lfilter(num, den, x)
    w, H = signal.freqz(num, den)
    sys = ct.tf(num, den, dt=True)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title('Impulse Response (Unit Step)')
    plt.grid(linestyle=':')

    plt.subplot(2, 2, 2)
    plt.plot((w / np.pi) * fs / 2, 20 * np.log10(np.abs(H)))
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Magnitude Response (Low-Pass)')
    plt.grid(linestyle=':')

    plt.subplot(2, 2, 3)
    ct.pzmap(sys, title=False, marker_size=16)

    plt.subplot(2, 2, 4)
    plt.plot((w / np.pi) * fs / 2, np.angle(H))
    plt.xlabel('Normalized Frequency (rad/sample)')
    plt.ylabel('Phase (Radians)')
    plt.title('Phase Response')
    plt.grid(linestyle=':')
    plt.tight_layout()


# ---------------------------------------------------------------
# 5.6 Example 1: Design a FIR low pass filter (Blackman window)
# ---------------------------------------------------------------
def lab5_5_6_example1():
    Fs = 8000.0   # Sampling frequency (Hz)
    TW = 500.0    # Transition width (Hz)
    PBE = 1500.0  # Passband Edge frequency (Hz)

    M = int(np.round(6 * Fs / TW))
    filter_length = M + 1

    Fc = PBE + TW / 2.0    # Cutoff frequency

    a = signal.firwin(
        numtaps=filter_length,
        cutoff=Fc,
        window='blackman',
        pass_zero='lowpass',
        fs=Fs
    )

    N = 512  # number of points for calculating frequency response
    f, h = signal.freqz(a, 1, worN=N, fs=Fs)

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(f, 20 * np.log10(np.abs(h)), linewidth=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.axvline(PBE, color='red', linestyle='--',
                label=f'Passband Edge ({PBE} Hz)')
    plt.axvline(PBE + TW, color='purple', linestyle='--',
                label='Stopband Edge (2000 Hz)')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.degrees(np.unwrap(np.angle(h))), linewidth=2, color='green')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle (Degree)")
    plt.grid()
    plt.tight_layout()

    print(f"Filter Order (M): {M}")
    print(f"Filter Length (Taps): {filter_length}")
    # Note: the manual prints an undefined variable {corner} here; the
    # intended value is the compensated ideal cutoff frequency Fc.
    print(f"Compensated Ideal Cutoff Frequency: {Fc} Hz")


# ---------------------------------------------------------------
# 5.6 Example 2: Design a FIR band pass filter (Hamming window)
# ---------------------------------------------------------------
def lab5_5_6_example2():
    Fs = 100    # Sampling frequency (Hz)
    M = 50      # Filter order
    Fl = 10     # Lower cutoff frequency (Hz)
    Fu = 20     # Upper cutoff frequency (Hz)
    filter_length = M + 1
    Fc = [Fl, Fu]  # Normalized cutoffs

    a = signal.firwin(
        numtaps=filter_length,
        cutoff=Fc,
        window='hamming',
        pass_zero='bandpass',
        fs=Fs
    )

    N = 512  # number of points for calculating frequency response
    f, h = signal.freqz(a, 1, worN=N, fs=Fs)

    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(f, 20 * np.log10(np.abs(h)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.axvline(Fl, color='red', linestyle='--',
                label=f'Lower cutoff ({Fl} Hz)')
    plt.axvline(Fu, color='purple', linestyle='--',
                label=f'Upper cutoff ({Fu} Hz)')
    plt.axhline(-3, color='orange', linestyle='-.', label='-3 dB Mark')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.degrees(np.unwrap(np.angle(h))), color='green')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Angle (Degree)")
    plt.grid()
    plt.tight_layout()


# ==================================================================
#           LABORATORY 5 : POST-LABORATORY PROBLEM SOLUTIONS
# ==================================================================

# 1. 50th order Highpass filter, Fs = 5kHz, fc = 1kHz, for each window
#    mentioned in the manual (rectangular, Bartlett, Hann, Hamming, Blackman)
def lab5_post_1():
    Fs = 5000
    fc = 1000
    M = 50
    windows = ['boxcar', 'bartlett', 'hann', 'hamming', 'blackman']

    plt.figure(figsize=(10, 8))
    for win in windows:
        a = signal.firwin(numtaps=M + 1, cutoff=fc, window=win,
                           pass_zero='highpass', fs=Fs)
        f, h = signal.freqz(a, 1, worN=512, fs=Fs)
        plt.subplot(2, 1, 1)
        plt.plot(f, 20 * np.log10(np.abs(h)), label=win)
        plt.subplot(2, 1, 2)
        plt.plot(f, np.degrees(np.unwrap(np.angle(h))), label=win)

    plt.subplot(2, 1, 1)
    plt.title('Magnitude Response - 50th Order Highpass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle (Degree)')
    plt.grid()
    plt.legend()
    plt.tight_layout()


# 2. FIR bandpass: passband 150-250Hz, transition width 50Hz, passband
#    ripple 0.05dB, stopband attenuation 50dB, Fs = 1kHz.
def lab5_post_2():
    Fs = 1000.0
    TW = 50.0
    Fl = 150.0
    Fu = 250.0
    # From Table 5.3, the Hamming window gives >50 dB stopband
    # attenuation (53 dB) with reasonable transition width, and is
    # chosen here (the Blackman window would also work but needs a
    # longer filter for the same transition width).
    M = int(np.round(6.6 * Fs / TW))   # Hamming main-lobe width factor
    filter_length = M + 1
    a = signal.firwin(numtaps=filter_length, cutoff=[Fl, Fu],
                       window='hamming', pass_zero='bandpass', fs=Fs)
    f, h = signal.freqz(a, 1, worN=512, fs=Fs)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f, 20 * np.log10(np.abs(h)))
    plt.axvline(Fl, color='red', linestyle='--')
    plt.axvline(Fu, color='purple', linestyle='--')
    plt.title('Magnitude Response (Hamming window)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.degrees(np.unwrap(np.angle(h))), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle (Degree)')
    plt.grid()
    plt.tight_layout()

    print(f"Filter Order (M): {M}")
    print("Chosen window: Hamming - provides ~53 dB stopband attenuation, "
          "which exceeds the 50 dB requirement, with a narrower transition "
          "band (and hence lower filter order) than the Blackman window.")


# 3. Characteristics of filters with the given transfer functions.
def lab5_post_3():
    systems = {
        'a) H(z)=1/(1+2z^-1)': ([1], [1, 2]),
        'b) H(z)=1/(1-2z^-1)': ([1], [1, -2]),
        # c) H(z)=1-2cos(w)z^-1+z^-2 : FIR notch-type filter (w = 1 rad used
        #    here as an example angular location for the zeros)
        'c) H(z)=1-2cos(1)z^-1+z^-2': ([1, -2 * np.cos(1), 1], [1]),
        # d) H(z)=(z^-1-a)/(1-az^-1), |a|<1  (a = 0.5 used as example)
        'd) H(z)=(z^-1-0.5)/(1-0.5z^-1)': ([-0.5, 1], [1, -0.5]),
    }
    for title, (num, den) in systems.items():
        x = np.zeros(30)
        x[0] = 1
        h = signal.lfilter(num, den, x)
        w, H = signal.freqz(num, den, worN=512)
        sys = ct.tf(num, den, dt=True)

        plt.figure(figsize=(10, 8))
        plt.suptitle(title)

        plt.subplot(2, 2, 1)
        plt.stem(h)
        plt.xlabel('n')
        plt.ylabel('h[n]')
        plt.title('Impulse Response')
        plt.grid(linestyle=':')

        plt.subplot(2, 2, 2)
        plt.plot(w, 20 * np.log10(np.abs(H) + 1e-12))
        plt.xlabel('Frequency (rad/sample)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Magnitude Response')
        plt.grid(linestyle=':')

        plt.subplot(2, 2, 3)
        ct.pzmap(sys, title=False)

        plt.subplot(2, 2, 4)
        plt.plot(w, np.angle(H))
        plt.xlabel('Frequency (rad/sample)')
        plt.ylabel('Phase (Radians)')
        plt.title('Phase Response')
        plt.grid(linestyle=':')
        plt.tight_layout()


# 4. High-Pass FIR: passband edge 150Hz, Fs=1000Hz, order 60, Blackman window.
def lab5_post_4():
    Fs = 1000
    fc = 150
    M = 60
    a = signal.firwin(numtaps=M + 1, cutoff=fc, window='blackman',
                       pass_zero='highpass', fs=Fs)
    f, h = signal.freqz(a, 1, worN=512, fs=Fs)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f, 20 * np.log10(np.abs(h)))
    plt.axvline(fc, color='red', linestyle='--')
    plt.title('Magnitude Response - Highpass FIR (Blackman)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.degrees(np.unwrap(np.angle(h))), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle (Degree)')
    plt.grid()
    plt.tight_layout()


# 5. Band-Stop FIR to reject 45-55Hz, Fs=300Hz, order 120, Kaiser beta=8.0
def lab5_post_5():
    Fs = 300
    M = 120
    a = signal.firwin(numtaps=M + 1, cutoff=[45, 55],
                       window=('kaiser', 8.0), pass_zero='bandstop', fs=Fs)
    f, h = signal.freqz(a, 1, worN=512, fs=Fs)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f, 20 * np.log10(np.abs(h)))
    plt.axvline(45, color='red', linestyle='--')
    plt.axvline(55, color='purple', linestyle='--')
    plt.title('Magnitude Response - Bandstop FIR (Kaiser, \u03b2=8.0)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.degrees(np.unwrap(np.angle(h))), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle (Degree)')
    plt.grid()
    plt.tight_layout()


# 6. Low-Pass FIR, cutoff 20kHz, Fs=100kHz, order 40, Rectangular window
#    (highlights the Gibbs phenomenon).
def lab5_post_6():
    Fs = 100000
    fc = 20000
    M = 40
    a = signal.firwin(numtaps=M + 1, cutoff=fc, window='boxcar',
                       pass_zero='lowpass', fs=Fs)
    f, h = signal.freqz(a, 1, worN=512, fs=Fs)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f, 20 * np.log10(np.abs(h)))
    plt.axvline(fc, color='red', linestyle='--')
    plt.title('Magnitude Response - Lowpass FIR (Rectangular window)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(f, np.degrees(np.unwrap(np.angle(h))), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Angle (Degree)')
    plt.grid()
    plt.tight_layout()


# Problems 7 and 8 of Laboratory 5 ask to solve textbook exercise problems
# from specific page ranges; these are not included here (external textbook).


# ==================================================================
#                       LABORATORY 6
#              Digital Filter Design (Part - 2)
# ==================================================================

# ---------------------------------------------------------------
# 6.5  Pole-Zero Method: bandpass filter, DC & 250Hz rejection,
#      passband centered at 125Hz, 3dB bandwidth 10Hz, Fs = 500Hz
# ---------------------------------------------------------------
def lab6_6_5():
    Fs = 500   # Sampling frequency (Hz)
    bw = 10    # Bandwidth (Hz)
    Fr1 = 0    # DC frequency
    Fr2 = 250  # Nyquist frequency (Fs/2)
    # Angle corresponding to the zeros (theta = 2*pi*F/Fs)
    theta1 = 2 * np.pi * (Fr1 / Fs)
    theta2 = 2 * np.pi * (Fr2 / Fs)
    # Zero locations on the Z-plane
    z1 = np.exp(1j * theta1)  # z = 1 (DC)
    z2 = np.exp(1j * theta2)  # z = -1 (Nyquist)

    Fp = 125   # Passband center frequency (Hz)
    # Angle corresponding to the pole
    thetap = 2 * np.pi * (Fp / Fs)
    # Pole magnitude (r) based on 3dB bandwidth approximation
    r = 1 - (bw / Fs) * np.pi
    # Pole locations (complex conjugate pair)
    p1 = r * np.exp(1j * thetap)
    p2 = r * np.exp(-1j * thetap)
    # Collect all poles and zeros
    all_zeros = np.array([z1, z2])
    all_poles = np.array([p1, p2])
    b = np.poly(all_zeros).real   # Numerator coefficients
    a = np.poly(all_poles).real   # Denominator coefficients
    print(f"Filter Numerator (b): {b}")
    print(f"Filter Denominator (a): {a}")
    # Calculate the response. worN=256 is the number of points.
    w, h = signal.freqz(b, a, worN=256, fs=Fs)

    plt.figure(1, figsize=(7, 6))
    # Subplot 1: Magnitude Response
    plt.subplot(2, 1, 1)
    # Normalize magnitude by the peak value
    plt.plot(w, np.abs(h) / np.max(np.abs(h)), linewidth=2)
    plt.title('Magnitude Response (Normalized)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Magnitude')
    plt.grid(True)
    # Subplot 2: Phase Response
    plt.subplot(2, 1, 2)
    # Convert phase from radians to degrees
    plt.plot(w, np.angle(h) * 180 / np.pi, color='green', linewidth=2)
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid(True)
    plt.tight_layout()
    # Create the discrete-time transfer function object
    sys = ct.tf(b, a, dt=1 / Fs)
    plt.figure(2, figsize=(4, 4))
    # Use pzmap to automatically generate the pole-zero plot
    ct.pzmap(sys, title=False, marker_size=8)


# ---------------------------------------------------------------
# 6.6.1  Example 1: Design a high pass filter from low pass prototype
#        filter using Bilinear Transformation. Fs=150Hz, fc=30Hz.
# ---------------------------------------------------------------
def lab6_6_6_1():
    Fc = 30    # Cut-off frequency (Hz)
    Fs = 150   # Sampling frequency (Hz)
    Ts = 1 / Fs  # Sampling period (s)

    wd = 2 * np.pi * Fc          # Pre-warping Calculation
    wa = 2 / Ts * np.tan(wd * Ts / 2)  # Pre-warped analog frequency

    print(f"Sampling Frequency (Fs): {Fs} Hz")
    print(f"Digital Cut-off Frequency (Fc): {Fc} Hz")
    print(f"Pre-warped Analog Cut-off (wa): {wa:.2f} rad/s")

    num_lp = [1]     # Analog Low-Pass Prototype
    den_lp = [1, 1]
    # Low-Pass to High-Pass Transformation (Analog Domain)
    num_hp_a, den_hp_a = signal.lp2hp(num_lp, den_lp, wa)
    # Bilinear Transformation (Analog to Digital)
    a, b = signal.bilinear(num_hp_a, den_hp_a, Fs)
    N = 512  # Number of FFT points
    f, hz = signal.freqz(a, b, worN=N, fs=Fs)
    # --- Phase Response Processing ---
    phi_rad = np.unwrap(np.angle(hz))
    phi_deg = np.degrees(phi_rad)

    plt.subplot(2, 1, 1)   # Subplot 1: Magnitude Response
    plt.plot(f, np.abs(hz), linewidth=2)
    plt.ylabel("Magnitude Response")
    plt.xlim(0, Fs / 2)
    plt.grid(True)

    plt.subplot(2, 1, 2)   # Subplot 2: Phase Response
    plt.plot(f, phi_deg, color='orange', linewidth=2)
    plt.ylabel("Phase Response (Degrees)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, Fs / 2)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\nDigital Filter Numerator: {a}")
    print(f"Digital Filter Denominator: {b}")


# ---------------------------------------------------------------
# 6.6.2  Example 2: Design a Butterworth Low-Pass Digital Filter
#        using Bilinear Transformation
# ---------------------------------------------------------------
def lab6_6_6_2():
    F_sample = 20000  # Sampling frequency (Hz)
    F_pass = 2000     # Passband edge frequency (2 kHz)
    F_stop = 5000     # Stopband edge frequency (5 kHz)
    Ap = 3            # Maximum passband attenuation (dB)
    As = 20           # Minimum stopband attenuation (dB)
    T = 1.0 / F_sample

    omega_pa = (2.0 / T) * np.tan(np.pi * F_pass / F_sample)  # Pre-warped passband freq.
    omega_sa = (2.0 / T) * np.tan(np.pi * F_stop / F_sample)  # Pre-warped stopband freq.

    print(f"Sampling Frequency (Fs): {F_sample / 1000:.0f} kHz")
    print(f"Pre-warped Analog Passband Omega_pa: {omega_pa:.2f} rad/s")
    print(f"Pre-warped Analog Stopband Omega_sa: {omega_sa:.2f} rad/s")
    print("-" * 30)

    # --- Determine Analog Butterworth Filter Order (N) ---
    N, omega_c = signal.buttord(omega_pa, omega_sa, Ap, As, analog=True)

    print(f"Selected Filter Order (N): {N}")
    print(f"Analog Cutoff Frequency (omega_c): {omega_c:.2f} rad/s")
    print("-" * 30)
    # --- Design the Digital Filter using Scipy (BLT is applied internally) ---
    b, a = signal.butter(
        N,                  # Filter Order
        F_pass,             # Cutoff Frequency in Hz
        btype='lowpass',    # Low-pass filter
        analog=False,       # Design a digital filter
        output='ba',        # Return num (b) and den (a) coefficients
        fs=F_sample         # Provide sampling frequency for BLT pre-warping
    )

    print("Digital Filter Coefficients:")
    print(f"Numerator (b): {b}")
    print(f"Denominator (a): {a}")
    print("-" * 30)
    # Calculate the frequency response of the digital filter
    w, h = signal.freqz(b, a, fs=F_sample)

    magnitude_db = 20 * np.log10(abs(h))  # Convert magnitude to dB
    f_hz = w  # w is in Hz because we passed fs=F_sample to freqz

    plt.plot(f_hz / 1000, magnitude_db)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(linestyle='--')
    plt.xlim(0, F_sample / 2000)  # Plot up to Nyquist frequency (10 kHz)
    plt.ylim(-40, 5)
    # Passband edge (2 kHz, -3 dB)
    plt.plot(F_pass / 1000, -Ap, 'go',
             label=f'Passband Edge ({F_pass / 1000:.1f} kHz, -{Ap} dB)')
    plt.axvline(F_pass / 1000, color='g', linestyle='--')
    plt.axhline(-Ap, color='g', linestyle='--')
    # Stopband edge (5 kHz, -20 dB)
    plt.plot(F_stop / 1000, -As, 'ro',
             label=f'Stopband Edge ({F_stop / 1000:.1f} kHz, -{As} dB)')
    plt.axvline(F_stop / 1000, color='r', linestyle='--')
    plt.axhline(-As, color='r', linestyle='--')

    plt.legend()
    plt.show()


# ---------------------------------------------------------------
# 6.6.3  Example 3: Design a Chebyshev Type II Digital Bandstop Filter
#        using Bilinear Transformation
# ---------------------------------------------------------------
def lab6_6_6_3():
    Fs = 8000    # Sampling frequency (Hz)
    Ap = 3       # Maximum passband loss/ripple (dB)
    As = 40      # Minimum stopband attenuation/ripple (dB)
    # Passband edges (frequencies to KEEP)
    Fp1 = 1000   # 1.0 kHz
    Fp2 = 3000   # 3.0 kHz
    # Stopband edges (frequencies to REJECT)
    Fs1 = 1500   # 1.5 kHz
    Fs2 = 2500   # 2.5 kHz

    Nyquist = Fs / 2.0  # Normalize Frequencies for SciPy
    Wp = [Fp1 / Nyquist, Fp2 / Nyquist]  # Normalized Passband Edges (Wp)
    Ws = [Fs1 / Nyquist, Fs2 / Nyquist]  # Normalized Stopband Edges (Ws)

    print(f"Sampling Frequency (Fs): {Fs / 1000:.1f} kHz")
    print(f"Passband Loss (Ap): {Ap} dB (Monotonic Passband)")
    print(f"Stopband Attenuation (As): {As} dB (Equiripple Stopband)")
    print(f"Normalized Passband Edges (Wp): [{Wp[0]:.3f}, {Wp[1]:.3f}]*Nyquist")
    print(f"Normalized Stopband Edges (Ws): [{Ws[0]:.3f}, {Ws[1]:.3f}]*Nyquist")
    print("-" * 40)

    # --- Determine Filter Order (N) and Natural Frequencies (Wn) ---
    # It uses the passband loss (gpass) and the stopband attenuation (gstop).
    N, Wn = signal.cheb2ord(
        wp=Wp,          # Passband normalized frequencies
        ws=Ws,          # Stopband normalized frequencies
        gpass=Ap,       # Maximum loss in the passband (Ap)
        gstop=As,       # Minimum attenuation in the stopband (As)
        analog=False    # Design a digital filter
    )

    print(f"Required Filter Order (N): {N}")
    print(f"Chebyshev Natural Frequencies (Wn): "
          f"[{Wn[0]:.3f}, {Wn[1]:.3f}]*Nyquist")
    print("-" * 40)

    # --- Design the Digital Filter Coefficients using cheby2 ---
    # cheby2 computes the filter coefficients (b, a).
    # It uses the Stopband Attenuation (rs) to define the equiripple floor.
    b, a = signal.cheby2(
        N=N,                # Filter Order
        rs=As,              # Stopband ripple/rejection (As)
        Wn=Wn,              # Critical frequencies
        btype='bandstop',
        analog=False,       # Design a digital filter
        output='ba'         # Return numerator (b) and denominator (a) coefficients
    )

    print("Digital Filter Coefficients:")
    print(f"Numerator (b): {b}")
    print(f"Denominator (a): {a}")
    print("-" * 40)
    # Calculate the frequency response
    w, h = signal.freqz(b, a, fs=Fs)

    # Convert magnitude to dB
    magnitude_db = 20 * np.log10(np.abs(h))
    frequencies_hz = w

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_hz / 1000, magnitude_db,
             label='Chebyshev Type II Response (Equiripple Stopband)')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(which='both', axis='both', linestyle='--')

    plt.xlim(0, Nyquist / 1000)  # Set the x-axis limit up to the Nyquist frequency
    plt.ylim(-60, 5)  # Set a lower limit to clearly show 40 dB attenuation

    # Highlight Passband Edge (Ap)
    # The response must be at least -Ap at the passband edge.
    plt.axhline(-Ap, color='g', linestyle='--',
                label=f'Max Passband Loss (-{Ap} dB)')

    # Highlight Stopband Attenuation (As) - this is the equiripple floor
    plt.axhline(-As, color='r', linestyle='--',
                label=f'Equiripple Stopband Floor (-{As} dB)')
    plt.fill_between([Fs1 / 1000, Fs2 / 1000], -60, -As, color='pink',
                      label='Equiripple Stopband Region')

    # Highlight Passband Edges
    plt.axvline(Fp1 / 1000, color='b', linestyle=':')
    plt.axvline(Fp2 / 1000, color='b', linestyle=':', label='Passband Edges')

    plt.legend()
    plt.show()


# ==================================================================
#           LABORATORY 6 : POST-LABORATORY PROBLEM SOLUTIONS
# ==================================================================

# 1. Notch filter (pole-zero placement): notch freq 50Hz, 3dB width +-5Hz,
#    Fs=500Hz. Show response, and output for x(t)=2cos(60*pi*t)+cos(100*pi*t)
#    sampled at 500Hz (note: cos(100*pi*t) has f=50Hz, the notch frequency).
def lab6_post_1():
    Fs = 500
    f0 = 50
    bw = 10   # total 3dB width = 2*5Hz
    theta0 = 2 * np.pi * (f0 / Fs)
    z1 = np.exp(1j * theta0)
    z2 = np.exp(-1j * theta0)
    r = 1 - (bw / Fs) * np.pi
    p1 = r * np.exp(1j * theta0)
    p2 = r * np.exp(-1j * theta0)

    b = np.poly([z1, z2]).real
    a = np.poly([p1, p2]).real
    print(f"Filter Numerator (b): {b}")
    print(f"Filter Denominator (a): {a}")

    w, h = signal.freqz(b, a, worN=512, fs=Fs)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response of Notch Filter (f0 = 50 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h, deg=True), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.grid()
    plt.tight_layout()

    n = np.arange(0, 500)
    t = n / Fs
    x = 2 * np.cos(60 * np.pi * t) + np.cos(100 * np.pi * t)
    # x(t) has components at f = 30 Hz and f = 50 Hz; the notch filter
    # should reject the 50 Hz component.
    y = signal.lfilter(b, a, x)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title('Input x[n] = 2cos(60\u03c0t) + cos(100\u03c0t)')
    plt.xlabel('t (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.title('Output y[n] after Notch Filter (50 Hz removed)')
    plt.xlabel('t (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.tight_layout()


# 2. Bandstop filter (pole-zero placement), Omega0 = pi/10 (complete
#    attenuation), Omega_w = 2*Omega_cf = pi/20 (bandstop width).
def lab6_post_2():
    omega0 = np.pi / 10
    omega_w = np.pi / 20
    r = 1 - (omega_w / 2)
    z1 = np.exp(1j * omega0)
    z2 = np.exp(-1j * omega0)
    p1 = r * np.exp(1j * omega0)
    p2 = r * np.exp(-1j * omega0)

    b = np.poly([z1, z2]).real
    a = np.poly([p1, p2]).real
    print(f"Filter Numerator (b): {b}")
    print(f"Filter Denominator (a): {a}")

    w, h = signal.freqz(b, a, worN=512)
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Bandstop Filter (Pole-Zero Placement)')
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.tight_layout()


# 3. Bandpass filter from lowpass prototype H(s)=1/(s+1) using BLT.
#    Passband 200-300Hz, Fs=2kHz, filter order N=2.
def lab6_post_3():
    Fs = 2000
    Ts = 1 / Fs
    Fp1, Fp2 = 200, 300
    wd1 = 2 * np.pi * Fp1
    wd2 = 2 * np.pi * Fp2
    wa1 = 2 / Ts * np.tan(wd1 * Ts / 2)
    wa2 = 2 / Ts * np.tan(wd2 * Ts / 2)
    w0 = np.sqrt(wa1 * wa2)
    bw = wa2 - wa1

    num_lp = [1]     # Analog Low-Pass Prototype H(s) = 1/(s+1)
    den_lp = [1, 1]
    num_bp_a, den_bp_a = signal.lp2bp(num_lp, den_lp, w0, bw)
    b, a = signal.bilinear(num_bp_a, den_bp_a, Fs)
    print(f"Digital Filter Numerator (b): {b}")
    print(f"Digital Filter Denominator (a): {a}")

    w, h = signal.freqz(b, a, worN=512, fs=Fs)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Bandpass Filter from H(s)=1/(s+1)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h, deg=True), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid()
    plt.tight_layout()


# 4. Butterworth digital bandpass: passband 1500-2500Hz, stopband
#    1000-3000Hz, Fs=8kHz, passband ripple 1dB, stopband atten 30dB, BLT.
def lab6_post_4():
    Fs = 8000
    Ap = 1
    As = 30
    Fp = [1500, 2500]
    Fst = [1000, 3000]
    Nyq = Fs / 2
    Wp = [f / Nyq for f in Fp]
    Ws = [f / Nyq for f in Fst]

    N, Wn = signal.buttord(Wp, Ws, Ap, As, analog=False)
    print(f"Filter Order (N): {N}")
    print(f"Natural Frequencies (Wn): {Wn}")

    b, a = signal.butter(N, Wn, btype='bandpass', analog=False,
                          output='ba', fs=Fs)
    print(f"Numerator (b): {b}")
    print(f"Denominator (a): {a}")

    w, h = signal.freqz(b, a, worN=512, fs=Fs)
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Butterworth Bandpass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.tight_layout()


# 5. Chebyshev Type I highpass: passband 2500Hz, stopband 1500Hz,
#    Fs=8kHz, passband ripple 3dB, stopband atten 40dB, BLT.
def lab6_post_5():
    Fs = 8000
    Ap = 3
    As = 40
    Fp = 2500
    Fst = 1500
    Nyq = Fs / 2
    Wp = Fp / Nyq
    Ws = Fst / Nyq

    N, Wn = signal.cheb1ord(Wp, Ws, Ap, As, analog=False)
    print(f"Filter Order (N): {N}")
    print(f"Natural Frequency (Wn): {Wn}")

    b, a = signal.cheby1(N, Ap, Wn, btype='highpass', analog=False,
                          output='ba', fs=Fs)
    print(f"Numerator (b): {b}")
    print(f"Denominator (a): {a}")

    w, h = signal.freqz(b, a, worN=512, fs=Fs)
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Chebyshev Type I Highpass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.tight_layout()


# 6. Chebyshev Type II bandpass: passband 1500-2500Hz (3dB loss), stopband
#    below 1000Hz and above 3000Hz (40dB attenuation), Fs=8kHz, BLT.
def lab6_post_6():
    Fs = 8000
    Ap = 3
    As = 40
    Fp1, Fp2 = 1500, 2500
    Fs1, Fs2 = 1000, 3000
    Nyq = Fs / 2
    Wp = [Fp1 / Nyq, Fp2 / Nyq]
    Ws = [Fs1 / Nyq, Fs2 / Nyq]

    N, Wn = signal.cheb2ord(Wp, Ws, Ap, As, analog=False)
    print(f"Filter Order (N): {N}")
    print(f"Natural Frequencies (Wn): {Wn}")

    b, a = signal.cheby2(N, As, Wn, btype='bandpass', analog=False,
                          output='ba', fs=Fs)
    print(f"Numerator (b): {b}")
    print(f"Denominator (a): {a}")

    w, h = signal.freqz(b, a, worN=512, fs=Fs)
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Chebyshev Type II Bandpass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()
    plt.tight_layout()


# 7. Chebyshev Type I bandpass, order N=3, passband 1500-3000Hz.
#    (Fs and passband ripple are not specified in the problem; Fs=8kHz
#    and Ap=1dB are assumed, consistent with the other Lab 6 problems.)
def lab6_post_7():
    N = 3
    Fs = 8000
    Ap = 1
    Wn = [1500 / (Fs / 2), 3000 / (Fs / 2)]
    b, a = signal.cheby1(N, Ap, Wn, btype='bandpass', analog=False,
                          output='ba', fs=Fs)
    w, h = signal.freqz(b, a, worN=512, fs=Fs)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Chebyshev Type I Bandpass (N=3)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h, deg=True), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid()
    plt.tight_layout()


# 8. Chebyshev Type I band-reject, order N=3, stopband 1500-3000Hz.
#    (Fs and passband ripple assumed same as Problem 7.)
def lab6_post_8():
    N = 3
    Fs = 8000
    Ap = 1
    Wn = [1500 / (Fs / 2), 3000 / (Fs / 2)]
    b, a = signal.cheby1(N, Ap, Wn, btype='bandstop', analog=False,
                          output='ba', fs=Fs)
    w, h = signal.freqz(b, a, worN=512, fs=Fs)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response - Chebyshev Type I Band-Reject (N=3)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(w, np.angle(h, deg=True), color='green')
    plt.title('Phase Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.grid()
    plt.tight_layout()


# Problem 9 of Laboratory 6 asks to solve textbook exercise problems from
# specific page ranges; not included here (external textbook required).


# ==================================================================
#                              MAIN
# ==================================================================
if __name__ == "__main__":
    # ---- Laboratory 4 example codes ----
    lab4_4_2_1()
    lab4_4_2_2()
    lab4_4_2_3()
    lab4_4_3_1()
    lab4_4_3_2()
    lab4_4_3_3()
    lab4_4_3_4()
    lab4_4_4_1()   # requires lcapy
    lab4_4_4_2()   # requires lcapy
    lab4_4_4_3()   # requires lcapy
    lab4_4_4_4()
    lab4_4_5_1()
    lab4_4_5_2()
    lab4_4_5_3()
    lab4_4_5_4()

    # ---- Laboratory 4 post-lab problems ----
    lab4_post_1()
    lab4_post_2()
    lab4_post_3()
    lab4_post_4()
    lab4_post_5()
    lab4_post_6()
    lab4_post_7()

    # ---- Laboratory 5 example codes ----
    lab5_5_5_1()
    lab5_5_5_2()
    lab5_5_5_3()
    lab5_5_6_example1()
    lab5_5_6_example2()

    # ---- Laboratory 5 post-lab problems ----
    lab5_post_1()
    lab5_post_2()
    lab5_post_3()
    lab5_post_4()
    lab5_post_5()
    lab5_post_6()

    # ---- Laboratory 6 example codes ----
    lab6_6_5()
    lab6_6_6_1()
    lab6_6_6_2()
    lab6_6_6_3()

    # ---- Laboratory 6 post-lab problems ----
    lab6_post_1()
    lab6_post_2()
    lab6_post_3()
    lab6_post_4()
    lab6_post_5()
    lab6_post_6()
    lab6_post_7()
    lab6_post_8()

    plt.show()
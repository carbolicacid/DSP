# =============================================================================
#  EEE 3218 DSP LAB -- EXPERIMENTS 4, 5, 6  +  ALL POST-LAB PROBLEMS
#  One big copy-paste reference. Sections are marked with banners like this.
#  NOTE: a few numbers in the post-lab problems were blurry/unclear in the
#  manual scan -- those lines are marked "!! CHECK MANUAL !!" so you know
#  exactly which ones to double-check, everything else is solid.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import dft
import control as ct


# #############################################################################
# ############################   EXPERIMENT 4   ##############################
# ####################  Transforms & Discrete-Time Systems  ##################
# #############################################################################

# ---- 4.1  DFT via the DFT matrix ------------------------------------------
N = 5
x = np.array([1, 1, 0, 0, 1], dtype=float)

W = dft(N)                       # build the N x N DFT matrix
X = W @ x                        # forward DFT (matrix multiply)
k = np.arange(N)
MagX = np.abs(X)
PhaseX = np.angle(X)
x_rec = np.real((np.conjugate(W) @ X) / N)   # inverse DFT

plt.figure()
plt.subplot(3, 1, 1); plt.stem(k, MagX);   plt.title('DFT Magnitude')
plt.subplot(3, 1, 2); plt.stem(k, PhaseX); plt.title('DFT Phase')
plt.subplot(3, 1, 3); plt.stem(np.arange(N), x_rec); plt.title('Reconstructed x[n] (IDFT)')
plt.tight_layout(); plt.show()


# ---- 4.2  FFT ---------------------------------------------------------------
t0 = 0.2
ts = 8.3333e-4
t = np.arange(-t0/2, t0/2 + ts, ts)
x_sinc = np.sinc(100 * t)

Nfft = 256
Xf = np.fft.fft(x_sinc, Nfft)
freq = np.fft.fftfreq(Nfft, d=ts)

plt.figure()
plt.subplot(2, 1, 1); plt.plot(t, x_sinc); plt.title('Sinc pulse (time domain)')
plt.subplot(2, 1, 2)
plt.stem(np.fft.fftshift(freq), np.abs(np.fft.fftshift(Xf)) / Nfft)
plt.title('Magnitude Spectrum'); plt.xlabel('Frequency (Hz)')
plt.tight_layout(); plt.show()


# ---- 4.3  Convolution via DFT/FFT (zero-pad first!) -------------------------
def dft_convolution(x, h):
    L = len(x) + len(h) - 1
    X = np.fft.fft(x, L)
    H = np.fft.fft(h, L)
    y = np.real(np.fft.ifft(X * H))
    return y

x_ex = np.array([1, 2, 3, 4])
h_ex = np.array([1, 1, 1])
y_dft_conv = dft_convolution(x_ex, h_ex)
y_direct = np.convolve(x_ex, h_ex)
print("DFT-based convolution:", y_dft_conv)
print("np.convolve (should match):", y_direct)


# ---- 4.4  Modulation / Demodulation -----------------------------------------
Fs_mod = 5000
t_mod = np.arange(0, 0.02, 1/Fs_mod)
message = np.sin(2*np.pi*50*t_mod)          # a 50Hz message signal
carrier = np.cos(2*np.pi*250*t_mod)         # 250Hz carrier

modulated = message * carrier                # shifts spectrum to +-250Hz
demodulated = modulated * carrier             # shifts back down (+ junk at 500Hz)

# low-pass filter to remove the junk (zero out high FFT bins)
Dm = np.fft.fft(demodulated)
freqs_mod = np.fft.fftfreq(len(t_mod), d=1/Fs_mod)
Dm[np.abs(freqs_mod) > 100] = 0
recovered = np.real(np.fft.ifft(Dm))

plt.figure()
plt.subplot(3,1,1); plt.plot(t_mod, message); plt.title('Original Message')
plt.subplot(3,1,2); plt.plot(t_mod, modulated); plt.title('Modulated Signal')
plt.subplot(3,1,3); plt.plot(t_mod, recovered); plt.title('Recovered (Demodulated + LPF)')
plt.tight_layout(); plt.show()


# ---- 4.5  Z-transform: partial fraction expansion ---------------------------
num_z = [1, 2, -1]
den_z = [1, -1, 0.3561]
r, p, kdir = signal.residuez(num_z, den_z)
print("Residues:", r, " Poles:", p, " Direct terms:", kdir)
num_back, den_back = signal.invresz(r, p, kdir)   # rebuild -> should match num_z/den_z


# ---- 4.6  Discrete-time system: the reusable 4-STEP RECIPE ------------------
def characterize_system(num, den, n_len=30, fs=None, title=""):
    """Impulse response, magnitude, pole-zero map, phase -- reuse everywhere."""
    x_imp = np.zeros(n_len); x_imp[0] = 1
    h = signal.lfilter(num, den, x_imp)
    w, H = signal.freqz(num, den, fs=fs) if fs is not None else signal.freqz(num, den)
    sys = ct.tf(num, den, dt=True)

    plt.figure(figsize=(9, 7))
    plt.subplot(2, 2, 1); plt.stem(h); plt.title('Impulse Response')
    plt.subplot(2, 2, 2); plt.plot(w, 20*np.log10(np.abs(H)+1e-12)); plt.title('Magnitude (dB)')
    plt.subplot(2, 2, 3); ct.pzmap(sys, title=False); plt.title('Pole-Zero Map')
    plt.subplot(2, 2, 4); plt.plot(w, np.angle(H)); plt.title('Phase (rad)')
    plt.suptitle(title)
    plt.tight_layout(); plt.show()
    return h, w, H

characterize_system([1, 2, -1], [1, -1, 0.3561], title="Example System")


# ---- 4.7  Notch filter example (frequency response, 50Hz notch) -------------
num_notch = [1, -1.6180, 1]
den_notch = [1, -1.5371, 0.9025]
Fs_notch = 512
w_n, h_n = signal.freqz(num_notch, den_notch, worN=256, fs=Fs_notch)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_n, 20*np.log10(np.abs(h_n))); plt.title('Notch Filter Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_n, np.angle(h_n, deg=True)); plt.title('Notch Filter Phase (deg)')
plt.tight_layout(); plt.show()


# =============================================================================
# EXPERIMENT 4 -- POST-LABORATORY PROBLEMS
# =============================================================================

# ---- P4.1: y[n]-4y[n-1]+4y[n-2] = x[n]-x[n-1] --------------------------------
# difference-equation form for lfilter: coefficients on y-side (den) first,
# then x-side (num). y[n]-4y[n-1]+4y[n-2] -> den=[1,-4,4]; x[n]-x[n-1] -> num=[1,-1]
num_p1 = [1, -1]
den_p1 = [1, -4, 4]

n_p1 = np.arange(-20, 26)
imp_in = (n_p1 == 0).astype(float)
h_p1 = signal.lfilter(num_p1, den_p1, imp_in)
plt.figure(); plt.stem(n_p1, h_p1); plt.title('P4.1: Impulse response, -20<=n<=25')
plt.xlabel('n'); plt.show()

# response to x[n] = (5 + 3cos(0.2*pi*n) + 4sin(0.6*pi*n)) * u[n]
n2 = np.arange(0, 50)
x_p1 = (5 + 3*np.cos(0.2*np.pi*n2) + 4*np.sin(0.6*np.pi*n2))
y_p1 = signal.lfilter(num_p1, den_p1, x_p1)
plt.figure(); plt.stem(n2, y_p1); plt.title('P4.1: Output for given x[n]')
plt.xlabel('n'); plt.show()


# ---- P4.2: frequency response of a given H(z)  !! CHECK MANUAL !! -----------
# manual scan was unclear on the exact H(z) here -- plug in your num/den
# from your own copy, template below works as-is once you do:
num_p2 = [2]          # <- replace with your actual numerator
den_p2 = [1, -0.5]    # <- replace with your actual denominator (placeholder)
w_p2, H_p2 = signal.freqz(num_p2, den_p2)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_p2, 20*np.log10(np.abs(H_p2)+1e-12)); plt.title('P4.2 Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_p2, np.angle(H_p2)); plt.title('P4.2 Phase (rad)')
plt.tight_layout(); plt.show()


# ---- P4.3: H(z) = 1 / [(z-1)(z-0.5)(z-0.2)]  --  pole-zero + stability -------
poles_p3 = [1, 0.5, 0.2]
den_p3 = np.poly(poles_p3).real     # denominator from the poles
num_p3 = [1]                        # numerator = 1 (no zeros)
sys_p3 = ct.tf(num_p3, den_p3, dt=True)
plt.figure(); ct.pzmap(sys_p3); plt.title('P4.3: Pole-Zero Map')
plt.show()
print("P4.3 poles:", np.roots(den_p3), " -> pole at z=1 sits ON the unit circle:")
print("   => system is only MARGINALLY stable (not strictly stable).")


# ---- P4.4: partial fraction expansion  !! CHECK MANUAL !! -------------------
# the exact polynomial coefficients were unclear in the scan -- replace these
# with your manual's actual num/den, the residuez call itself is correct as-is
num_p4 = [2, 16, 44, 56]     # <- replace with your actual numerator coefficients
den_p4 = [3, 18, 1, 82]      # <- replace with your actual denominator coefficients (placeholder)
r4, p4, k4 = signal.residuez(num_p4, den_p4)
print("P4.4 residues:", r4, " poles:", p4, " direct terms:", k4)


# ---- P4.5: sample x(t)=sinc^2(100t) at 5kHz, find DFT spectrum --------------
Fs_p5 = 5000
t_p5 = np.arange(-0.1, 0.1, 1/Fs_p5)
x_p5 = np.sinc(100*t_p5)**2
Xp5 = np.fft.fft(x_p5)
freq_p5 = np.fft.fftfreq(len(t_p5), d=1/Fs_p5)
plt.figure()
plt.subplot(2,1,1); plt.plot(t_p5, x_p5); plt.title('P4.5: sinc^2(100t)')
plt.subplot(2,1,2)
plt.plot(np.fft.fftshift(freq_p5), np.abs(np.fft.fftshift(Xp5)))
plt.title('P4.5: DFT Magnitude Spectrum'); plt.xlabel('Frequency (Hz)')
plt.tight_layout(); plt.show()


# ---- P4.6: autocorrelation of x[n]={1,1,2,3,5,8,13} via FFT -----------------
x_p6 = np.array([1, 1, 2, 3, 5, 8, 13], dtype=float)
L_p6 = 2*len(x_p6) - 1
Xp6 = np.fft.fft(x_p6, L_p6)
autocorr_p6 = np.real(np.fft.ifft(Xp6 * np.conj(Xp6)))
autocorr_p6 = np.fft.fftshift(autocorr_p6)
lags_p6 = np.arange(-(len(x_p6)-1), len(x_p6))
plt.figure()
plt.subplot(2,1,1); plt.stem(lags_p6, autocorr_p6); plt.title('P4.6: Autocorrelation (time domain)')
plt.subplot(2,1,2); plt.stem(np.abs(np.fft.fft(x_p6))); plt.title('P4.6: Spectrum of x[n]')
plt.tight_layout(); plt.show()


# ---- P4.7: FDM system -- modulate two messages onto two carriers, combine ---
t0_p7 = 0.1
Fs_p7 = 5000
t_p7 = np.arange(-t0_p7, t0_p7, 1/Fs_p7)
m1 = np.sinc(100*t_p7)
m2 = np.sinc(100*t_p7)**2
c1 = np.cos(2*np.pi*250*t_p7)     # fc1 = 250Hz
c2 = np.cos(2*np.pi*750*t_p7)     # fc2 = 750Hz

y_p7 = m1*c1 + m2*c2               # FDM combined output

def show_spectrum(sig, fs, title):
    Sf = np.fft.fftshift(np.fft.fft(sig))
    f_ax = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1/fs))
    plt.plot(f_ax, np.abs(Sf)); plt.title(title); plt.xlabel('Frequency (Hz)')

plt.figure(figsize=(8,7))
plt.subplot(3,1,1); show_spectrum(m1, Fs_p7, 'P4.7: |M1(f)|')
plt.subplot(3,1,2); show_spectrum(m2, Fs_p7, 'P4.7: |M2(f)|')
plt.subplot(3,1,3); show_spectrum(y_p7, Fs_p7, 'P4.7: |Y(f)| (FDM output)')
plt.tight_layout(); plt.show()

# P4.8, P4.9: textbook exercise problems -- worked by hand, no code needed.


# #############################################################################
# ############################   EXPERIMENT 5   ##############################
# ####################   Digital Filter Design (Part 1): FIR   ###############
# #############################################################################

# ---- 5.1  Characterize simple filters (reuse characterize_system from above)
characterize_system([1], [1, -1], title="First-order IIR: H(z)=1/(1-z^-1)")
characterize_system([1/3, 1/3, 1/3], [1], title="3-point Moving Average")

r_res = 0.9; fs_res = 100; fc_res = 10
w0_res = 2*np.pi*(fc_res/fs_res)
characterize_system([1], [1, -2*r_res*np.cos(w0_res), r_res**2], n_len=60,
                     title="Digital Resonator (r=0.9)")


# ---- 5.2  FIR Low-pass via windowing method (Blackman window) ---------------
Fs5 = 8000.0
TW5 = 500.0
PBE5 = 1500.0
M5 = int(np.round(6 * Fs5 / TW5))
Fc5 = PBE5 + TW5/2

fir_lp = signal.firwin(numtaps=M5+1, cutoff=Fc5, window='blackman',
                        pass_zero='lowpass', fs=Fs5)
f5, h5 = signal.freqz(fir_lp, 1, worN=512, fs=Fs5)
plt.figure()
plt.subplot(2,1,1); plt.plot(f5, 20*np.log10(np.abs(h5)+1e-12)); plt.title('FIR LP Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(f5, np.degrees(np.unwrap(np.angle(h5)))); plt.title('FIR LP Phase (linear)')
plt.tight_layout(); plt.show()


# ---- 5.3  FIR Band-pass via windowing (Hamming window) -----------------------
Fs5b = 100.0
F1, Fu = 10, 20          # passband edges (Hz)
numtaps5b = 101
fir_bp = signal.firwin(numtaps=numtaps5b, cutoff=[F1, Fu], window='hamming',
                        pass_zero='bandpass', fs=Fs5b)
f5b, h5b = signal.freqz(fir_bp, 1, worN=512, fs=Fs5b)
plt.figure()
plt.subplot(2,1,1)
plt.plot(f5b, 20*np.log10(np.abs(h5b)+1e-12))
plt.axvline(F1, color='red', ls='--', label=f'Lower cutoff ({F1} Hz)')
plt.axvline(Fu, color='purple', ls='--', label=f'Upper cutoff ({Fu} Hz)')
plt.axhline(-3, color='orange', ls='-.', label='-3 dB Mark')
plt.legend(fontsize=8); plt.title('FIR Bandpass Magnitude (dB)')
plt.subplot(2,1,2)
plt.plot(f5b, np.degrees(np.unwrap(np.angle(h5b))), color='green')
plt.title('FIR Bandpass Phase')
plt.tight_layout(); plt.show()


# =============================================================================
# EXPERIMENT 5 -- POST-LABORATORY PROBLEMS
# =============================================================================

# ---- P5.1: 50th-order Highpass, Fs=5kHz, cutoff=1kHz, every window ----------
Fs_51 = 5000
Fc_51 = 1000
order_51 = 50
windows_51 = ['boxcar', 'hamming', 'hann', 'blackman']

plt.figure(figsize=(9,7))
for i, win in enumerate(windows_51):
    h_51 = signal.firwin(numtaps=order_51+1, cutoff=Fc_51, window=win,
                          pass_zero='highpass', fs=Fs_51)
    w_51, H_51 = signal.freqz(h_51, 1, worN=512, fs=Fs_51)
    plt.subplot(2,2,i+1)
    plt.plot(w_51, 20*np.log10(np.abs(H_51)+1e-12))
    plt.title(f'P5.1: {win} window'); plt.xlabel('Hz')
plt.tight_layout(); plt.show()


# ---- P5.2: FIR bandpass, spec-driven window choice via Kaiser ---------------
Fs_52 = 1000
passband_52 = [150, 250]
trans_width_52 = 50
ripple_db_52 = 0.05
atten_db_52 = 50
nyq_52 = Fs_52/2

# Kaiser is the natural choice here because you can hit an EXACT ripple/
# attenuation spec by computing beta and the order directly:
atten_target = max(ripple_db_52, atten_db_52)     # Kaiser sizes off worst-case
N_52, beta_52 = signal.kaiserord(atten_target, trans_width_52/nyq_52)
if N_52 % 2 == 0: N_52 += 1   # ensure odd length for a bandpass FIR

fir_52 = signal.firwin(N_52, passband_52, window=('kaiser', beta_52),
                        pass_zero='bandpass', fs=Fs_52)
w_52, H_52 = signal.freqz(fir_52, 1, worN=512, fs=Fs_52)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_52, 20*np.log10(np.abs(H_52)+1e-12)); plt.title(f'P5.2: Kaiser (beta={beta_52:.2f}, N={N_52})')
plt.subplot(2,1,2); plt.plot(w_52, np.degrees(np.unwrap(np.angle(H_52)))); plt.title('P5.2 Phase')
plt.tight_layout(); plt.show()


# ---- P5.3: characterize 4 given H(z)  !! CHECK MANUAL for (a) and (b) !! ----
# (c) and (d) were legible in the scan:
characterize_system([1, -2*np.cos(np.pi/4), 1], [1], title="P5.3c: H(z)=1-2z^-1cos(w0)+z^-2")
a_p53 = 0.5   # example value, |a|<1 -- replace if manual specifies a different a
characterize_system([0, 1], [1, -a_p53], title="P5.3d: H(z) = z/(z-a)")
# (a), (b): scan was unclear -- once you have the exact H(z), just call:
#   characterize_system(num, den, title="P5.3a") / "P5.3b"


# ---- P5.4: Highpass FIR, cutoff=150Hz, Fs=1000Hz, order=60, Blackman --------
fir_54 = signal.firwin(numtaps=61, cutoff=150, window='blackman',
                        pass_zero='highpass', fs=1000)
w_54, H_54 = signal.freqz(fir_54, 1, worN=512, fs=1000)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_54, 20*np.log10(np.abs(H_54)+1e-12)); plt.title('P5.4: Highpass Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_54, np.degrees(np.unwrap(np.angle(H_54)))); plt.title('P5.4 Phase')
plt.tight_layout(); plt.show()


# ---- P5.5: Band-stop FIR, reject 45-55Hz, Fs=300Hz, order=120, Kaiser(8.0) --
fir_55 = signal.firwin(numtaps=121, cutoff=[45, 55], window=('kaiser', 8.0),
                        pass_zero='bandstop', fs=300)
w_55, H_55 = signal.freqz(fir_55, 1, worN=512, fs=300)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_55, 20*np.log10(np.abs(H_55)+1e-12)); plt.title('P5.5: Bandstop Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_55, np.degrees(np.unwrap(np.angle(H_55)))); plt.title('P5.5 Phase')
plt.tight_layout(); plt.show()


# ---- P5.6: Low-pass FIR, cutoff=20kHz, Fs=100kHz, order=40, Rectangular -----
# (this one is DESIGNED to show the Gibbs phenomenon -- expect visible ripple)
fir_56 = signal.firwin(numtaps=41, cutoff=20000, window='boxcar',
                        pass_zero='lowpass', fs=100000)
w_56, H_56 = signal.freqz(fir_56, 1, worN=512, fs=100000)
plt.figure()
plt.plot(w_56, 20*np.log10(np.abs(H_56)+1e-12))
plt.title('P5.6: Rectangular window -- notice the Gibbs ripple')
plt.xlabel('Hz'); plt.show()

# P5.7, P5.8: textbook exercise problems -- worked by hand, no code needed.


# #############################################################################
# ############################   EXPERIMENT 6   ##############################
# ####################   Digital Filter Design (Part 2): IIR   ###############
# #############################################################################

# ---- 6.1  Pole-zero placement (bandpass/notch design) -----------------------
def polezero_design(Fs, zero_freqs, pole_freq, bw):
    """zero_freqs: list of frequencies (Hz) to completely reject.
       pole_freq : frequency (Hz) to boost (passband center).
       bw        : desired 3dB bandwidth (Hz)."""
    zeros = [np.exp(1j*2*np.pi*fz/Fs) for fz in zero_freqs]
    thetap = 2*np.pi*(pole_freq/Fs)
    r = 1 - (bw/Fs)*np.pi
    poles = [r*np.exp(1j*thetap), r*np.exp(-1j*thetap)]
    b = np.poly(zeros).real
    a = np.poly(poles).real
    return b, a

b61, a61 = polezero_design(Fs=500, zero_freqs=[0, 250], pole_freq=125, bw=10)
w61, H61 = signal.freqz(b61, a61, worN=256, fs=500)
plt.figure()
plt.subplot(2,1,1); plt.plot(w61, np.abs(H61)); plt.title('6.1: Pole-Zero Design Magnitude')
plt.subplot(2,1,2); plt.plot(w61, np.angle(H61, deg=True)); plt.title('6.1 Phase (deg)')
plt.tight_layout(); plt.show()


# ---- 6.2  Butterworth (bilinear transform) -----------------------------------
def design_butterworth(Fpass, Fstop, Ap, As, Fs, btype='lowpass'):
    b, a = signal.iirdesign(Fpass, Fstop, Ap, As, fs=Fs, ftype='butter')
    return b, a

b_bw, a_bw = design_butterworth(2000, 5000, 3, 20, 20000, btype='lowpass')
w_bw, H_bw = signal.freqz(b_bw, a_bw, fs=20000)
plt.figure(); plt.plot(w_bw, 20*np.log10(np.abs(H_bw)+1e-12))
plt.title('6.2: Butterworth Lowpass'); plt.xlabel('Hz'); plt.ylabel('dB'); plt.show()


# ---- 6.3  Chebyshev Type II (bilinear transform) -----------------------------
Fs63 = 8000; Ap63 = 3; As63 = 40
Wp63 = [1000/(Fs63/2), 3000/(Fs63/2)]
Ws63 = [1500/(Fs63/2), 2500/(Fs63/2)]
N63, Wn63 = signal.cheb2ord(Wp63, Ws63, Ap63, As63)
b63, a63 = signal.cheby2(N63, As63, Wn63, btype='bandstop')
w63, H63 = signal.freqz(b63, a63, fs=Fs63)
plt.figure(); plt.plot(w63, 20*np.log10(np.abs(H63)+1e-12))
plt.title(f'6.3: Chebyshev II Bandstop (N={N63})'); plt.xlabel('Hz'); plt.show()


# ---- 6.4  Bilinear transform FROM an analog prototype directly --------------
# use this whenever the problem GIVES you an analog H(s) directly
def bilinear_from_analog(num_s, den_s, fs):
    return signal.bilinear(num_s, den_s, fs=fs)

# example: analog prototype H(s) = 1/(s+1)
b_bl, a_bl = bilinear_from_analog([1], [1, 1], fs=2000)
w_bl, H_bl = signal.freqz(b_bl, a_bl, fs=2000)
plt.figure(); plt.plot(w_bl, 20*np.log10(np.abs(H_bl)+1e-12))
plt.title('6.4: Digital filter via bilinear transform of H(s)=1/(s+1)')
plt.xlabel('Hz'); plt.show()


# =============================================================================
# EXPERIMENT 6 -- POST-LABORATORY PROBLEMS
# =============================================================================

# ---- P6.1: Notch filter, f0=50Hz, 3dB width=+-5Hz, Fs=500Hz -----------------
b_p61, a_p61 = polezero_design(Fs=500, zero_freqs=[50], pole_freq=50, bw=10)
# (zero AND pole both at 50Hz -- classic notch: zero kills it, nearby pole keeps
#  everything else flat. bw=10 gives roughly +-5Hz width around the notch)
w_p61, H_p61 = signal.freqz(b_p61, a_p61, worN=256, fs=500)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_p61, 20*np.log10(np.abs(H_p61)+1e-12)); plt.title('P6.1: Notch Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_p61, np.angle(H_p61, deg=True)); plt.title('P6.1 Phase (deg)')
plt.tight_layout(); plt.show()

# test signal: x(t) = 2cos(60t) + cos(100t),  !! CHECK MANUAL for exact freq scaling !!
Fs_p61 = 500
t_p61 = np.arange(0, 1, 1/Fs_p61)
x_p61 = 2*np.cos(60*t_p61) + np.cos(100*t_p61)
y_p61 = signal.lfilter(b_p61, a_p61, x_p61)
plt.figure()
plt.subplot(2,1,1); plt.plot(t_p61, x_p61); plt.title('P6.1: Input signal x[n]')
plt.subplot(2,1,2); plt.plot(t_p61, y_p61); plt.title('P6.1: Output after notch filter')
plt.tight_layout(); plt.show()


# ---- P6.2: Bandstop via pole-zero, center freq pi/2, width pi ---------------
# here frequencies are given directly as ANGULAR (radians), not Hz --
# same math, just skip the Hz->radian conversion since it's already in radians
theta0_p62 = np.pi/2
bw_rad_p62 = np.pi
zero_p62 = np.exp(1j*theta0_p62)
r_p62 = 1 - bw_rad_p62/(2*np.pi)   # crude radius estimate from bandwidth
pole1_p62 = r_p62*np.exp(1j*theta0_p62)
pole2_p62 = r_p62*np.exp(-1j*theta0_p62)
b_p62 = np.poly([zero_p62, np.conj(zero_p62)]).real
a_p62 = np.poly([pole1_p62, pole2_p62]).real
print("P6.2 filter coefficients: b =", b_p62, " a =", a_p62)
w_p62, H_p62 = signal.freqz(b_p62, a_p62, worN=256)
plt.figure(); plt.plot(w_p62, 20*np.log10(np.abs(H_p62)+1e-12))
plt.title('P6.2: Bandstop Frequency Response'); plt.show()


# ---- P6.3: Bandpass via bilinear transform of analog LP prototype -----------
# H(s) prototype given in manual -- !! CHECK MANUAL for exact H(s) !!
# using a generic 1st-order lowpass prototype as placeholder:
num_s_p63 = [1]
den_s_p63 = [1, 1]     # H(s) = 1/(s+1)  <- replace with your manual's exact H(s)
Fs_p63 = 2000
wo_p63 = 2*np.pi*250        # center frequency (Hz -> rad/s), sqrt(200*300)~=245Hz
bw_p63 = 2*np.pi*100         # bandwidth in rad/s (300-200=100Hz)
b_lp, a_lp = num_s_p63, den_s_p63
b_bp_analog, a_bp_analog = signal.lp2bp(b_lp, a_lp, wo=wo_p63, bw=bw_p63)
b_p63, a_p63 = signal.bilinear(b_bp_analog, a_bp_analog, fs=Fs_p63)
w_p63, H_p63 = signal.freqz(b_p63, a_p63, fs=Fs_p63)
plt.figure(); plt.plot(w_p63, 20*np.log10(np.abs(H_p63)+1e-12))
plt.title('P6.3: Bandpass via Bilinear Transform'); plt.xlabel('Hz'); plt.show()
print("P6.3 filter coefficients: b =", b_p63, " a =", a_p63)


# ---- P6.4: Butterworth bandpass, BLT -----------------------------------------
b_p64, a_p64 = signal.iirdesign([1500, 2500], [1000, 3000], 1, 30, fs=8000, ftype='butter')
w_p64, H_p64 = signal.freqz(b_p64, a_p64, fs=8000)
plt.figure(); plt.plot(w_p64, 20*np.log10(np.abs(H_p64)+1e-12))
plt.title('P6.4: Butterworth Bandpass'); plt.xlabel('Hz'); plt.show()


# ---- P6.5: Chebyshev Type I highpass, BLT ------------------------------------
b_p65, a_p65 = signal.iirdesign(2500, 1500, 3, 40, fs=8000, ftype='cheby1')
w_p65, H_p65 = signal.freqz(b_p65, a_p65, fs=8000)
plt.figure(); plt.plot(w_p65, 20*np.log10(np.abs(H_p65)+1e-12))
plt.title('P6.5: Chebyshev I Highpass'); plt.xlabel('Hz'); plt.show()


# ---- P6.6: Chebyshev Type II bandpass, BLT -----------------------------------
b_p66, a_p66 = signal.iirdesign([1500, 2500], [1000, 3000], 3, 40, fs=8000, ftype='cheby2')
w_p66, H_p66 = signal.freqz(b_p66, a_p66, fs=8000)
plt.figure(); plt.plot(w_p66, 20*np.log10(np.abs(H_p66)+1e-12))
plt.title('P6.6: Chebyshev II Bandpass'); plt.xlabel('Hz'); plt.show()


# ---- P6.7: Chebyshev Type I bandpass, fixed order N=3 ------------------------
rp_p67 = 1   # ripple in dB -- manual didn't specify, adjust if given
b_p67, a_p67 = signal.cheby1(3, rp_p67, [1500, 3000], btype='bandpass', fs=8000)
w_p67, H_p67 = signal.freqz(b_p67, a_p67, fs=8000)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_p67, 20*np.log10(np.abs(H_p67)+1e-12)); plt.title('P6.7 Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_p67, np.angle(H_p67, deg=True)); plt.title('P6.7 Phase (deg)')
plt.tight_layout(); plt.show()


# ---- P6.8: Chebyshev Type I band-REJECT, fixed order N=3 ---------------------
rp_p68 = 1   # ripple in dB -- manual didn't specify, adjust if given
b_p68, a_p68 = signal.cheby1(3, rp_p68, [1500, 3000], btype='bandstop', fs=8000)
w_p68, H_p68 = signal.freqz(b_p68, a_p68, fs=8000)
plt.figure()
plt.subplot(2,1,1); plt.plot(w_p68, 20*np.log10(np.abs(H_p68)+1e-12)); plt.title('P6.8 Magnitude (dB)')
plt.subplot(2,1,2); plt.plot(w_p68, np.angle(H_p68, deg=True)); plt.title('P6.8 Phase (deg)')
plt.tight_layout(); plt.show()

# P6.9: textbook exercise problems -- worked by hand, no code needed.

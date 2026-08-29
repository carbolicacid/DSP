# ============================
# Step 1: Record voice and plot it
# ============================
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample
import matplotlib.pyplot as plt
import os

fs = 44100
duration = 10  # seconds

print("Recording... Speak now!")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Recording finished!")

recording = recording.flatten()

# Save original
write('original.wav', fs, (recording * 32767).astype(np.int16))

# Plot
time = np.linspace(0, duration, len(recording))
plt.figure(figsize=(12, 4))
plt.plot(time, recording)
plt.title("Original Voice Waveform (44.1 kHz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()



# ============================
# Step 2: Sample at 8 kHz
# ============================
num_samples_8k = int(duration * 8000)
recording_8k = resample(recording, num_samples_8k)

write('sampled_8k.wav', 8000, (recording_8k * 32767).astype(np.int16))

print("Original WAV size:", round(os.path.getsize("original.wav") / 1024, 2), "KB")
print("8kHz WAV size:   ", round(os.path.getsize("sampled_8k.wav") / 1024, 2), "KB")



# ============================
# Step 3: Quantization
# ============================
def quantize_signal(signal, bits):
    signal_norm = np.clip(signal, -1.0, 1.0)
    levels = 2 ** bits
    quantized = np.round(signal_norm * (levels - 1)) / (levels - 1)
    return quantized

original_norm = recording  # already flattened

for bits in [3, 4, 8, 10]:
    quant = quantize_signal(original_norm, bits)
    
    quant_int = (quant * 32767).astype(np.int16)
    write(f'quantized_{bits}bit.wav', fs, quant_int)
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(time[:len(quant)], original_norm[:len(quant)], label='Original', alpha=0.6)
    plt.plot(time[:len(quant)], quant, label=f'{bits}-bit Quantized', linewidth=1.2)
    plt.title(f'Quantization with {bits} bits')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Playing {bits}-bit version...")
    sd.play(quant_int, fs)
    sd.wait()
    
    

# ============================
# Step 4: Sinc Reconstruction
# ============================
def sinc_reconstruct(samples, upsample_factor=8):
    """Simple upsampling using scipy resample (approximates sinc)"""
    return resample(samples, len(samples) * upsample_factor)

for bits in [3, 4, 8, 10]:
    quant = quantize_signal(original_norm, bits)
    upsample_factor = 8
    reconstructed = sinc_reconstruct(quant)
    
    # Plot first 1000 samples for clarity
    n = 1000
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(quant[:n], label=f'{bits}-bit Quantized')
    plt.title(f'{bits}-bit Quantized Signal')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed[:n*upsample_factor], label='Sinc Reconstructed')
    plt.title('Sinc Interpolated Reconstruction')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Save reconstructed
    recon_int = (reconstructed * 32767).astype(np.int16)
    write(f'recon_{bits}bit.wav', fs * upsample_factor, recon_int[:len(recon_int)])
    
    
    
    
    
    
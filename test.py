import numpy as np
import time

from DFT import fourier_transform, inverse_fourier_transform
from FFT import fft, ifft
from STFT import stft, istft


# Generate a test signal
fs = 4096  # Sampling rate
second = 1
num_samples = fs * second  # Total number of samples
t = np.linspace(0, second, num_samples)  # Time vector

x = 0
for Hz in range(1, 10, 2):
    x += 1/Hz * np.sin( 2*np.pi * Hz*t )  # Input signal

# Compute the DFT of the test signal
start_time = time.time()
spectrum = fourier_transform(x)
x_hat = inverse_fourier_transform(spectrum)
print('DFT time: ', time.time() - start_time)

# Compute the FFT of the test signal
start_time = time.time()
spectrum = fft(x)
x_hat = ifft(spectrum)
print('FFT time: ', time.time() - start_time)

# Compute the np.fft of the test signal
start_time = time.time()
spectrum = np.fft.fft(x)
x_hat = np.fft.ifft(spectrum)
print('np.fft time: ', time.time() - start_time)

# print(np.allclose(fourier_transform(x), np.fft.fft(x)))
# print(np.allclose(fft(x), np.fft.fft(x)))

window_size = 128
fft_size = 128
hop_size = 32

# Compute the STFT of the test signal
start_time = time.time()
stft_matrix = stft(x, window_size, fft_size, hop_size)
x_hat = istft(stft_matrix, window_size, fft_size, hop_size)
print('STFT time: ', time.time() - start_time)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def stft(x, window_size, hop_size):
    window = np.hanning(window_size)
    num_frames = int(np.ceil(len(x) / hop_size))
    stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex128)

    for i in range(num_frames):
        start = i * hop_size
        end = min(start + window_size, len(x))

        frame = x[start:end] * window[:end-start]
        stft_matrix[:, i] = np.fft.fft(frame, window_size)

    return stft_matrix

def istft(stft_matrix, window_size, hop_size):
    num_samples = (stft_matrix.shape[1] - 1) * hop_size + window_size

    window = np.hanning(window_size)
    output = np.zeros(num_samples)
    window_sum = np.zeros(num_samples)

    for i in range(stft_matrix.shape[1]):
        start = i * hop_size
        end = start + window_size

        frame = np.real(np.fft.ifft(stft_matrix[:, i], window_size))

        output[start:end] += frame * window
        window_sum[start:end] += window

    # Normalize the output signal by dividing it by the sum of the window values
    output /= window_sum

    return output


def stft(x, fft_size=1024, hop_size=256):
    print('input signal', x.shape)
    w = np.hanning(fft_size)
    # w = signal.windows.hann(fft_size, sym=False)

    # Padding the signal so that frames are centered around their midpoint
    x = np.pad(x, int(fft_size // 2), mode='reflect')
    print('padding signal', x.shape)

    # Window the signal
    x_w = np.array([w * x[i:i + fft_size] for i in range(0, len(x) - fft_size, hop_size)])
    print('window signal', x_w.shape) # = math.ceil((len(x) - fft_size) / hop_size)

    # Compute the FFT
    X = np.fft.rfft(x_w, axis=-1)
    print('rfft', X.shape)

    return X

def istft(X, fft_size=1024, hop_size=256):
    w = np.hanning(fft_size)
    # w = signal.windows.hann(fft_size, sym=False)

    # Compute the IFFT
    x_w = np.fft.irfft(X, axis=-1)
    print('irfft', x_w.shape)

    # Overlap-add the windows
    x = np.zeros((len(x_w)-1)*hop_size + fft_size)
    for n,i in enumerate(range(0, len(x) - fft_size, hop_size)):
        x[i:i+fft_size] += w * x_w[n]
    print('overlap add', x.shape)

    # Remove padding
    x = x[int(fft_size // 2):-int(fft_size // 2)]
    print('remove padding output signal', x.shape)

    return x

second = 10
sampling_rate = 16000
fs = sampling_rate * second  # Total number of samples
t = np.linspace(0, second, fs)  # Time vector

x = 0
for Hz in range(1, 10, 2):
    x += 1/Hz * np.sin( 2*np.pi * Hz*t )  # Input signal

# Compute the STFT of the test signal
window_size = 1024
hop_size = 256
stft_matrix = stft(x, window_size, hop_size)

# Compute the inverse STFT of the STFT matrix
x_hat = istft(stft_matrix, window_size, hop_size)

# Plot the magnitude of the STFT
plt.imshow(np.abs(stft_matrix).T, origin='lower', aspect='auto', cmap='inferno')
plt.xlabel('Time (frames)')
plt.ylabel('Frequency (bins)')
plt.show()

# Plot the original and reconstructed signals
t2 = np.linspace(0, 1, len(x_hat))
fig, axs = plt.subplots(2, 1)

axs[0].plot(t, x, label='Original signal')
axs[1].plot(t2, x_hat, label='Reconstructed signal')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Amplitude')
plt.legend()
plt.show()


import torch
import torchaudio
torchaudio.save('test.wav', torch.tensor(x).float().unsqueeze(0), 44100)
torchaudio.save('test_hat.wav', torch.tensor(x_hat).float().unsqueeze(0), 44100)
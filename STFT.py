import numpy as np
import matplotlib.pyplot as plt

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




second = 2
sampling_rate = 44100
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
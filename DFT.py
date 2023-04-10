import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def inverse_fourier_transform(X):
    N = len(X)
    x = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x


second = 2
fs = 100 * second  # Sampling rate
t = np.linspace(0, second, fs)  # Time vector

x = 0
for Hz in range(1, 10, 2):
    x += 1/Hz * np.sin( 2*np.pi * Hz*t )  # Input signal

# Compute the Fourier transform of the signal
X1 = fourier_transform(x) / (fs / 2)
X2 = np.fft.fft(x) / (fs / 2)
x_recon = inverse_fourier_transform(X1)

# Plot the input signal and its Fourier transform
fig, axs = plt.subplots(4, 1)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Original signal')

axs[1].plot(np.arange(fs), np.abs(X1))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')
axs[1].set_title('Fourier transform (my function)')

axs[2].plot(np.arange(fs), np.abs(X2))
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Magnitude')
axs[2].set_title('Fourier transform np.fft.fft')

axs[3].plot(x_recon.real)
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Amplitude')
axs[3].set_title('Reconstructed signal')

plt.show()



import numpy as np
import matplotlib.pyplot as plt

def fft(signal):
    N = len(signal)
    if N <= 1:
        return signal

    even = fft(signal[0::2])
    odd = fft(signal[1::2])

    # ====================================================== #
    # ----------------- method 1 (for loop) ---------------- #
    # ====================================================== #

    # spectrum = np.zeros_like(signal, dtype=np.complex128)
    # for k in range(N//2):
    #     factor = np.exp(-2j * np.pi * k / N)
    #     T = factor * odd[k]
    #     spectrum[k] = even[k] + T
    #     spectrum[k + N//2] = even[k] - T


    # ====================================================== #
    # -------------- method 2 (vectorization) -------------- #
    # ====================================================== #

    factor = np.exp(-2j*np.pi * np.arange(N)/ N)
    spectrum = np.concatenate(
        [even + factor[:N//2] * odd,
        even + factor[N//2:] * odd]
        )

    return spectrum


def ifft(spectrum):
    N = len(spectrum)
    if N <= 1:
        return spectrum
    
    even = ifft(spectrum[0::2])
    odd = ifft(spectrum[1::2])

    # ====================================================== #
    # ----------------- method 1 (for loop) ---------------- #
    # ====================================================== #

    # signal = np.zeros_like(spectrum, dtype=np.complex128)
    # for k in range(N//2):
    #     factor = np.exp(2j * np.pi * k / N)
    #     T = factor * odd[k]
    #     signal[k] = even[k] + T
    #     signal[k + N//2] = even[k] - T
    

    # ====================================================== #
    # -------------- method 2 (vectorization) -------------- #
    # ====================================================== #

    factor = np.exp(2j*np.pi * np.arange(N)/ N)
    signal = np.concatenate(
        [even + factor[:N//2] * odd,
        even + factor[N//2:] * odd]
        )
    
    return signal


if __name__ == '__main__':
    fs = 128  # Sampling rate
    second = 2
    num_samples = fs * second  # Total number of samples
    t = np.linspace(0, second, num_samples)  # Time vector

    x = 0
    for Hz in range(1, 10, 2):
        x += 1/Hz * np.sin( 2*np.pi * Hz*t )  # Input signal

    # Compute the Fourier transform of the signal
    X1 = fft(x)
    X2 = np.fft.fft(x)
    x_recon = ifft(X1) / len(X1)

    # Equalize magnitude of the Fourier transform with amplitude of the signal
    X1 /= (num_samples / 2)
    X2 /= (num_samples / 2)

    # Compute the frequency vector
    freq = np.fft.fftfreq(num_samples, 1/fs)

    # Plot the input signal and its Fourier transform
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(t, x)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Original signal')

    axs[1].plot(freq, np.abs(X1))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_title('Fourier transform (my function)')

    axs[2].plot(freq, np.abs(X2))
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Magnitude')
    axs[2].set_title('Fourier transform np.fft.fft')

    axs[3].plot(t, x_recon.real)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_title('Reconstructed signal')

    plt.show()
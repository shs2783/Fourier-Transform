import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(signal):
    N = len(signal)
    spectrum = np.zeros(N, dtype=np.complex128)
    
    # ====================================================== #
    # -------------- method 1 (double for loop) ------------ #
    # ====================================================== #
    
    # for k in range(N):
    #     for n in range(N):
    #         spectrum[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)


    # ====================================================== #
    # ----- method 2 (single for loop + vectorization) ----- #
    # ====================================================== #
    
    # vector_n = np.arange(N)
    # for k in range(N):
    #     exp = np.exp(-2j * np.pi * k * vector_n / N)
    #     spectrum[k] = (signal * exp).sum()


    # ====================================================== #
    #    method 3 (vectorization + matrix multiplication)    #
    # ====================================================== #

    vector_n = np.arange(N)
    vector_k = vector_n.reshape(-1, 1)
    exp = np.exp(-2j * np.pi * vector_k * vector_n / N)
    spectrum = np.dot(exp, signal)

    return spectrum

def inverse_fourier_transform(spectrum):
    N = len(spectrum)
    signal = np.zeros(N, dtype=np.complex128)
    
    # ====================================================== #
    # -------------- method 1 (double for loop) ------------ #
    # ====================================================== #
    
    # for n in range(N):
    #     for k in range(N):
    #         signal[n] += spectrum[k] * np.exp(2j * np.pi * k * n / N)


    # ====================================================== #
    # ----- method 2 (single for loop + vectorization) ----- #
    # ====================================================== #
    
    # vector_k = np.arange(N)
    # for n in range(N):
    #     exp = np.exp(2j * np.pi * vector_k * n / N)
    #     signal[n] = (spectrum * exp).sum()


    # ====================================================== #
    #    method 3 (vectorization + matrix multiplication)    #
    # ====================================================== #
    
    vector_k = np.arange(N)
    vector_n = vector_k.reshape(-1, 1)
    exp = np.exp(2j * np.pi * vector_k * vector_n / N)
    signal = np.dot(exp, spectrum)
    
    return signal / N


if __name__ == '__main__':
    fs = 128  # Sampling rate
    second = 2
    num_samples = fs * second  # Total number of samples
    t = np.linspace(0, second, num_samples)  # Time vector

    x = 0
    for Hz in range(1, 10, 2):
        x += 1/Hz * np.sin( 2*np.pi * Hz*t )  # Input signal

    # Compute the Fourier transform of the signal
    X1 = fourier_transform(x)
    X2 = np.fft.fft(x)
    x_recon = inverse_fourier_transform(X1)

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
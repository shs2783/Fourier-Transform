import math
import numpy as np
import matplotlib.pyplot as plt

def stft(signal, fft_size=1024, window_size=1024, hop_size=256):
    num_samples = len(signal)
    num_pad_samples = len(signal) + fft_size

    # Padding the signal so that frames are centered around their midpoint
    signal = np.pad(signal, int(fft_size // 2), mode='reflect')
    window = np.hanning(window_size)  # = scipy.signal.windows.hann(window_size, sym=False)

    # ============================================================== #
    # --------------------- method 1 (for loop) -------------------- #
    # ============================================================== #

    # num_freq = window_size//2 + 1
    # num_frames = math.ceil(num_samples / hop_size)
    # stft_matrix = np.zeros((num_freq, num_frames), dtype=np.complex128)
    
    # for i in range(num_frames):
    #     start = i * hop_size
    #     end = start + fft_size

    #     frame = window * signal[start:end]
    #     stft_matrix[:, i] = np.fft.rfft(frame)


    # ============================================================== #
    # ---------------- method 2 (list comprehension) --------------- #
    # ============================================================== #

    x_w = [window * signal[i:i + fft_size] for i in range(0, num_samples, hop_size)]
    x_w = np.array(x_w).T
    stft_matrix = np.fft.rfft(x_w, axis=0)
    
    return stft_matrix

def istft(stft_matrix, window_size=1024, fft_size=1024, hop_size=256):
    num_freq, num_frames = stft_matrix.shape
    num_samples = (num_frames-1) * hop_size + fft_size
    window = np.hanning(window_size)

    signal = np.zeros(num_samples)
    window_sum = np.zeros(num_samples)
    x_w = np.fft.irfft(stft_matrix, axis=0)
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + fft_size

        signal[start:end] += window * x_w[:, i]
        window_sum[start:end] += window

    # Normalize the output signal by dividing it by the sum of the window values
    signal /= window_sum

    # Remove padding
    signal = signal[int(fft_size // 2): -int(fft_size // 2)]

    return signal


if __name__ == '__main__':
    # Generate a test signal
    second = 1
    sampling_rate = 8000
    fs = sampling_rate * second  # Total number of samples
    t = np.linspace(0, second, fs)  # Time vector

    x = 0
    x += 1/1 * np.sin( 2*np.pi * 440*t )  # Input signal
    x += 1/2 * np.sin( 2*np.pi * 880*t )  # Input signal
    x += 1/4 * np.sin( 2*np.pi * 1320*t )  # Input signal


    # Set the parameters
    window_size = 128
    fft_size = 128
    hop_size = 32

    # Compute the STFT of the test signal
    stft_matrix = stft(x, window_size, fft_size, hop_size)

    # Compute the inverse STFT of the STFT matrix
    x_hat = istft(stft_matrix, window_size, fft_size, hop_size)


    # Plot the magnitude of the STFT
    plt.imshow(np.abs(stft_matrix), origin='lower', aspect='auto', cmap='inferno')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (bins)')
    plt.show()

    # Plot the original and reconstructed signals
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x)
    axs[1].plot(x_hat)
    axs[1].set_title('Reconstructed signal')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Amplitude')
    plt.show()


    # Test with torchaudio
    import torch
    import torchaudio

    signals, sampling_rate = torchaudio.load('guitar.wav')
    signals = signals.mean(dim=0)

    torch_stft_matrix = torch.stft(signals, 
                                fft_size, hop_size, window_size, 
                                window=torch.hann_window(window_size), 
                                center=True, 
                                normalized=False, 
                                onesided=True,
                                return_complex=True,
                                pad_mode='reflect')

    # torch_stft_matrix = torch.complex(torch_stft_matrix[..., 0], torch_stft_matrix[..., 1])  # in case of return_complex = False
    torch_stft_matrix = torch_stft_matrix.numpy()

    istft_signal = istft(torch_stft_matrix, window_size, fft_size, hop_size)
    istft_signal = torch.from_numpy(istft_signal)

    istft_signal = istft_signal.unsqueeze(0).float()
    torchaudio.save('torch_test.wav', istft_signal, sampling_rate)
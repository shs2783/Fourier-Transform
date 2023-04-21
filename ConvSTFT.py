'''https://github.com/huyanxin/DeepComplexCRN/blob/bc6fd38b0af9e8feb716c81ff8fbacd7f71ad82f/conv_stft.py'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

def init_kernels(win_len, fft_size, win_type='hann', onesided=True, inverse=False):
    N = fft_size
    window = get_window(win_type, win_len, fftbins=True)
    
    if onesided:  # recommand
        fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    else:
        fourier_basis = np.fft.fft(np.eye(N))[:win_len]

    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], axis=1).T

    if inverse:
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    kernel = kernel[:, None, :]
    window = window[None, :, None]

    kernel = torch.from_numpy(kernel.astype(np.float32))
    window = torch.from_numpy(window.astype(np.float32))
    return kernel, window

class ConvSTFT(nn.Module):
    def __init__(self, win_len, hop_size, fft_size=None, win_type='hann', center=True, onesided=True, return_mag_phase=False, fix=True):
        super(ConvSTFT, self).__init__() 
        
        if fft_size is None:
            self.fft_size = win_len
        else:
            self.fft_size = fft_size
        
        kernel, _ = init_kernels(win_len, self.fft_size, win_type, onesided)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)

        self.fft_size = self.fft_size
        self.hop_size = hop_size
        self.win_len = win_len
        self.center = center
        self.return_mag_phase = return_mag_phase
        self.pad = self.fft_size // 2

    def forward(self, inputs):
        if inputs.dim() == 1:
            inputs = inputs[None, None, :]
        elif inputs.dim() == 2:
            inputs = inputs[:, None, :]
        
        if self.center:
            inputs = F.pad(inputs, [self.pad, self.pad], mode='reflect')
        outputs = F.conv1d(inputs, self.weight, stride=self.hop_size)
        
        if self.return_mag_phase:
            dim = self.fft_size//2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase
        else:
            return outputs

class ConviSTFT(nn.Module):
    def __init__(self, win_len, hop_size, fft_size=None, win_type='hann', center=True, onesided=True, fix=True):
        super(ConviSTFT, self).__init__() 
        
        if fft_size is None:
            self.fft_size = win_len
        else:
            self.fft_size = fft_size

        kernel, window = init_kernels(win_len, self.fft_size, win_type, onesided, inverse=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

        self.fft_size = self.fft_size
        self.hop_size = hop_size
        self.win_len = win_len
        self.center = center
        self.pad = self.fft_size // 2

    def forward(self, inputs, phase=None, output_length=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        output_length: (int) adjust the length of output
        """ 

        if phase is not None:
            mags = inputs
            real = mags * torch.cos(phase)
            imag = mags * torch.sin(phase)
            inputs = torch.cat([real, imag], dim=1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.hop_size)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.hop_size)
        outputs = outputs / (coff + 1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)

        if self.center:
            outputs = outputs[..., self.pad:]
            outputs = outputs[..., :-self.pad] if output_length is None else outputs

        if output_length is not None:
            outputs = outputs[..., :output_length]
        
        return outputs.squeeze(1).float()


if __name__ == '__main__':
    # Generate a test signal
    second = 1
    fs = 8000  # sampling rate
    num_samples = fs * second  # Total number of samples
    t = np.linspace(0, second, num_samples)  # Time vector

    x = 0
    x += 1/1 * np.sin( 2*np.pi * 500*t )  # Input signal
    x += 1/2 * np.sin( 2*np.pi * 1000*t )  # Input signal
    x += 1/4 * np.sin( 2*np.pi * 1500*t )  # Input signal
    x += 1/8 * np.sin( 2*np.pi * 2000*t )  # Input signal
    x = torch.from_numpy(x).float().reshape(1, -1)

    # Set the parameters
    window_size = 400
    fft_size = 400
    hop_size = 200

    import torchaudio
    x, sampling_rate = torchaudio.load('guitar.wav')
    num_samples = x.shape[1]
    fs = sampling_rate

    stft = ConvSTFT(window_size, hop_size, fft_size, return_mag_phase=True)
    istft = ConviSTFT(window_size, hop_size, fft_size)

    # Compute the STFT of the test signal
    mags, phase = stft(x)

    # Compute the inverse STFT of the STFT matrix
    x_hat = istft(mags, phase)

    # Convert magnitude to dB
    mags = mags[0].numpy()
    dB = 10 * np.log(mags + 1e-8)
    dB = np.maximum(dB, dB.max() - 80)

    x = x[0].numpy()
    x_hat = x_hat[0].numpy()
    freq = np.fft.rfftfreq(num_samples, 1/fs)

    # Plot the dB of the STFT
    plt.imshow(dB, origin='lower', aspect='auto', cmap='inferno', extent=[0, second, 0, freq[-1]])
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

    ### torch stft and custom istft
    torch_stft_matrix = torch.stft(signals,
                                fft_size, hop_size, window_size,
                                window=torch.hann_window(window_size),
                                center=True,
                                normalized=False,
                                onesided=True,
                                return_complex=False,
                                pad_mode='reflect')

    stft_matrix = torch.cat([torch_stft_matrix[..., 0], torch_stft_matrix[..., 1]], dim=1)

    istft = ConviSTFT(window_size, hop_size, fft_size)
    istft_signal = istft(stft_matrix)

    torchaudio.save('torch_test1.wav', istft_signal, sampling_rate)


    ### custom stft and torch istft
    stft = ConvSTFT(window_size, hop_size, fft_size, return_mag_phase=False)

    stft_matrix = stft(signals)
    stft_matrix = torch.stack([stft_matrix[:, :fft_size//2 + 1, :],
                               stft_matrix[:, fft_size//2 + 1:, :]],
                               dim=-1)

    istft_signal = torch.istft(stft_matrix,
                                fft_size, hop_size, window_size,
                                window=torch.hann_window(window_size),
                                center=True,
                                normalized=False,
                                onesided=True,
                                return_complex=False)

    torchaudio.save('torch_test2.wav', istft_signal, sampling_rate)
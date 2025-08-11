import numpy as np
import torch


def rayleigh_sommerfeld_2d(U_in, wavelength, z, L):


    # Get grid size from the input field
    grid_size_c = U_in.shape[0]
    grid_size_r = U_in.shape[1]

    # Define spatial frequency coordinates
    fx = torch.fft.fftshift(torch.fft.fftfreq(grid_size_r, d=(L / grid_size_r)))  # Frequency coordinates
    fy = torch.fft.fftshift(torch.fft.fftfreq(grid_size_c, d=(L / grid_size_c)))
    FX, FY = torch.meshgrid(fx, fy)
    FX = torch.transpose(FX, 0, 1)
    FY = torch.transpose(FY, 0, 1)

    # Compute wave number k
    k = 2 * torch.pi / wavelength

    # Compute Rayleigh–Sommerfeld Transfer Function (Fourier domain)
    sqrt_term = 1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    H = torch.exp(1j * k * z * torch.sqrt(sqrt_term))


    # Compute the Fourier transform of the input field
    U_in_ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(U_in)))

    # Apply transfer function and inverse transform
    U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(U_in_ft * H)))

    return U_out

def get_the_max(raw, out, in1):

    H, W = raw.shape
    bh = out
    bw = in1

    nh, nw = np.ceil(H // bh), np.ceil(W // bw)

    blocks = raw.unfold(0, nh, nh).unfold(1, nw, nw)

    max_values = blocks.amax(dim=(2, 3))

    return max_values


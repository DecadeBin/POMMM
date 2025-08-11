import numpy as np
import torch


def huygens_fresnel_2d(U_in, wavelength, z, L):


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
    sqrt_term = (wavelength * FX) ** 2 + (wavelength * FY) ** 2
    H = torch.exp(-1j * k * z / 2 * sqrt_term) * np.exp(1j * k * z)


    # Compute the Fourier transform of the input field
    U_in_ft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(U_in)))

    # Apply transfer function and inverse transform
    U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(U_in_ft * H)))

    return U_out

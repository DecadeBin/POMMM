import numpy as np
import torch


def rayleigh_sommerfeld(U_in, wavelength, z, L, d):

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

    if (d == 0):

        # Compute Rayleigh–Sommerfeld Transfer Function (Fourier domain)
        sqrt_term = 1 - (wavelength * FY) ** 2
        H = torch.exp(1j * k * z * torch.sqrt(sqrt_term))

        # Compute the Fourier transform of the input field
        U_in_ft = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(U_in, dim=0), dim=0), dim=0)

        # Apply transfer function and inverse transform
        U_out = torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(U_in_ft * H, dim=0), dim=0), dim=0)
    else:

        # Compute Rayleigh–Sommerfeld Transfer Function (Fourier domain)
        sqrt_term = 1 - (wavelength * FX) ** 2
        H = torch.exp(1j * k * z * torch.sqrt(sqrt_term))

        # Compute the Fourier transform of the input field
        U_in_ft = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(U_in, dim=1), dim=1), dim=1)

        # Apply transfer function and inverse transform
        U_out= torch.fft.ifftshift(torch.fft.ifft(torch.fft.ifftshift(U_in_ft * H, dim=1), dim=1), dim=1)

    return U_out

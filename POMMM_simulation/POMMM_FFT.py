import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
from Rayleigh_Sommerfeld_Diffraction import get_the_max

indim1 = 100
indim2 = 100
odim2 = 100
input = torch.rand((1,indim1,indim2))
weight = torch.rand(1,odim2,indim2)

rep = 200
N = indim2 * rep
M = odim2
bmm_result = input.bmm(weight.transpose(1,2))
bmm_result = bmm_result / (torch.max(torch.max(bmm_result, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2) + 1e-8)

delta = 3.74e-6
g = 1
pi = 3.1415927410125732

t = (torch.linspace(0, N - 1, N) * delta).unsqueeze(0)
sig = torch.zeros(M, N, dtype=torch.complex64)

for i in range(indim1):
    phase = 1j * 2 * pi * i * t / (delta * 2 * indim1)
    phase = torch.exp(phase)
    sig[math.floor(M / 2 - indim1 / 2 * g + i * g): math.floor(M / 2 - indim1 / 2 * g + i * g + 1), :] = phase
full_sig = sig
fft_sig = torch.fft.fftshift(torch.fft.fft2(full_sig))

weight = weight
out_row = input.shape[-2]
out_col = odim2

input = input.repeat_interleave(rep, dim=2)
weight = weight.repeat_interleave(rep, dim=2)

input_pad_num_row = math.floor((M - input.shape[-2]) / 2)
input_pad_num_col = math.floor((N - input.shape[-1]) / 2)
weight_pad_num_row = math.floor((M - weight.shape[-2]) / 2)
weight_pad_num_col = math.floor((N - weight.shape[-1]) / 2)

pad_input = nn.ZeroPad2d(padding=(input_pad_num_col, input_pad_num_col, input_pad_num_row, input_pad_num_row))
input = pad_input(input)

full_matrix_A = (input)
full_matrix_B = (weight)

fft_sig_full = torch.fft.fftshift(torch.fft.fft(full_sig.unsqueeze(0) * full_matrix_A, dim=1), dim=1)
opc = fft_sig_full * full_matrix_B

#
opc_fft = torch.fft.fftshift(torch.fft.fft(opc, dim=2), dim=2)
opc_fft = torch.abs(opc_fft)

# Comparison
amplitude = (torch.abs(fft_sig[:,  N // 2 - (N // indim1 // 2) // 2 : N // 2 + (N // indim1 // 2) * (indim1 - 1 // 2)]))
temp = opc_fft[:, :, N // 2 - (N // indim1 // 2) // 2 : N // 2 + (N // indim1 // 2) * (indim1 - 1 // 2)].squeeze(0).squeeze(0)

Weight_temp = get_the_max(amplitude, odim2, indim1)
Weight_c = Weight_temp / torch.max(Weight_temp)
Weight_c_a = Weight_c.numpy()

Pommm_temp = get_the_max(temp, odim2, indim1)
Matrix_Pommm = Pommm_temp / torch.max(Pommm_temp)

plt.figure(figsize=(8, 8))
plt.imshow(Weight_c_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Weights')
plt.show()

bmm_result = bmm_result.squeeze(0)
bmm_result_a = bmm_result.numpy()
res_temp = torch.abs(Matrix_Pommm / Weight_c)
res = res_temp.transpose(0, 1)
Res1_a = res.numpy()

plt.figure(figsize=(8, 8))
plt.imshow(Res1_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('POMMM Results')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(bmm_result_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Standard MMM Results')
plt.show()

err = torch.mean(torch.abs(res - bmm_result)/ (bmm_result + 1e-8))

print(err)
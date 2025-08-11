import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt

indim1 = 3
indim2 = 576
odim2 = 16384
input = torch.rand((1,indim1,indim2))

weight = torch.rand(1,odim2,indim2)
rep = 20
N = indim2 * rep
M = odim2

bmm_result = input.bmm(weight.transpose(1,2))
bmm_result = bmm_result / (torch.max(torch.max(bmm_result, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2) + 1e-8)
delta = 3.74e-6

g = 1
pi = 3.1415927410125732

t = (torch.linspace(0, N - 1, N) * delta).unsqueeze(0) # 固定值
sig = torch.zeros(M, N, dtype=torch.complex64)

for i in range(indim1):
    phase = 1j * 2 * pi * i * t / (delta * 2 * indim1)
    phase = torch.exp(phase)
    sig[math.floor(M / 2 - indim1 / 2 * g + i * g): math.floor(M / 2 - indim1 / 2 * g + i * g + 1), :] = phase  # 固定值
full_sig = sig  # 固定值

fft_sig = torch.fft.fftshift(torch.fft.fft2(full_sig))  # 固定值


out_row = input.shape[-2]

input = input.repeat_interleave(rep, dim=2)
weight = weight.repeat_interleave(rep, dim=2)

input_pad_num_row = math.floor((M - input.shape[-2]) / 2)
input_pad_num_col = math.floor((N - input.shape[-1]) / 2)
weight_pad_num_row = math.floor((M - weight.shape[-2]) / 2)
weight_pad_num_col = math.floor((N - weight.shape[-1]) / 2)
if indim1%2==0:
    pad_input = nn.ZeroPad2d(padding=(input_pad_num_col, input_pad_num_col, input_pad_num_row, input_pad_num_row))
else:
    pad_input = nn.ZeroPad2d(padding=(input_pad_num_col, input_pad_num_col, input_pad_num_row, input_pad_num_row+1))
input = pad_input(input)

full_matrix_A = (input)
full_matrix_B = (weight)

fft_sig_full = torch.fft.fftshift(torch.fft.fft(full_sig.unsqueeze(0) * full_matrix_A, dim=1), dim=1)

opc = fft_sig_full * full_matrix_B

#
opc_fft = torch.fft.fftshift(torch.fft.fft(opc, dim=2), dim=2)
opc_fft = torch.abs(opc_fft)

I_out_a = opc_fft.squeeze(0).numpy()


temp = opc_fft[:, :, N // 2 : N].unsqueeze(1)

matrix_C = nn.functional.max_pool2d(temp, kernel_size=(1,math.floor(N / 2 / indim1 / 2)), stride=[1, math.floor(N / 2 / indim1)])

amplitude = (torch.abs(fft_sig).unsqueeze(0))[:, :, N // 2 : N].unsqueeze(1)
amplitude = nn.functional.max_pool2d(amplitude, kernel_size=(1,math.floor(N / 2 / indim1 / 2)), stride=[1, math.floor(N / 2 / indim1)])
amplitude = (amplitude / (torch.max(amplitude) + 1e-8))  # 固定值-------------------------------------------------------------------------------


res_temp = torch.abs(matrix_C / amplitude)[:,0,:,:]
# res_temp = (res_temp * scale)
res_temp = res_temp / (torch.max(torch.max(res_temp, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2) + 1e-8)
# res = torch.round(res_temp*255 / torch.max(res_temp)).transpose(0,1)[:50,:48]
res = res_temp.transpose(1, 2)[:, :out_row, :odim2]


err = torch.mean(torch.abs(res - bmm_result)/ (bmm_result+1e-8))

print(err)
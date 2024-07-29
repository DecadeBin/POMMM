import torch
import torch.nn as nn
import time
import torch.nn.functional as F

# Given input and weight, output light matrix calculations,
# Replacement of standard vit, normalised quantit and 8bit quantit by adjusting
# whether the weights are non-negative or not, the quantisation method,
# and the matrix multiplication method.
def quantum_matrix_mutiply_fc_optical(input,weight):
    weight = torch.abs(weight)
    weight = weight.transpose(0, 1)
    weight = weight.unsqueeze(0)
    weight = weight.repeat(input.shape[0], 1, 1)

    input, input_max = norm_matrix_channel(input)# quantum_matrix_channel(input)
    weight, weight_max = norm_matrix_channel(weight,-2)# quantum_matrix_channel(weight,-2)

    max_multi = input_max.bmm(weight_max)
    out = optical_mvm(input,weight).to(torch.float32)
    out = out*(max_multi)
    return out



delta = 7.56e-6 / 2
N = 1900
M = 50
L = 50
g = 2
scale = 0.0625
pi = 3.1415927410125732
q = torch.ones(2, N * 2).cuda()
p = torch.ones(1,2).cuda()
pp = torch.ones(L,2).cuda()
t_temp = (torch.linspace(0, 2 * N - 2, N) * delta).unsqueeze(0).cuda() # 固定值
t = torch.zeros(1, 2 * N).cuda()
sig = torch.zeros(2 * M, 2 * N,dtype = torch.complex64).cuda()
for i in range(N):
    t[0,2 * i:2 * (i+1)] = t_temp[0,i] * p   # 固定值
for i in range(L):
    phase = 1j * 2 * pi * (i+1)* t / (delta * 4 * L)
    phase = torch.exp(phase)
    sig[int(M - L / 2 * g + i * g) : int(M - L / 2 * g + i * g + 2),:] = q * phase  # 固定值
full_sig = torch.zeros(4 * M, 4 * N,dtype=torch.complex64).cuda()
full_sig[M : 3 * M, N  : 3 * N] = sig  # 固定值
fft_sig = torch.fft.fftshift(torch.fft.fft2(full_sig))  # 固定值

row = torch.abs(fft_sig[2 * M-1,:])
col = torch.abs(fft_sig[:,2 * N + int(N / L)])
row_l = torch.zeros(1,L)
col_l = torch.zeros(L,1)
for i in range(L):
    row_l[:, i] = row[2 * N + (i + 1) * int(N / L)]
    col_l[i, :] = col[2 * M - int(L / 2) * g + i * g - 1]
batch = 4
amplitude_temp = col_l.unsqueeze(0).bmm(row_l.unsqueeze(0)).squeeze(0)
amplitude = (amplitude_temp / torch.max(amplitude_temp)).unsqueeze(0).cuda()  # 固定值-------------------------------------------------------------------------------
feature_dim = L
iter = int(N / feature_dim)*2



def optical_mvm(input,weight):
    weight = weight.transpose(1,2)
    out_row = input.shape[-2]
    out_col = weight.shape[-2]
    input_pad_num_row = L - input.shape[-2]
    input_pad_num_col = L - input.shape[-1]
    weight_pad_num_row = L - weight.shape[-2]
    weight_pad_num_col = L - weight.shape[-1]

    pad_input = nn.ZeroPad2d(padding=(0,input_pad_num_col,0,input_pad_num_row))
    input = pad_input(input)
    pad_weight = nn.ZeroPad2d(padding=(0,weight_pad_num_col,0,weight_pad_num_row))
    weight = pad_weight(weight)

    input = input.repeat_interleave(2, dim=1)
    input = input.repeat_interleave(iter, dim=2)
    weight = weight.repeat_interleave(2, dim=1)
    weight = weight.repeat_interleave(iter, dim=2)

    pad_full = nn.ZeroPad2d(padding=(N, N, M, M))
    full_matrix_A = pad_full(input)
    full_matrix_B = pad_full(weight)

    fft_sig_full = torch.fft.fftshift(torch.fft.fft(full_sig * full_matrix_A, dim=1), dim=1)
    opc = fft_sig_full * full_matrix_B  #
    opc_fft = torch.fft.fftshift(torch.fft.fft(opc, dim=2), dim=2)
    opc_fft = torch.abs(opc_fft)
    temp = opc_fft[:,2*M-L:2*M+L,2*N+int(N/L):3*N+1].unsqueeze(1)
    conv_kernal = torch.tensor(
        [[1.],
        [0]]
    ).unsqueeze(0).unsqueeze(0).cuda()
    matrix_C = torch.nn.functional.conv2d(temp,conv_kernal,stride=[2,int(N/L)]).squeeze(1)

    res_temp = torch.abs(matrix_C / amplitude)
    # res_temp = (res_temp * scale)
    res_temp = res_temp / (torch.max(torch.max(res_temp, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2))
    # res = torch.round(res_temp*255 / torch.max(res_temp)).transpose(0,1)[:50,:48]
    res = res_temp.transpose(1, 2)[:, :out_row, :out_col]
    return res

def norm_matrix_channel(input,diim = -1):
    # input_max = torch.max(input,dim=diim).values.unsqueeze(diim)
    input_max = (torch.max(torch.max(input, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2))
    input = (input/ (input_max+1e-8))
    return input, input_max

def image2matrix(input,patch = 4,stride = 4):
    b = input.shape[0]  #  input[batch,1,28,28]
    input_pad = nn.ZeroPad2d(padding=(1, 2, 1, 2)) # 31,31
    input = input_pad(input)
    out = torch.zeros(b,patch*patch,patch*patch).cuda()
    for i in range(patch):
        for j in range(patch):
            tmp = input[:,0,stride*i:stride*i + patch,stride*j:stride*j+patch].reshape(b,1,patch*patch)
            out[:,patch*i+j,:] = tmp[:,0,:]

    return out

def quantum_matrix_channel(input,diim = -1):
    input_max = torch.max(input,dim=diim).values.unsqueeze(diim)
    input = torch.round(input * 255 / (input_max+1e-8))
    return input, input_max



class ocnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        weight = torch.empty(args.embed_dim,36)
        torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
        self.weight = nn.Parameter(weight,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,36,args.embed_dim),requires_grad=True)

    def forward(self, input, i = 0):
        input = image2matrix(input,patch=6,stride=5)  # 得到[batch,49,16]
        x_cnn = input
        bias = self.bias.repeat(input.shape[0], 1, 1)  # 得到[batch,1,50]


        # weight = self.weight # 标准mmm
        # weight = weight.repeat_interleave(3,1)
        # 将weight和x到50*50
        input_pad = nn.ZeroPad2d(padding=(0,14,0,14))
        input = input_pad(input)
        weight_pad = nn.ZeroPad2d(padding=(0,14,0,0))
        weight = weight_pad(self.weight)
        x = quantum_matrix_mutiply_fc_optical(input,weight)  # 输入[batch,49,16] ,权重 [16,50]
        x = x[:,:36,:50] + bias
        return x,x_cnn
class omm(nn.Module):
    def __init__(self, para_num_in,para_num_out):
        super().__init__()
        weight = torch.empty(para_num_out,para_num_in)
        torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
        self.weight = nn.Parameter(weight,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,para_num_out),requires_grad=True)

    def forward(self, input, i = 0):
        bias = self.bias.unsqueeze(0)
        bias = bias.repeat(input.shape[0], 1, 1)

        x = quantum_matrix_mutiply_fc_optical(input,self.weight)
        x = x+ bias

        return x
import numpy as np
class cnn_fc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ocnn = ocnn(args)
        self.active = nn.ReLU()

        self.fc =torch.nn.Linear(36*50,10) # omm(36*50,10)
        # self.fc2 = nn.Linear(2500,10)
    def forward(self, x,i):
        # 第一层卷积，相当于49*16*16*50的矩阵乘法
        x = x+torch.ones(1,28,28).cuda()
        x,x_cnn = self.ocnn(x)
        x = self.active(x)
        x = x.reshape(batch,1,36*50)
        x_fc = x
        x = self.fc(x)

        x = x.squeeze(1)

        return x,x_cnn,x_fc

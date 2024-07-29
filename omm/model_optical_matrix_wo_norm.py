import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
import random

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)             # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)        # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True  # NVD深度训练加速算法库

# Given input and weight, output light matrix calculations,
# Replacement of standard vit, normalised quantit and 8bit quantit by adjusting
# whether the weights are non-negative or not, the quantisation method,
# and the matrix multiplication method.
def quantum_matrix_mutiply_fc_optical(input,weight):
    weight = torch.abs(weight)  # self.weight#
    weight = weight.transpose(0, 1)
    weight = weight.unsqueeze(0)
    weight = weight.repeat(input.shape[0], 1, 1)

    input, input_max =norm_matrix_channel(input)# quantum_matrix_channel(input) #
    weight, weight_max = norm_matrix_channel(weight,-2)# quantum_matrix_channel(weight,-2)#

    max_multi = input_max.bmm(weight_max)
    # out = input.bmm(weight).to(torch.float32)
    out = optical_mvm(input,weight).to(torch.float32)
    out = out*(max_multi)
    return out

# Optical matrix multiplication simulation code ############################################################################################### ↓

delta = 7.56e-6 / 2
N = 1900 # N越大越准
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
amplitude = (amplitude_temp / torch.max(amplitude_temp)).unsqueeze(0).cuda()  # Fixed value-------------------------------------------------------------------------------
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
    res_temp = res_temp / (torch.max(torch.max(res_temp, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2)+1e-8)
    # res = torch.round(res_temp*255 / torch.max(res_temp)).transpose(0,1)[:50,:48]
    res = res_temp.transpose(1, 2)[:, :out_row, :out_col]
    return res

# Optical matrix multiplication simulation code ############################################################################################### ↑


def image2matrix(input,patch = 4,stride = 4):
    b = input.shape[0]  #  input[batch,1,28,28]
    # input_pad = nn.ZeroPad2d(padding=(1, 2, 1, 2)) # 31,31
    # input = input_pad(input)
    patch_num = 28//patch
    out = torch.zeros(b,patch_num * patch_num,patch*patch).cuda()
    for i in range(patch_num):
        for j in range(patch_num):
            tmp = input[:,0,stride*i:stride*(i+1) ,stride*j:stride*(j+1)].reshape(b,1,patch*patch)
            out[:,patch_num*i+j,:] = tmp[:,0,:]

    return out

class ocnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        weight = torch.empty(args.embed_dim,16)
        torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
        self.weight = nn.Parameter(weight,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,49,args.embed_dim),requires_grad=True)

    def forward(self, input, i = 0):
        input = image2matrix(input,patch=4,stride=4)  # 得到[batch,49,16]

        bias = self.bias.repeat(input.shape[0], 1, 1)  # 得到[batch,1,50]


        # weight = self.weight
        # weight = weight.repeat_interleave(3,1)
        # 将weight和x到50*50
        input_pad = nn.ZeroPad2d(padding=(0,34,0,1))
        input = input_pad(input)
        weight_pad = nn.ZeroPad2d(padding=(0,34,0,0))
        weight = weight_pad(self.weight)
        x = quantum_matrix_mutiply_fc_optical(input,weight)  # 输入[batch,49,16] ,权重 [16,50]
        x = x[:,:49,:50] + bias
        return x

def norm_layer(input):
    min = torch.min(input, dim=-1).values.unsqueeze(-1)
    max = torch.max(input, dim=-1).values.unsqueeze(-1)
    norm = (input - min) / (max - min + 0.000001)
    return norm
rand_scale = 0.0
class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 =ocnn(args)#  nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        # padding = (1, 2, 1, 2)# B C IH IW -> B E IH/P IW/P (Embedding the patches)
        # x = F.pad(x, padding, "constant", 0) # 如果是Fminist数据集，需要padding成35*35，然后用5*5卷积得到7*7的特征
        x = x+torch.ones(1,28,28).cuda()
        x = self.conv1(x)
        # x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        # x = x.transpose(1, 2)  # B E S -> B S E
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        return x
# 8bit量化精度 分类50%



# Quantisation of matrices before multiplication using OMM
def quantum_matrix_channel(input,diim = -1):
    input_max = torch.max(input,dim=diim).values.unsqueeze(diim)
    input = torch.round(input * 255 / (input_max+1e-8))
    # if torch.isnan(torch.sum(input)):
    #     input = torch.where(torch.isnan(input), input * 255 /input_max , input)

    return input, input_max






def norm_matrix_channel(input,diim = -1):
    input_max = torch.max(input,dim=diim).values.unsqueeze(diim)
    input = (input / (input_max+1e-8))
    return input, input_max


class omm(nn.Module):
    def __init__(self, args):
        super().__init__()
        weight = torch.empty(args.embed_dim,args.embed_dim)
        torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
        self.weight = nn.Parameter(weight,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(args.embed_dim),requires_grad=True)

    def forward(self, input, i = 0):
        bias = self.bias.unsqueeze(0).unsqueeze(1)
        bias = bias.repeat(input.shape[0], 1, 1)

        x = quantum_matrix_mutiply_fc_optical(input,self.weight) + bias


        return x

class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = omm(args)
        self.keys = omm(args)
        self.values = omm(args)

    def forward(self, x,i):
        m, s, e = x.shape  # x是经过abs(norm()),需要量化到0-255，后续QKV的全连接层权重和偏置也要量化。目前都是float32
        # 每个都开始通过光矩阵方法，添加适当误差
        # 可以直接写成bmm(weight)+bias

        xq =self.queries(x)


        xk = self.keys(x)


        xv = self.values(x)



        # xq_max用于量化输入，计算完矩阵乘法后，需要后续在除以，才能用后面的softmax，
        xq = xq.reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        xq = xq.transpose(1, 2)   # B, Q, H, HE -> B, H, Q, HE



        xk = xk.reshape(m, s, self.n_attention_heads, self.head_embed_dim)
        xk = xk.transpose(1, 2) # B, K, H, HE -> B, H, K, HE



        xv = xv.reshape(m, s, self.n_attention_heads, self.head_embed_dim)
        xv = xv.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE


        xq = xq.reshape([-1, s, self.head_embed_dim])  # B, H, Q, HE -> (BH), Q, HE
        xk = xk.reshape([-1, s, self.head_embed_dim])  # B, H, K, HE -> (BH), K, HE
        xv = xv.reshape([-1, s, self.head_embed_dim])  # B, H, V, HE -> (BH), V, HE
        xk = xk.transpose(1, 2)  # (BH), K, HE -> (BH), HE, K




        x_attention = xq.bmm(xk)
        # x_attention = xq.bmm(xk)  # (BH), Q, HE  .  (BH), HE, K -> (BH), Q, K   QKV三个矩阵相乘，需要量化。

        # 光矩阵
        # x_attention = x_attention * (1 + torch.randn(x_attention.shape) * rand_scale)
        x_attention = torch.softmax(x_attention, dim=-1)
        # 光矩阵
        x = x_attention.bmm(xv)
        # x = x_attention.bmm(xv)  # (BH), Q, K . (BH), V, HE -> (BH), Q, HE
        x = x.reshape([-1, self.n_attention_heads, s, self.head_embed_dim])  # (BH), Q, HE -> B, H, Q, HE
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(m, s, e)  # B, Q, H, HE -> B, Q, E
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfAttention(args)
        self.fc1 =  omm(args)# nn.Linear(args.embed_dim, args.embed_dim )
        self.activation = nn.ReLU()
        self.fc2 =  nn.Linear(args.embed_dim , args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)

        '''
        LayerNorm其主要作用是对每个样本的所有特征进行归一化，也就是说，它对单个样本的不同特征进行操作。
        BatchNorm则是对一个batch-size样本内的每个特征做归一化，也就是针对不同样本的同一特征做操作。
        '''
    def forward(self, x,i):
        xn = torch.abs(x)
        x = self.attention(xn,i)  # Skip connections
        x1 = x
        xn2 = self.norm1(x)  # 第二个层归一化

        # 光矩阵
        x = self.fc1(xn2)
        x = self.activation(x)
        x = self.fc2(x)

        x = x1 + x  # Skip connections
        return x


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        self.encoder = Encoder(args)
        self.norm2 = nn.LayerNorm(args.embed_dim) # 第三个层归一化
        self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(args)

    def forward(self, x,i):
        x = self.embedding(x)
        x = self.norm(x)  # 第一个层归一化
        x = self.encoder(x,i)
        x = self.norm2(x)
        x = self.classifier(x)
        return x



import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import math
import random
import numpy as np
torch.cuda.set_device(0)

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True  # NVD深度训练加速算法库

seed_torch(0)

def cal_omm_fix_para(N):


    # input[(batch,indim1,indim2)], weight[batch,odim2,indim2]

    # bmm_result = input.bmm(weight.transpose(1,2))
    # bmm_result = bmm_result / (torch.max(torch.max(bmm_result, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2) + 1e-8)

    delta = 7.56e-6 / 2
    M = 50  # input.shape[1] # indim1
    g = 2
    scale = 0.0625
    pi = 3.1415927410125732
    q = torch.ones(2, N * 2).cuda()
    p = torch.ones(1, 2).cuda()
    t_temp = (torch.linspace(0, 2 * N - 2, N) * delta).unsqueeze(0).cuda()  # 固定值
    t = torch.zeros(1, 2 * N).cuda()
    sig = torch.zeros(2 * M, 2 * N, dtype=torch.complex64).cuda()
    for i in range(N):
        t[0, 2 * i:2 * (i + 1)] = t_temp[0, i] * p  # 固定值

    indim50 = 50
    indim50_2 = 50
    sig_50 = torch.zeros(2 * M, 2 * N, dtype=torch.complex64).cuda()
    for i in range(indim50):
        phase = 1j * 2 * pi * (i + 1) * t / (delta * 4 * indim50)
        phase = torch.exp(phase)
        sig_50[math.floor(M - indim50 / 2 * g + i * g): math.floor(M - indim50 / 2 * g + i * g + 2),
        :] = q * phase  # 固定值
    full_sig_50 = torch.zeros(4 * M, 4 * N, dtype=torch.complex64).cuda()
    full_sig_50[M: 3 * M, N: 3 * N] = sig_50  # 固定值
    # sig_pad_1 = M*4 - sig.shape[0]

    fft_sig_50 = torch.fft.fftshift(torch.fft.fft2(full_sig_50))  # 固定值
    iter_50 = math.floor(N / indim50_2) * 2

    indim1_1 = 1
    indim1_2 = 1800
    sig_1 = torch.zeros(2 * M, 2 * N, dtype=torch.complex64).cuda()
    for i in range(indim1_1):
        phase = 1j * 2 * pi * (i + 1) * t / (delta * 4 * indim1_1)
        phase = torch.exp(phase)
        sig_1[math.floor(M - indim1_1 / 2 * g + i * g): math.floor(M - indim1_1 / 2 * g + i * g + 2),
        :] = q * phase  # 固定值
    full_sig_1 = torch.zeros(4 * M, 4 * N, dtype=torch.complex64).cuda()
    full_sig_1[M: 3 * M, N: 3 * N] = sig_1  # 固定值
    # # sig_pad_1 = M*4 - sig.shape[0]
    fft_sig_1 = torch.fft.fftshift(torch.fft.fft2(full_sig_1))  # 固定值
    iter_1 = math.floor(N / indim1_2) * 2
    return fft_sig_50,iter_50,full_sig_50,fft_sig_1,iter_1,full_sig_1

# N = 900
#
# # input[(batch,indim1,indim2)], weight[batch,odim2,indim2]
#
# # bmm_result = input.bmm(weight.transpose(1,2))
# # bmm_result = bmm_result / (torch.max(torch.max(bmm_result, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2) + 1e-8)
#
# delta = 7.56e-6 / 2
# M = 50  # input.shape[1] # indim1
# g = 2
# scale = 0.0625
# pi = 3.1415927410125732
# q = torch.ones(2, N * 2).cuda()
# p = torch.ones(1, 2).cuda()
# t_temp = (torch.linspace(0, 2 * N - 2, N) * delta).unsqueeze(0).cuda()  # 固定值
# t = torch.zeros(1, 2 * N).cuda()
# sig = torch.zeros(2 * M, 2 * N, dtype=torch.complex64).cuda()
# for i in range(N):
#     t[0, 2 * i:2 * (i + 1)] = t_temp[0, i] * p  # 固定值
#
#
# indim50 = 50
# indim50_2 = 50
# sig_50 = torch.zeros(2 * M, 2 * N, dtype=torch.complex64).cuda()
# for i in range(indim50):
#     phase = 1j * 2 * pi * (i + 1) * t / (delta * 4 * indim50)
#     phase = torch.exp(phase)
#     sig_50[math.floor(M - indim50 / 2 * g + i * g): math.floor(M - indim50 / 2 * g + i * g + 2), :] = q * phase  # 固定值
# full_sig_50 = torch.zeros(4 * M, 4 * N, dtype=torch.complex64).cuda()
# full_sig_50[M: 3 * M, N: 3 * N] = sig_50  # 固定值
# # sig_pad_1 = M*4 - sig.shape[0]
#
# fft_sig_50 = torch.fft.fftshift(torch.fft.fft2(full_sig_50))  # 固定值
# iter_50 = math.floor(N / indim50_2) * 2
#
#
# indim1_1 = 1
# indim1_2 = 1800
# sig_1 = torch.zeros(2 * M, 2 * N, dtype=torch.complex64).cuda()
# for i in range(indim1_1):
#     phase = 1j * 2 * pi * (i + 1) * t / (delta * 4 * indim1_1)
#     phase = torch.exp(phase)
#     sig_1[math.floor(M - indim1_1 / 2 * g + i * g): math.floor(M - indim1_1 / 2 * g + i * g + 2), :] = q * phase  # 固定值
# full_sig_1 = torch.zeros(4 * M, 4 * N, dtype=torch.complex64).cuda()
# full_sig_1[M: 3 * M, N: 3 * N] = sig_1  # 固定值
# # # sig_pad_1 = M*4 - sig.shape[0]
# fft_sig_1 = torch.fft.fftshift(torch.fft.fft2(full_sig_1))  # 固定值
# iter_1 = math.floor(N / indim1_2) * 2

def add_noise(data,std = 0.05):
    noise = 1 + torch.normal(mean=0, std=std, size=data.shape).cuda()
    data = data * noise
    return data
# self.handle2 = self.ocnn.register_forward_hook(print_post_shape)
def omm_hook(args,module, input, output):
    input = input[0] # b,1800
    bias = module.bias.unsqueeze(0).unsqueeze(1)
    bias = bias.repeat(input.shape[0], 1, 1)
    weight = torch.abs(module.weight)
    x = matrix_mutiply_optical(args,input, weight, qunatum=True, omm=True, name=None) + bias
    output = x
    # output.data = add_noise(x)
    # output.requires_grad = True
    return output

def ocnn_hook(args,module, input, output):
    input = input[0]
    if module.weight.shape[-1]==36:
        input = image2matrix_2fc(input, patch=6, stride=5)
    else:
        input = image2matrix_vit(input, patch=4, stride=4)  # 得到[batch,49,16]

    bias = module.bias.repeat(input.shape[0], 1, 1)  # 得到[batch,1,50]


    weight = torch.abs(module.weight)
    # weight = weight.repeat_interleave(3,1)
    # 将weight和x到50*50
    out_shape1 = input.shape[1]
    input_pad_1 = 50 - input.shape[1]
    input_pad_2 = 50 - input.shape[2]
    input_pad = nn.ZeroPad2d(padding=(0, input_pad_2, 0, input_pad_1))
    input = input_pad(input)
    weight_pad_1 = 50 - weight.shape[0]
    weight_pad_2 = 50 - weight.shape[1]
    weight_pad = nn.ZeroPad2d(padding=(0, weight_pad_2, 0, weight_pad_1))
    weight = weight_pad(weight)

    x = matrix_mutiply_optical(args,input, weight, qunatum=True,
                               omm=True)  # 输入[batch,49,16] ,权重 [16,50]
    # print(1)

    x = x[:, :out_shape1, :50] + bias
    output = x
    return output

def optical_mvm(args,input,weight):
    weight = weight.transpose(1,2)
    # input[(batch,indim1,indim2)], weight[batch,odim2,indim2]
    indim1 = input.shape[1]
    # indim2 = input.shape[2]
    odim2 = weight.shape[1]


    if indim1==50:
        fft_sig = args.fft_sig_50# .clone()
        iter = args.iter_50
        full_sig = args.full_sig_50
    elif indim1==1:
        fft_sig = args.fft_sig_1# .clone()
        iter = args.iter_1
        full_sig = args.full_sig_1
    out_row = input.shape[-2]

    input = input.repeat_interleave(args.g, dim=1)
    input = input.repeat_interleave(iter, dim=2)
    weight = weight.repeat_interleave(args.g, dim=1)
    weight = weight.repeat_interleave(iter, dim=2)

    input_pad_num_row = math.floor((args.M * 4 - input.shape[-2]) / 2)
    input_pad_num_col = math.floor((args.N * 4 - input.shape[-1]) / 2)
    weight_pad_num_row = math.floor((args.M * 4 - weight.shape[-2]) / 2)
    weight_pad_num_col = math.floor((args.N * 4 - weight.shape[-1]) / 2)

    pad_input = nn.ZeroPad2d(padding=(input_pad_num_col, input_pad_num_col, input_pad_num_row, input_pad_num_row))
    input = pad_input(input)
    pad_weight = nn.ZeroPad2d(padding=(weight_pad_num_col, weight_pad_num_col, weight_pad_num_row, weight_pad_num_row))
    weight = pad_weight(weight)

    # pad_full = nn.ZeroPad2d(padding=(N, N, M, M))
    full_matrix_A = (input)
    full_matrix_B = (weight)

    fft_sig_full = torch.fft.fftshift(torch.fft.fft(full_sig.unsqueeze(0) * full_matrix_A, dim=1), dim=1)
    # plt.imshow(torch.abs(fft_sig_full)[0].data.cpu().numpy())
    # plt.show()
    opc = fft_sig_full * full_matrix_B

    #
    opc_fft = torch.fft.fftshift(torch.fft.fft(opc, dim=2), dim=2)
    opc_fft = torch.abs(opc_fft)
    # plt.imshow(opc_fft[0].data.cpu().numpy())
    # plt.show()
    temp = opc_fft[:, 2 * args.M - odim2:2 * args.M + odim2, 2 * args.N + math.floor(args.N / indim1):3 * args.N + 3].unsqueeze(1)
    # conv_kernal = torch.tensor(
    #     [[1.],
    #      [0]]
    # ).unsqueeze(0).unsqueeze(0).cuda()
    # matrix_C = torch.nn.functional.conv2d(temp, conv_kernal, stride=[2, math.floor(N / indim1)]).squeeze(1)
    matrix_C = nn.functional.max_pool2d(temp, kernel_size=(2, 3), stride=[2, math.floor(args.N / indim1)])

    amplitude = (torch.abs(fft_sig).unsqueeze(0))[:, 2 * args.M - odim2:2 * args.M + odim2,
                2 * args.N + math.floor(args.N / indim1):3 * args.N + 3].unsqueeze(1)
    amplitude = nn.functional.max_pool2d(amplitude, kernel_size=(2, 3), stride=[2, math.floor(args.N / indim1)])
    amplitude = (amplitude / (torch.max(
        amplitude) + 1e-8)).cuda()  # 固定值-------------------------------------------------------------------------------

    res_temp = torch.abs(matrix_C / amplitude)[:, 0, :, :]
    # res_temp = (res_temp * scale)
    res_temp = res_temp / (torch.max(torch.max(res_temp, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2) + 1e-8)
    # res = torch.round(res_temp*255 / torch.max(res_temp)).transpose(0,1)[:50,:48]
    res = res_temp.transpose(1, 2)[:, :out_row, :odim2]

    # err = torch.mean(torch.abs(res - bmm_result) / (bmm_result + 1e-8))
    #
    # print(err)
    return res

def matrix_mutiply_optical(args,input,weight,qunatum=False,omm=True,name=None):
    # weight = torch.abs(weight)
    weight = weight.transpose(0, 1)
    weight = weight.unsqueeze(0)
    weight = weight.repeat(input.shape[0], 1, 1)
    if qunatum==False  :#or input.shape[1]==1:
        input, input_max = norm_matrix_channel(input)# quantum_matrix_channel(input)
        weight, weight_max = norm_matrix_channel(weight,-2)# quantum_matrix_channel(weight,-2)
    else:
        input, input_max =  quantum_matrix_channel(input)
        weight, weight_max =  quantum_matrix_channel(weight,-2)

    max_multi = input_max.bmm(weight_max)

    if omm==False:# or name=='v': # #
        out = input.bmm(weight).to(torch.float32)
        out = out / (torch.max(torch.max(out, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2)+1e-8)
    else:
        out = optical_mvm(args,input,weight).to(torch.float32)
    # out_bmm = input.bmm(weight).to(torch.float32)
    # out_bmm = out_bmm / (torch.max(torch.max(out_bmm, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2)+1e-8)
    # err = torch.mean(torch.abs((out - out_bmm))/(out+1e-8))
    # print(err)
    out = out * (max_multi)

    return out

def norm_matrix_channel(input,diim = -1):
    input_max = torch.max(input,dim=diim).values.unsqueeze(diim)
    # input_max = (torch.max(torch.max(input, dim=2).values, 1).values.unsqueeze(1).unsqueeze(2))
    input = (input/ (input_max+1e-8))
    return input, input_max

def image2matrix_2fc(input,patch = 6,stride = 5):
    b = input.shape[0]  #  input[batch,1,28,28]
    input_pad = nn.ZeroPad2d(padding=(1, 2, 1, 2)) # 31,31
    input = input_pad(input)
    out = torch.zeros(b,patch*patch,patch*patch).cuda()
    for i in range(patch):
        for j in range(patch):
            tmp = input[:,0,stride*i:stride*i + patch,stride*j:stride*j+patch].reshape(b,1,patch*patch)
            out[:,patch*i+j,:] = tmp[:,0,:]
    return out
def image2matrix_vit(input,patch = 4,stride = 4):
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

def norm_layer(input):
    min = torch.min(input, dim=-1).values.unsqueeze(-1)
    max = torch.max(input, dim=-1).values.unsqueeze(-1)
    norm = (input - min) / (max - min + 0.000001)
    return norm
def quantum_matrix_channel(input,diim = -1):
    input_max = torch.max(input,dim=diim).values.unsqueeze(diim)
    input = torch.round(input * 255 / (input_max+1e-8))
    return input, input_max


class ocnn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.model_name==  '2fc':
            weight = torch.empty(args.embed_dim, 36)
            torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
            self.weight = nn.Parameter(weight, requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, 36, args.embed_dim), requires_grad=True)
        else:
            weight = torch.empty(args.embed_dim,16)
            torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
            self.weight = nn.Parameter(weight,requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,49,args.embed_dim),requires_grad=True)

    def forward(self, input, i = 0):
        if self.args.model_name==  '2fc':
            input = image2matrix_2fc(input, patch=6, stride=5)
        else:
            input = image2matrix_vit(input,patch=4,stride=4)  # 得到[batch,49,16]

        bias = self.bias.repeat(input.shape[0], 1, 1)  # 得到[batch,1,50]

        if self.args.exp_type==  'standard_mmm': # mmm_01,omm,standard_mmm,mmm0_255
            weight = self.weight
        else:
            weight = torch.abs(self.weight)
        # weight = weight.repeat_interleave(3,1)
        # 将weight和x到50*50
        out_shape1 = input.shape[1]
        input_pad_1 = 50 - input.shape[1]
        input_pad_2 = 50 - input.shape[2]
        input_pad = nn.ZeroPad2d(padding=(0,input_pad_2,0,input_pad_1))
        input = input_pad(input)
        weight_pad_1 = 50 - weight.shape[0]
        weight_pad_2 = 50 - weight.shape[1]
        weight_pad = nn.ZeroPad2d(padding=(0,weight_pad_2,0,weight_pad_1))
        weight = weight_pad(weight)


        x = matrix_mutiply_optical(self.args,input,weight,qunatum=self.args.qunatum,omm=self.args.omm)  # 输入[batch,49,16] ,权重 [16,50]



        x = x[:, :out_shape1, :50] + bias
        return x
class omm(nn.Module):
    def __init__(self, args,indim1=50,indim2=50,outdim2=50):
        super().__init__()
        self.args = args
        weight = torch.empty(outdim2,indim2)
        torch.nn.init.trunc_normal_(weight, std=0.1) + 0.  # 生成截断正态分布的随机数
        self.weight = nn.Parameter(weight,requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(outdim2),requires_grad=True)

    def forward(self, input, name = None):
        bias = self.bias.unsqueeze(0).unsqueeze(1)
        bias = bias.repeat(input.shape[0], 1, 1)
        if self.args.exp_type==  'standard_mmm': # mmm_01,omm,standard_mmm,mmm0_255
            weight = self.weight
        else:
            weight = torch.abs(self.weight)

        x = matrix_mutiply_optical(self.args,input,weight,qunatum=self.args.qunatum,omm=self.args.omm,name=name) + bias
        return x


rand_scale = 0.0

# 8bit量化精度 分类50%



# Quantisation of matrices before multiplication using OMM


class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conv1 =ocnn(args)#  nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding
        if self.args.hook==True:
            self.cnn_hook = self.conv1.register_forward_hook(ocnn_hook)

    def forward(self, x):
        if self.args.hook == True:
            try:
                self.cnn_hook.remove()
            except:
                pass
            self.cnn_hook = self.conv1.register_forward_hook(ocnn_hook)

        # 第一层卷积，相当于49*16*16*50的矩阵乘法
        if self.args.hook==False:
            try:
                self.cnn_hook.remove()
            except:
                pass

        # padding = (1, 2, 1, 2)# B C IH IW -> B E IH/P IW/P (Embedding the patches)
        # x = F.pad(x, padding, "constant", 0) # 如果是Fminist数据集，需要padding成35*35，然后用5*5卷积得到7*7的特征
        x = x+torch.ones(1,28,28).cuda()
        x = self.conv1(x)
        # x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        # x = x.transpose(1, 2)  # B E S -> B S E
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        return x


class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = omm(args)
        self.keys = omm(args)
        self.values = omm(args)
        if args.hook==True:
            self.queries_hook = self.queries.register_forward_hook(omm_hook)
            self.keys_hook = self.keys.register_forward_hook(omm_hook)
            self.values_hook = self.values.register_forward_hook(omm_hook)

    def forward(self, x,i):
        m, s, e = x.shape  # x是经过abs(norm()),需要量化到0-255，后续QKV的全连接层权重和偏置也要量化。目前都是float32
        # 每个都开始通过光矩阵方法，添加适当误差
        # 可以直接写成bmm(weight)+bias

        if self.args.hook == True:
            # 有可能出现本来没有hook，但是后来有了
            try:
                self.queries_hook.remove()
                self.keys_hook.remove()
                self.values_hook.remove()
            except:
                pass
            self.queries_hook = self.queries.register_forward_hook(omm_hook)
            self.keys_hook = self.keys.register_forward_hook(omm_hook)
            self.values_hook = self.values.register_forward_hook(omm_hook)


        if self.args.hook==False:
            try:
                self.queries_hook.remove()
                self.keys_hook.remove()
                self.values_hook.remove()
            except:
                pass


        xq =self.queries(x)


        xk = self.keys(x)


        xv = self.values(x,'v')



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
        self.args = args
        self.attention = SelfAttention(args)
        self.fc1 =  omm(args)# nn.Linear(args.embed_dim, args.embed_dim )
        self.activation = nn.ReLU()
        self.fc2 =  nn.Linear(args.embed_dim , args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        if self.args.hook==True:
            self.fc1_hook = self.fc1.register_forward_hook(omm_hook)

    def forward(self, x,i):
        if self.args.exp_type == 'three_stage':
            if self.training==True:
                if self.args.stage_time==2:
                    self.args.hook=True
                if self.args.stage_time==3:
                    for param in self.attention.parameters():
                        param.requires_grad = False
                    for param in self.norm1.parameters():
                        param.requires_grad = False
                    self.args.hook=True
            else:
                self.args.hook = True
        # # 第一层卷积，相当于49*16*16*50的矩阵乘法

        xn = torch.abs(x)
        x = self.attention(xn,i)  # Skip connections
        x1 = x
        xn2 = self.norm1(x)  # 第二个层归一化
        if self.args.exp_type == 'three_stage':
            if self.training==True:
                if  self.args.stage_time==2:
                    self.args.hook=False
                if  self.args.stage_time==3:
                    self.args.hook=True
            else:
                self.args.hook = True
        if self.args.hook == True:
            try:
                self.fc1_hook.remove()
            except:
                pass
            self.fc1_hook = self.fc1.register_forward_hook(omm_hook)

        if self.args.hook==False:
            try:
                self.fc1_hook.remove()
            except:
                pass

        # 光矩阵
        xn2 = torch.abs(xn2)
        x = self.fc1(xn2,'v')
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
        self.args = args
        self.embedding = EmbedLayer(args)
        self.encoder = Encoder(args)
        self.norm2 = nn.LayerNorm(args.embed_dim) # 第三个层归一化
        self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(args)

    def forward(self, x,i):
        if self.args.exp_type=='three_stage'==True:
            if self.training==False:
                self.args.hook = True
                x = self.embedding(x)
                x = self.norm(x)  # 第一个层归一化
                x = self.encoder(x, i)
                x = self.norm2(x)
                x = self.classifier(x)


            if self.args.stage_time==1:
                self.args.hook = True
                x = self.embedding(x)
                self.args.hook = False
                x = self.norm(x)  # 第一个层归一化
                x = self.encoder(x,i)
                x = self.norm2(x)
                x = self.classifier(x)
            if self.args.stage_time==2 or self.args.stage_time==3:
                self.args.hook = True
                for param in self.embedding.parameters():
                    param.requires_grad = False
                for param in self.norm.parameters():
                    param.requires_grad = False
                x = self.embedding(x)
                x = self.norm(x)  # 第一个层归一化
                x = self.encoder(x,i)
                x = self.norm2(x)
                x = self.classifier(x)

        else:
            x = self.embedding(x)
            x = self.norm(x)  # 第一个层归一化
            x = self.encoder(x, i)
            x = self.norm2(x)
            x = self.classifier(x)
        return x


class cnn_fc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ocnn = ocnn(args)
        self.active = nn.ReLU()

        self.fc = torch.nn.Linear(36*50,10)# omm(36*50,10)
        # self.fc = omm(args,indim1=0,indim2=36*50,outdim2=10)#


        # if self.args.hook==True:
        # if self.args.hook==True:
        #     self.handle_cnn = self.ocnn.register_forward_hook(ocnn_hook)
        if self.args.hook == True:
            self.handle_fc = self.ocnn.register_forward_hook(ocnn_hook)

    def forward(self, x,i):
        # print('forward start')

        if self.args.hook == True:
            try:
                self.handle_fc.remove()
            except:
                pass
            self.handle_fc = self.ocnn.register_forward_hook(ocnn_hook)


        # 第一层卷积，相当于49*16*16*50的矩阵乘法
        x = x+torch.ones(1,28,28).cuda()

        x = self.ocnn(x)
        x = self.active(x)
        x = x.contiguous().view(-1,1,36*50)


        # if self.args.exp_type==  'standard_mmm': # mmm_01,omm,standard_mmm,mmm0_255
        #     pass
        # else:
        #     self.fc.weight.data = torch.abs(self.fc.weight)

        x = self.fc(x)

        x = x.squeeze(1)


        return x

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from Rayleigh_Sommerfeld_Diffraction_1d import rayleigh_sommerfeld  # Import the diffraction function
from Rayleigh_Sommerfeld_Diffraction import rayleigh_sommerfeld_2d  # Import the diffraction function

# Matrix parameters
in1 = 4
in2 = 4
out = 4
torch.manual_seed(64)
matrix_a = torch.randint(0, 255, (in1, in2)).to(torch.float32)
torch.manual_seed(77)
matrix_a_w1 = torch.randint(0, 255, (in1, in2)).to(torch.float32)
torch.manual_seed(98)
# torch.manual_seed(2)
matrix_a_w2 = torch.randint(0, 255, (in1, in2)).to(torch.float32)

torch.manual_seed(32)
matrix_b = torch.randint(0, 255, (in2, out)).to(torch.float32)
torch.manual_seed(43)
matrix_b_w1 = torch.randint(0, 255, (in2, out)).to(torch.float32)
torch.manual_seed(7)
# torch.manual_seed(85)
matrix_b_w2 = torch.randint(0, 255, (in2, out)).to(torch.float32)

# Matrix on spatial light
grid_size = 4000
rep = int(np.floor(grid_size / in2))
matrix_a_l_s = torch.repeat_interleave(matrix_a, rep, dim=1)
matrix_b_t_s = matrix_b.transpose(0,1)
matrix_b_l_s = torch.repeat_interleave(matrix_b_t_s, rep, dim=1)

matrix_a_l_s_w1 = torch.repeat_interleave(matrix_a_w1, rep, dim=1)
matrix_b_t_s_w1 = matrix_b_w1.transpose(0,1)
matrix_b_l_s_w1 = torch.repeat_interleave(matrix_b_t_s_w1, rep, dim=1)

matrix_a_l_s_w2 = torch.repeat_interleave(matrix_a_w2, rep, dim=1)
matrix_b_t_s_w2 = matrix_b.transpose(0,1)
matrix_b_l_s_w2 = torch.repeat_interleave(matrix_b_t_s_w2, rep, dim=1)


matrix_a_c_s = torch.ones(in1, in2 * rep) * 255
matrix_b_c_s = torch.ones(out, in2 * rep) * 255

# Optical parameters

# wavelength method1
# wavelength = 0.532e-6  # 532 nm
# wavelength1 = 0.532e-6 * 10 / 11
# wavelength2 = 0.532e-6 * 10 / 9

# wavelength method2
wavelength = 0.350e-6 * 2  # 532 nm
wavelength1 = 0.350e-6
wavelength2 = 0.350e-6 * 3

z = 200e-3  # Propagation distance (200 mm)
px = 3.6e-6 # Pixel size of SLM (3.6 μm)
grid_size_r = (rep * in2) # Resolution (r)
grid_size_c = (max(in1, out)) # Resolution (c)
L_r = grid_size_r * px  # Row
L_c = grid_size_c * px  # Col
L = max(L_r, L_c) # Window size

k = 2 * torch.pi / wavelength
k1 = 2 * torch.pi / wavelength1
k2 = 2 * torch.pi / wavelength2

# Create spatial coordinates and phase
sep1 = 10
sep = int(4000 / out)

x = torch.linspace(-L / 2, L / 2, grid_size)
linear_phase = torch.zeros(in1 * (sep1), grid_size).to(torch.complex64)

matrix_a_l = torch.zeros(in1 * (sep1), in2 * rep)
matrix_b_l = torch.zeros(out * (sep), in2 * rep)
matrix_a_l_w1 = torch.zeros(in1 * (sep1), in2 * rep)
matrix_b_l_w1 = torch.zeros(out * (sep), in2 * rep)
matrix_a_l_w2 = torch.zeros(in1 * (sep1), in2 * rep)
matrix_b_l_w2 = torch.zeros(out * (sep), in2 * rep)

matrix_a_c = torch.zeros(in1 * (sep1), in2 * rep)
matrix_b_c = torch.zeros(out * (sep), in2 * rep)
for i in range(in1):
    # linear_phase[sep1 * i,:] = torch.exp(1j * (2 * torch.pi * (i + 1) / (2 * px * (in1 + 1)) * x)) # Linear phases method1
    linear_phase[sep1 * i,:] = torch.exp(1j * (2 * torch.pi * ((i + 1) / (20 * px * (in1 + 1)) + 2 / (2 * px * (in1 + 1))) * x))  # Linear phases method2
    matrix_a_l[sep1 * i, :] = matrix_a_l_s[i, :]
    matrix_a_l_w1[sep1 * i, :] = matrix_a_l_s_w1[i, :]
    matrix_a_l_w2[sep1 * i, :] = matrix_a_l_s_w2[i, :]

    matrix_a_c[sep1 * i, :] = matrix_a_c_s[i, :]

for j in range(out):
    matrix_b_l[sep * j, :] = matrix_b_l_s[j, :]
    matrix_b_l_w1[sep * j, :] = matrix_b_l_s_w1[j, :]
    matrix_b_l_w2[sep * j, :] = matrix_b_l_s_w2[j, :]
    matrix_b_c[sep * j, :] = matrix_b_c_s[j, :]

matrix_a_lp = matrix_a_l * linear_phase # Phase modulation
matrix_a_lp_w1 = matrix_a_l_w1 * linear_phase # Phase modulation
matrix_a_lp_w2 = matrix_a_l_w2 * linear_phase # Phase modulation
matrix_a_cp = matrix_a_c * linear_phase # Phase modulation (All ones matrix)

a_rate = 3 # Aperture expended rate
L_full = L * a_rate
grid_size_full = int(grid_size * a_rate)
x_full = torch.linspace(-L_full / 2, L_full / 2, grid_size_full)
y_full = torch.linspace(-L_full / 2, L_full / 2, grid_size_full)
X, Y = torch.meshgrid(x_full, y_full)
X = torch.transpose(X,0 ,1)
Y = torch.transpose(Y,0 ,1)


lens_phase_1 = torch.exp(-1j * k * (Y**2) / (2 * z))
lens_phase_2 = torch.exp(-1j * k * (X**2) / (2 * z))

lens_phase_1_w1 = torch.exp(-1j * k1 * (Y**2) / (2 * z))
lens_phase_2_w1 = torch.exp(-1j * k1 * (X**2) / (2 * z))

lens_phase_1_w2 = torch.exp(-1j * k2 * (Y**2) / (2 * z))
lens_phase_2_w2 = torch.exp(-1j * k2 * (X**2) / (2 * z))



input_pad_num_row = math.floor((L / px * a_rate - matrix_a_lp.shape[-1])/2)
input_pad_num_col = math.floor((L / px * a_rate - matrix_a_lp.shape[-0])/2)
weight_pad_num_row = math.floor((L / px * a_rate - matrix_b_l.shape[-1])/2)
weight_pad_num_col = math.floor((L / px * a_rate - matrix_b_l.shape[-0])/2)

pad_input = nn.ZeroPad2d(padding=(input_pad_num_row, input_pad_num_row, input_pad_num_col, input_pad_num_col))
U_in = pad_input(matrix_a_lp)
U_in_w1 = pad_input(matrix_a_lp_w1)
U_in_w2 = pad_input(matrix_a_lp_w2)
U_in_c = pad_input(matrix_a_cp)

pad_weight = nn.ZeroPad2d(padding=(input_pad_num_row, input_pad_num_row, weight_pad_num_col, weight_pad_num_col))
weight = pad_weight(matrix_b_l)
weight_w1 = pad_weight(matrix_b_l_w1)
weight_w2 = pad_weight(matrix_b_l_w2)
weight_c = pad_weight(matrix_b_c)

# Wavelength 0
U_out1 = rayleigh_sommerfeld(U_in, wavelength, z, L_full,0)
U_out2 = rayleigh_sommerfeld(U_out1 * lens_phase_1, wavelength, z, L_full,0)
U_out3 = rayleigh_sommerfeld(U_out2 * weight, wavelength, z, L_full,1)
U_out4 = rayleigh_sommerfeld(U_out3 * lens_phase_2, wavelength, z, L_full,1)

U_out1_c = rayleigh_sommerfeld(U_in_c, wavelength, z, L_full,0)
U_out2_c = rayleigh_sommerfeld(U_out1_c * lens_phase_1, wavelength, z, L_full,0)
U_out3_c = rayleigh_sommerfeld(U_out2_c * weight_c, wavelength, z, L_full,1)
U_out4_c = rayleigh_sommerfeld(U_out3_c * lens_phase_2, wavelength, z, L_full,1)

# Wavelength 1
U_out1_w1 = rayleigh_sommerfeld(U_in_w1, wavelength1, z, L_full,0)
U_out2_w1 = rayleigh_sommerfeld(U_out1_w1 * lens_phase_1_w1, wavelength1, z, L_full,0)
U_out3_w1 = rayleigh_sommerfeld(U_out2_w1 * weight_w1, wavelength1, z, L_full,1)
U_out4_w1 = rayleigh_sommerfeld(U_out3_w1 * lens_phase_2_w1, wavelength1, z, L_full,1)

U_out1_c_w1 = rayleigh_sommerfeld(U_in_c, wavelength1, z, L_full,0)
U_out2_c_w1 = rayleigh_sommerfeld(U_out1_c_w1 * lens_phase_1_w1, wavelength1, z, L_full,0)
U_out3_c_w1 = rayleigh_sommerfeld(U_out2_c_w1 * weight_c, wavelength1, z, L_full,1)
U_out4_c_w1 = rayleigh_sommerfeld(U_out3_c_w1 * lens_phase_2_w1, wavelength1, z, L_full,1)

# Wavelength 2
U_out1_w2 = rayleigh_sommerfeld(U_in_w2, wavelength2, z, L_full,0)
U_out2_w2 = rayleigh_sommerfeld(U_out1_w2 * lens_phase_1_w2, wavelength2, z, L_full,0)
U_out3_w2 = rayleigh_sommerfeld(U_out2_w2 * weight_w2, wavelength2, z, L_full,1)
U_out4_w2 = rayleigh_sommerfeld(U_out3_w2 * lens_phase_2_w2, wavelength2, z, L_full,1)

U_out1_c_w2 = rayleigh_sommerfeld(U_in_c, wavelength2, z, L_full,0)
U_out2_c_w2 = rayleigh_sommerfeld(U_out1_c_w2 * lens_phase_1_w2, wavelength2, z, L_full,0)
U_out3_c_w2 = rayleigh_sommerfeld(U_out2_c_w2 * weight_c, wavelength2, z, L_full,1)
U_out4_c_w2 = rayleigh_sommerfeld(U_out3_c_w2 * lens_phase_2_w2, wavelength2, z, L_full,1)


# Compute intensity

U_out4_w012 = U_out4 + U_out4_w1 + U_out4_w2
U_out4_c_w012 = U_out4_c + U_out4_c_w1 + U_out4_c_w2

I_out = torch.abs(U_out4_w012)
I_out_a = I_out.numpy()

I_out_c = torch.abs(U_out4_c_w012)
I_out_c_a = I_out_c.numpy()

plt.plot(figsize=(6, 5))
aa = int(I_out.shape[0] / 2 - sep) # choose the layer of the result tensor
plt.plot(I_out_a[aa,:])
plt.xticks(
    ticks=np.linspace(0, grid_size_full,5),
    labels=np.linspace(-L_full / 2 * 1e3, L_full / 2 * 1e3,5)
)
plt.ylabel('Amplitude')
plt.xlabel("Position (mm)")
plt.show()


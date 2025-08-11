import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from Rayleigh_Sommerfeld_Diffraction_1d import rayleigh_sommerfeld  # Import the diffraction function
from Rayleigh_Sommerfeld_Diffraction import rayleigh_sommerfeld_2d  # Import the diffraction function
from Rayleigh_Sommerfeld_Diffraction import get_the_max

# Matrix parameters
in1 = 20
in2 = 20
out = 20
torch.manual_seed(64)
matrix_a = torch.randint(0, 255, (in1, in2)).to(torch.float32)
torch.manual_seed(32)
matrix_b = torch.randint(0, 255, (in2, out)).to(torch.float32)
matrix_c = torch.matmul(matrix_a, matrix_b)
max_value = torch.max(matrix_c)
matrix_c_nor = (matrix_c / max_value)
matrix_c_int16 = (matrix_c_nor * 65535).to(torch.uint16)

# Matrix on spatial light
grid_size = 4000
rep = int(np.floor(grid_size / in2))
matrix_a_l_s = torch.repeat_interleave(matrix_a, rep, dim=1)
matrix_b_t_s = matrix_b.transpose(0,1)
matrix_b_l_s = torch.repeat_interleave(matrix_b_t_s, rep, dim=1)

matrix_a_c_s = torch.ones(in1, in2 * rep) * 255
matrix_b_c_s = torch.ones(out, in2 * rep) * 255

# Optical parameters
wavelength = 0.532e-6  # 532 nm
z = 200e-3  # Propagation distance (200 mm)
px = 3.6e-6 # Pixel size of SLM (3.6 μm)
grid_size_r = (rep * in2) # Resolution (r)
grid_size_c = (max(in1, out)) # Resolution (c)
L_r = grid_size_r * px  # Row
L_c = grid_size_c * px  # Col
L = max(L_r, L_c) # Window size

k = 2 * torch.pi / wavelength

# Create spatial coordinates and phase
sep1 = 10
sep = int(4000 / out)

x = torch.linspace(-L / 2, L / 2, grid_size)
linear_phase = torch.zeros(in1 * (sep1), grid_size).to(torch.complex64)

matrix_a_l = torch.zeros(in1 * (sep1), in2 * rep)
matrix_b_l = torch.zeros(out * (sep), in2 * rep)
matrix_a_c = torch.zeros(in1 * (sep1), in2 * rep)
matrix_b_c = torch.zeros(out * (sep), in2 * rep)
for i in range(in1):
    linear_phase[sep1 * i,:] = torch.exp(1j * (2 * torch.pi * i / (2 * px * (in1)) * x)) # Linear phases
    matrix_a_l[sep1 * i, :] = matrix_a_l_s[i, :]
    matrix_a_c[sep1 * i, :] = matrix_a_c_s[i, :]

for j in range(out):
    matrix_b_l[sep * j, :] = matrix_b_l_s[j, :]
    matrix_b_c[sep * j, :] = matrix_b_c_s[j, :]

matrix_a_lp = matrix_a_l * linear_phase # Phase modulation
matrix_a_cp = matrix_a_c * linear_phase # Phase modulation (All ones matrix)

a_rate = 3 # Aperture expended rate
L_full = L * a_rate
grid_size_full = int(grid_size * a_rate)
x_full = torch.linspace(-L_full / 2, L_full / 2, grid_size_full)
y_full = torch.linspace(-L_full / 2, L_full / 2, grid_size_full)
X, Y = torch.meshgrid(x_full, y_full)
X = torch.transpose(X,0 ,1)
Y = torch.transpose(Y,0 ,1)

# lens_phase = torch.exp(-1j * k * (X**2 + Y**2) / (2 * z))
lens_phase_1 = torch.exp(-1j * k * (Y**2) / (2 * z))
lens_phase_2 = torch.exp(-1j * k * (X**2) / (2 * z))


input_pad_num_row = math.floor((L / px * a_rate - matrix_a_lp.shape[-1])/2)
input_pad_num_col = math.floor((L / px * a_rate - matrix_a_lp.shape[-0])/2)
weight_pad_num_row = math.floor((L / px * a_rate - matrix_b_l.shape[-1])/2)
weight_pad_num_col = math.floor((L / px * a_rate - matrix_b_l.shape[-0])/2)

pad_input = nn.ZeroPad2d(padding=(input_pad_num_row, input_pad_num_row, input_pad_num_col, input_pad_num_col))
U_in = pad_input(matrix_a_lp) # Optical input with aperture
U_in_c = pad_input(matrix_a_cp) # Optical input with aperture (All ones matrix)

pad_weight = nn.ZeroPad2d(padding=(input_pad_num_row, input_pad_num_row, weight_pad_num_col, weight_pad_num_col))
weight = pad_weight(matrix_b_l) # Optical weight with aperture
weight_c = pad_weight(matrix_b_c) # Optical weight with aperture

# Compute diffraction using Rayleigh–Sommerfeld method
U_out1 = rayleigh_sommerfeld(U_in, wavelength, z, L_full,0)
U_out2 = rayleigh_sommerfeld(U_out1 * lens_phase_1, wavelength, z, L_full,0)
U_out3 = rayleigh_sommerfeld(U_out2 * weight, wavelength, z, L_full,1)
U_out4 = rayleigh_sommerfeld(U_out3 * lens_phase_2, wavelength, z, L_full,1)
del [U_out1,U_out2,U_out3]

U_out1_c = rayleigh_sommerfeld(U_in_c, wavelength, z, L_full,0)
U_out2_c = rayleigh_sommerfeld(U_out1_c * lens_phase_1, wavelength, z, L_full,0)
U_out3_c = rayleigh_sommerfeld(U_out2_c * weight_c, wavelength, z, L_full,1)
U_out4_c = rayleigh_sommerfeld(U_out3_c * lens_phase_2, wavelength, z, L_full,1)
del [U_out1_c,U_out2_c,U_out3_c]

# Compute intensity
I_out = torch.abs(U_out4)
I_out_a = I_out.numpy()

I_out_c = torch.abs(U_out4_c)
I_out_c_a = I_out_c.numpy()

plt.figure(figsize=(8, 8))
plt.imshow(I_out_c_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Raw optical field of weights')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(I_out_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Raw optical field of POMMM')
plt.show()

# Comparison
Temp_c = I_out_c[int(grid_size_full / 2 - out * sep / 2 - sep / 2): int(grid_size_full / 2 + (out / 2 - 1) * sep + sep / 2),
         int(grid_size_full / 2 - math.floor(wavelength * z / (px**2) / 2 / (in1) * 1 / 2)) : int(grid_size_full / 2 + math.floor(wavelength * z / (px**2) / 2 / (in1) * (in1 - 1 / 2)))]
Weight_temp = get_the_max(Temp_c, out, in1)
Weight_c = Weight_temp / torch.max(Weight_temp)

Temp = I_out[int(grid_size_full / 2 - out * sep / 2 - sep / 2): int(grid_size_full / 2 + out * sep / 2 - sep / 2),
         int(grid_size_full / 2 - math.floor(wavelength * z / (px**2) / 2 / (in1) * 1 / 2)) : int(grid_size_full / 2 + math.floor(wavelength * z / (px**2) / 2 / (in1) * (in1 - 1 / 2)))]
Pommm_temp = get_the_max(Temp, out, in1)
Matrix_Pommm = Pommm_temp / torch.max(Pommm_temp)

Res = (Matrix_Pommm / Weight_c).transpose(0,1)
Res1 = Res / torch.max(Res)

Res1_a = Res1.numpy()
Weight_c_a = Weight_c.numpy()


plt.figure(figsize=(8, 8))
plt.imshow(Weight_c_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Weights')
plt.show()
#
plt.figure(figsize=(8, 8))
plt.imshow(Res1_a, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('POMMM Results')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(matrix_c_nor, cmap='hot')
plt.colorbar(label='Intensity (a.u.)')
plt.title('Standard MMM Results')
plt.show()


err = torch.abs(Res1 - matrix_c_nor)/ (matrix_c_nor+1e-8)
ave_err = torch.mean(err)

print(ave_err)


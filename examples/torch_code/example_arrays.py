import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Functions ===
def update_1d_array(x):
    x[int(1)] = 3
    return x

def update_3d_array(x):
    x[int(1), int(1), int(1)] = 3
    return x

def update_2d_array(x):
    x[int(1), int(1)] = 3
    return x

# === Program ===
x = torch.tensor([1, 2, 3, 5, 6, 7], device=DEVICE)
y = (x[int(0):int(3)] + x[int(0):int(3)])
z = (y + torch.tensor([1, 3, 4], device=DEVICE))
print(x)
print(y)
print(z)
u0 = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.75, 0.75, 0.0], [0.0, 0.75, 0.75, 10.0], [0.0, 0.0, 0.0, 0.0]], device=DEVICE)
u00 = u0[int(0), int(0)]
u01 = u0[int(0), int(1)]
u02 = u0[int(0), int(2)]
u03 = u0[int(0), int(3)]
u10 = u0[int(1), int(0)]
u11 = u0[int(1), int(1)]
u12 = u0[int(1), int(2)]
u13 = u0[int(1), int(3)]
u20 = u0[int(2), int(0)]
u21 = u0[int(2), int(1)]
u22 = u0[int(2), int(2)]
u23 = u0[int(2), int(3)]
u30 = u0[int(3), int(0)]
u31 = u0[int(3), int(1)]
u32 = u0[int(3), int(2)]
u33 = u0[int(3), int(3)]
A = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], device=DEVICE)
A00 = A[int(0), int(0)]
A11 = A[int(1), int(1)]
A22 = A[int(2), int(2)]
T = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], device=DEVICE)
T0 = T[int(0)]
T12 = T[int(1), int(2)]
T000 = T[int(0), int(0), int(0)]
T123 = T[int(1), int(2), int(3)]
T012 = T[int(0), int(1), int(2)]
prog_1d = torch.tensor([1.0, 1.0], device=DEVICE)
prog_2d = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=DEVICE)
prog_3d = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], device=DEVICE)
prog_1d[int(1)] = 2
prog_2d[int(1), int(1)] = 2
prog_3d[int(1), int(1), int(1)] = 2
print(prog_1d)
print(prog_2d)
print(prog_3d)
func_1d = torch.tensor([1.0, 1.0], device=DEVICE)
func_2d = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=DEVICE)
func_3d = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], device=DEVICE)
print(update_1d_array(func_1d))
print(update_2d_array(func_2d))
print(update_3d_array(func_3d))
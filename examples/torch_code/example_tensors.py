import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import print

# === Program ===
v = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
print(v)
omega = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
print(omega)
T = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], device=DEVICE)
v = torch.tensor([[1.0], [2.0], [3.0]], device=DEVICE)
w = (T @ v)
print(w)
T = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], device=DEVICE)
v = torch.tensor([[1.0], [2.0], [3.0]], device=DEVICE)
w = (T @ v)
print(w)
print(T)
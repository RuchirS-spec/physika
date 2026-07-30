import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Program ===
hbar = 1.0
mass = 1.0
angular_frequency = 1.0
pi = 3.141592653589793
N_levels = 5
x_max = 6.0
N_grid = 601
dx = ((2.0 * x_max) / (N_grid - 1))
position = torch.stack([((-x_max) + (i * dx)) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
potential = torch.stack([(((((0.5 * mass) * angular_frequency) * angular_frequency) * position[int(i)]) * position[int(i)]) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
wavefunctions = torch.stack([torch.stack([((n + i) * 0.0) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
hamiltonian_wavefunctions = torch.stack([torch.stack([((n + i) * 0.0) for _fi_i in range(int(N_grid)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]]) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
energies = torch.stack([(n * 0.0) for _fi_n in range(int(N_levels)) for n in [torch.tensor(float(_fi_n), device=DEVICE)]])
x = 0.0
gaussian = 0.0
n_real = 0.0
next_n_real = 0.0
first_coefficient = 0.0
second_coefficient = 0.0
second_derivative = 0.0
kinetic_part = 0.0
potential_part = 0.0
energy_numerator = 0.0
normalization_integral = 0.0
normalization_constant = (1.0 / (pi ** 0.25))
for i in range(int(0), int(N_grid)):
    x = position[int(i)]
    gaussian = torch.exp((((-0.5) * x) * x) if isinstance((((-0.5) * x) * x), torch.Tensor) else torch.tensor(float((((-0.5) * x) * x))))
    wavefunctions[int(0), int(i)] = (normalization_constant * gaussian)
one_level = 1
if N_levels > one_level:
    for i in range(int(0), int(N_grid)):
        x = position[int(i)]
        wavefunctions[int(1), int(i)] = ((torch.sqrt(2.0 if isinstance(2.0, torch.Tensor) else torch.tensor(float(2.0))) * x) * wavefunctions[int(0), int(i)])
two_levels = 2
if N_levels > two_levels:
    for n in range(int(1), int((N_levels - 1))):
        n_real = (n * 1.0)
        next_n_real = (n_real + 1.0)
        first_coefficient = torch.sqrt((2.0 / next_n_real) if isinstance((2.0 / next_n_real), torch.Tensor) else torch.tensor(float((2.0 / next_n_real))))
        second_coefficient = torch.sqrt((n_real / next_n_real) if isinstance((n_real / next_n_real), torch.Tensor) else torch.tensor(float((n_real / next_n_real))))
        for i in range(int(0), int(N_grid)):
            x = position[int(i)]
            wavefunctions[int((n + 1)), int(i)] = (((first_coefficient * x) * wavefunctions[int(n), int(i)]) - (second_coefficient * wavefunctions[int((n - 1)), int(i)]))
for n in range(int(0), int(N_levels)):
    for i in range(int(1), int((N_grid - 1))):
        second_derivative = (((wavefunctions[int(n), int((i + 1))] - (2.0 * wavefunctions[int(n), int(i)])) + wavefunctions[int(n), int((i - 1))]) / (dx * dx))
        kinetic_part = (((-(hbar * hbar)) / (2.0 * mass)) * second_derivative)
        potential_part = (potential[int(i)] * wavefunctions[int(n), int(i)])
        hamiltonian_wavefunctions[int(n), int(i)] = (kinetic_part + potential_part)
for n in range(int(0), int(N_levels)):
    energy_numerator = 0.0
    normalization_integral = 0.0
    for i in range(int(1), int((N_grid - 1))):
        energy_numerator = (energy_numerator + ((wavefunctions[int(n), int(i)] * hamiltonian_wavefunctions[int(n), int(i)]) * dx))
        normalization_integral = (normalization_integral + ((wavefunctions[int(n), int(i)] * wavefunctions[int(n), int(i)]) * dx))
    energies[int(n)] = (energy_numerator / normalization_integral)
    physika_print(energies[int(n)])

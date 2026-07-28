import torch
import torch.nn as nn
import torch.optim as optim
from physika.runtime import DEVICE

from physika.runtime import physika_print

# === Functions ===
def zero_array(length):
    values = torch.stack([(i * 0.0) for _fi_i in range(int(length)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return values

def zero_matrix(rows, columns):
    values = torch.stack([torch.stack([((i + j) * 0.0) for _fi_j in range(int(columns)) for j in [torch.tensor(float(_fi_j), device=DEVICE)]]) for _fi_i in range(int(rows)) for i in [torch.tensor(float(_fi_i), device=DEVICE)]])
    return values

def linspace(start, end, number):
    values = zero_array(number)
    spacing = ((end - start) / (number - 1))
    for i in range(int(0), int(number)):
        values[int(i)] = (start + (i * spacing))
    return values

def integrate(values, grid_spacing, number):
    integral = 0.0
    for i in range(int(0), int(number)):
        integral = (integral + (values[int(i)] * grid_spacing))
    return integral

def dot_product(first, second, number):
    value = 0.0
    for i in range(int(0), int(number)):
        value = (value + (first[int(i)] * second[int(i)]))
    return value

def normalize_vector(values, number):
    result = zero_array(number)
    norm = torch.sqrt(dot_product(values, values, number) if isinstance(dot_product(values, values, number), torch.Tensor) else torch.tensor(float(dot_product(values, values, number))))
    for i in range(int(0), int(number)):
        result[int(i)] = (values[int(i)] / norm)
    return result

def apply_hamiltonian(wavefunction, potential, kinetic_coefficient, number):
    result = zero_array(number)
    result[int(0)] = ((((2.0 * kinetic_coefficient) + potential[int(0)]) * wavefunction[int(0)]) - (kinetic_coefficient * wavefunction[int(1)]))
    for i in range(int(1), int((number - 1))):
        result[int(i)] = ((((-kinetic_coefficient) * wavefunction[int((i - 1))]) + (((2.0 * kinetic_coefficient) + potential[int(i)]) * wavefunction[int(i)])) - (kinetic_coefficient * wavefunction[int((i + 1))]))
    result[int((number - 1))] = (((-kinetic_coefficient) * wavefunction[int((number - 2))]) + (((2.0 * kinetic_coefficient) + potential[int((number - 1))]) * wavefunction[int((number - 1))]))
    return result

def jacobi_diagonalize():
    jacobi_not_converged = 0
    jacobi_converged = 1
    jacobi_converged_local = jacobi_not_converged
    for i in range(int(0), int(Krylov_dimension)):
        Ritz_vectors[int(i), int(i)] = 1.0
        for j in range(int(0), int(Krylov_dimension)):
            projected_work_matrix[int(i), int(j)] = projected_hamiltonian[int(i), int(j)]
    for rotation in range(int(0), int(jacobi_maximum_rotations)):
        jacobi_p = 0
        jacobi_q = 1
        jacobi_largest = torch.abs(projected_work_matrix[int(0), int(1)] if isinstance(projected_work_matrix[int(0), int(1)], torch.Tensor) else torch.tensor(float(projected_work_matrix[int(0), int(1)])))
        for i in range(int(0), int(Krylov_dimension)):
            for j in range(int((i + 1)), int(Krylov_dimension)):
                if torch.abs(projected_work_matrix[int(i), int(j)] if isinstance(projected_work_matrix[int(i), int(j)], torch.Tensor) else torch.tensor(float(projected_work_matrix[int(i), int(j)]))) > jacobi_largest:
                    jacobi_largest = torch.abs(projected_work_matrix[int(i), int(j)] if isinstance(projected_work_matrix[int(i), int(j)], torch.Tensor) else torch.tensor(float(projected_work_matrix[int(i), int(j)])))
                    jacobi_p = i
                    jacobi_q = j
        if jacobi_converged_local == jacobi_not_converged:
            if jacobi_largest <= jacobi_tolerance:
                jacobi_converged_local = jacobi_converged
        if jacobi_largest > jacobi_tolerance:
            jacobi_angle = (0.5 * atan2((2.0 * projected_work_matrix[int(jacobi_p), int(jacobi_q)]), (projected_work_matrix[int(jacobi_q), int(jacobi_q)] - projected_work_matrix[int(jacobi_p), int(jacobi_p)])))
            jacobi_cosine = torch.cos(jacobi_angle if isinstance(jacobi_angle, torch.Tensor) else torch.tensor(float(jacobi_angle)))
            jacobi_sine = torch.sin(jacobi_angle if isinstance(jacobi_angle, torch.Tensor) else torch.tensor(float(jacobi_angle)))
            jacobi_app = (projected_work_matrix[int(jacobi_p), int(jacobi_p)] + 0.0)
            jacobi_aqq = (projected_work_matrix[int(jacobi_q), int(jacobi_q)] + 0.0)
            jacobi_apq = (projected_work_matrix[int(jacobi_p), int(jacobi_q)] + 0.0)
            for k in range(int(0), int(Krylov_dimension)):
                if k != jacobi_p:
                    if k != jacobi_q:
                        jacobi_akp = (projected_work_matrix[int(k), int(jacobi_p)] + 0.0)
                        jacobi_akq = (projected_work_matrix[int(k), int(jacobi_q)] + 0.0)
                        projected_work_matrix[int(k), int(jacobi_p)] = ((jacobi_cosine * jacobi_akp) - (jacobi_sine * jacobi_akq))
                        projected_work_matrix[int(jacobi_p), int(k)] = projected_work_matrix[int(k), int(jacobi_p)]
                        projected_work_matrix[int(k), int(jacobi_q)] = ((jacobi_sine * jacobi_akp) + (jacobi_cosine * jacobi_akq))
                        projected_work_matrix[int(jacobi_q), int(k)] = projected_work_matrix[int(k), int(jacobi_q)]
            projected_work_matrix[int(jacobi_p), int(jacobi_p)] = ((((jacobi_cosine * jacobi_cosine) * jacobi_app) - (((2.0 * jacobi_sine) * jacobi_cosine) * jacobi_apq)) + ((jacobi_sine * jacobi_sine) * jacobi_aqq))
            projected_work_matrix[int(jacobi_q), int(jacobi_q)] = ((((jacobi_sine * jacobi_sine) * jacobi_app) + (((2.0 * jacobi_sine) * jacobi_cosine) * jacobi_apq)) + ((jacobi_cosine * jacobi_cosine) * jacobi_aqq))
            projected_work_matrix[int(jacobi_p), int(jacobi_q)] = 0.0
            projected_work_matrix[int(jacobi_q), int(jacobi_p)] = 0.0
            for k in range(int(0), int(Krylov_dimension)):
                jacobi_vkp = (Ritz_vectors[int(k), int(jacobi_p)] + 0.0)
                jacobi_vkq = (Ritz_vectors[int(k), int(jacobi_q)] + 0.0)
                Ritz_vectors[int(k), int(jacobi_p)] = ((jacobi_cosine * jacobi_vkp) - (jacobi_sine * jacobi_vkq))
                Ritz_vectors[int(k), int(jacobi_q)] = ((jacobi_sine * jacobi_vkp) + (jacobi_cosine * jacobi_vkq))
    for i in range(int(0), int(Krylov_dimension)):
        Ritz_values[int(i)] = projected_work_matrix[int(i), int(i)]
    for i in range(int(0), int((Krylov_dimension - 1))):
        jacobi_minimum = i
        for j in range(int((i + 1)), int(Krylov_dimension)):
            if Ritz_values[int(j)] < Ritz_values[int(jacobi_minimum)]:
                jacobi_minimum = j
        if jacobi_minimum != i:
            jacobi_temporary_value = (Ritz_values[int(i)] + 0.0)
            Ritz_values[int(i)] = Ritz_values[int(jacobi_minimum)]
            Ritz_values[int(jacobi_minimum)] = jacobi_temporary_value
            for k in range(int(0), int(Krylov_dimension)):
                jacobi_temporary_vector = (Ritz_vectors[int(k), int(i)] + 0.0)
                Ritz_vectors[int(k), int(i)] = Ritz_vectors[int(k), int(jacobi_minimum)]
                Ritz_vectors[int(k), int(jacobi_minimum)] = jacobi_temporary_vector
    return jacobi_converged_local

# === Program ===
atomic_mass_unit = 1.6605390666e-27
hbar = 1.054571817e-34
planck_constant = 6.62607015e-34
speed_of_light_cm = 29979245800.0
electron_volt = 1.602176634e-19
mass_H_u = 1.00784
mass_Cl_u = 35.45
mass_H = (mass_H_u * atomic_mass_unit)
mass_Cl = (mass_Cl_u * atomic_mass_unit)
reduced_mass_HCl = ((mass_H * mass_Cl) / (mass_H + mass_Cl))
dissociation_energy_eV = 4.61907
dissociation_energy_J = (dissociation_energy_eV * electron_volt)
equilibrium_distance = 1.2746e-10
morse_a = 18680000000.0
N_grid = 100
N_levels = 2
block_size = 2
Krylov_dimension = 40
r_min = 5e-11
r_max = 2e-10
grid_spacing = ((r_max - r_min) / (N_grid + 1))
r_start = (r_min + grid_spacing)
r_end = (r_max - grid_spacing)
bond_distance = linspace(r_start, r_end, N_grid)
distance_from_equilibrium = (bond_distance - equilibrium_distance)
morse_exponential = torch.exp(((-morse_a) * distance_from_equilibrium) if isinstance(((-morse_a) * distance_from_equilibrium), torch.Tensor) else torch.tensor(float(((-morse_a) * distance_from_equilibrium))))
morse_difference = (1.0 - morse_exponential)
potential_J = ((dissociation_energy_J * morse_difference) * morse_difference)
potential_eV = (potential_J / electron_volt)
kinetic_coefficient_J = ((hbar / grid_spacing) * (hbar / ((2.0 * reduced_mass_HCl) * grid_spacing)))
kinetic_coefficient_eV = (kinetic_coefficient_J / electron_volt)
trial_width = 2e-11
gaussian_argument = (distance_from_equilibrium / trial_width)
gaussian_envelope = torch.exp(((-gaussian_argument) * gaussian_argument) if isinstance(((-gaussian_argument) * gaussian_argument), torch.Tensor) else torch.tensor(float(((-gaussian_argument) * gaussian_argument))))
Krylov_basis = zero_matrix(Krylov_dimension, N_grid)
H_Krylov = zero_matrix(Krylov_dimension, N_grid)
projected_hamiltonian = zero_matrix(Krylov_dimension, Krylov_dimension)
projected_work_matrix = zero_matrix(Krylov_dimension, Krylov_dimension)
Ritz_vectors = zero_matrix(Krylov_dimension, Krylov_dimension)
Ritz_values = zero_array(Krylov_dimension)
jacobi_tolerance = 1e-07
jacobi_maximum_rotations = 10000
jacobi_largest = 0.0
jacobi_angle = 0.0
jacobi_cosine = 0.0
jacobi_sine = 0.0
jacobi_app = 0.0
jacobi_aqq = 0.0
jacobi_apq = 0.0
jacobi_akp = 0.0
jacobi_akq = 0.0
jacobi_vkp = 0.0
jacobi_vkq = 0.0
jacobi_temporary_value = 0.0
jacobi_temporary_vector = 0.0
jacobi_p = 0
jacobi_q = 1
jacobi_minimum = 0
candidate = zero_array(N_grid)
overlap = 0.0
candidate_norm = 0.0
for i in range(int(0), int(N_grid)):
    Krylov_basis[int(0), int(i)] = gaussian_envelope[int(i)]
Krylov_basis[int(0)] = normalize_vector(Krylov_basis[int(0)], N_grid)
for q_index in range(int(1), int(Krylov_dimension)):
    if q_index < block_size:
        for i in range(int(0), int(N_grid)):
            candidate[int(i)] = (gaussian_argument[int(i)] * Krylov_basis[int((q_index - 1)), int(i)])
    else:
        candidate = apply_hamiltonian(Krylov_basis[int((q_index - block_size))], potential_eV, kinetic_coefficient_eV, N_grid)
    for orthogonalization_pass in range(int(0), int(2)):
        for lower in range(int(0), int(q_index)):
            overlap = dot_product(Krylov_basis[int(lower)], candidate, N_grid)
            for i in range(int(0), int(N_grid)):
                candidate[int(i)] = (candidate[int(i)] - (overlap * Krylov_basis[int(lower), int(i)]))
    candidate_norm = torch.sqrt(dot_product(candidate, candidate, N_grid) if isinstance(dot_product(candidate, candidate, N_grid), torch.Tensor) else torch.tensor(float(dot_product(candidate, candidate, N_grid))))
    for i in range(int(0), int(N_grid)):
        Krylov_basis[int(q_index), int(i)] = (candidate[int(i)] / candidate_norm)
for q_index in range(int(0), int(Krylov_dimension)):
    H_Krylov[int(q_index)] = apply_hamiltonian(Krylov_basis[int(q_index)], potential_eV, kinetic_coefficient_eV, N_grid)
for row in range(int(0), int(Krylov_dimension)):
    for column in range(int(row), int(Krylov_dimension)):
        projected_hamiltonian[int(row), int(column)] = dot_product(Krylov_basis[int(row)], H_Krylov[int(column)], N_grid)
        projected_hamiltonian[int(column), int(row)] = projected_hamiltonian[int(row), int(column)]
physika_print(jacobi_diagonalize())
vibrational_energies_eV = zero_array(N_levels)
psi_raw = zero_matrix(N_levels, N_grid)
for n in range(int(0), int(N_levels)):
    vibrational_energies_eV[int(n)] = Ritz_values[int(n)]
    for i in range(int(0), int(N_grid)):
        for q_index in range(int(0), int(Krylov_dimension)):
            psi_raw[int(n), int(i)] = (psi_raw[int(n), int(i)] + (Krylov_basis[int(q_index), int(i)] * Ritz_vectors[int(q_index), int(n)]))
    psi_raw[int(n)] = normalize_vector(psi_raw[int(n)], N_grid)
normalization_factor = zero_array(N_levels)
psi = zero_matrix(N_levels, N_grid)
for n in range(int(0), int(N_levels)):
    normalization_factor[int(n)] = torch.sqrt(integrate((psi_raw[int(n)] * psi_raw[int(n)]), grid_spacing, N_grid) if isinstance(integrate((psi_raw[int(n)] * psi_raw[int(n)]), grid_spacing, N_grid), torch.Tensor) else torch.tensor(float(integrate((psi_raw[int(n)] * psi_raw[int(n)]), grid_spacing, N_grid))))
    for i in range(int(0), int(N_grid)):
        psi[int(n), int(i)] = (psi_raw[int(n), int(i)] / normalization_factor[int(n)])
transition_eV = (vibrational_energies_eV[int(1)] - vibrational_energies_eV[int(0)])
transition_J = (transition_eV * electron_volt)
wavenumber = (transition_J / (planck_constant * speed_of_light_cm))
wavelength_micrometer = (10000.0 / wavenumber)
physika_print(physika_print(vibrational_energies_eV[int(0)]))
physika_print(physika_print(vibrational_energies_eV[int(1)]))
physika_print(physika_print(transition_eV))
physika_print(physika_print(wavenumber))
physika_print(physika_print(wavelength_micrometer))
bond_distance_angstrom = (bond_distance / 1e-10)
equilibrium_distance_angstrom = (equilibrium_distance / 1e-10)
psi_angstrom = (torch.sqrt(1e-10 if isinstance(1e-10, torch.Tensor) else torch.tensor(float(1e-10))) * psi)

HCl Vibrational States in a Morse Potential
===========================================

This tutorial calculates the vibrational states of an HCl molecule using a one-dimensional Morse potential, which accounts for anharmonicity
and bond dissociation, making it more realistic than the simple harmonic-oscillator potential 
(`Kelly, 2023
<https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/05%3A_The_Harmonic_Oscillator_and_the_Rigid_Rotor/5.03%3A_The_Harmonic_Oscillator_Approximates_Vibrations>`_).

This tutorial demonstrates how to

#. Construct the Morse potential on a bond-distance grid.
#. Apply the differential Hamiltonian using finite differences.
#. Build an orthonormal block-Krylov basis
   (`Saad, 2011 <https://doi.org/10.1137/1.9781611970739>`_).
#. Construct and diagonalize the projected Hamiltonian using `Jacobi rotations
   <https://en.wikipedia.org/wiki/Jacobi_rotation>`_.
#. Reconstruct and normalize the lowest vibrational wavefunctions.
#. Calculate the :math:`v=0 \rightarrow v=1` transition energy and wavenumber.

Physical model
--------------

The vibrational motion is described by the
one-dimensional, time-independent Schrödinger equation

.. math::
   \hat H\psi_v(r)=E_v\psi_v(r),

where

* :math:`r` is the H--Cl internuclear distance,
* :math:`v` is the vibrational quantum number,
* :math:`\psi_v(r)` is the vibrational wavefunction, and
* :math:`E_v` is the corresponding vibrational energy.

Hamiltonian operator
~~~~~~~~~~~~~~~~~~~~

The original Hamiltonian is a continuous differential operator:

.. math::
   \hat H =
   -\frac{\hbar^2}{2\mu}\frac{d^2}{dr^2}+V(r).

The first term is the nuclear kinetic-energy operator, and the second term is
the potential-energy operator.

Reduced mass
^^^^^^^^^^^^

The relative vibrational motion of the H and Cl nuclei is described using the
reduced mass

.. math::
   \mu =
   \frac{m_{\mathrm H}m_{\mathrm{Cl}}}
        {m_{\mathrm H}+m_{\mathrm{Cl}}}.

Using ``mass_H_u = 1.00784`` and ``mass_Cl_u = 35.45`` gives

.. math::
   \mu = 1.6272938883\times 10^{-27}\ \mathrm{kg}.

Potential-energy operator
^^^^^^^^^^^^^^^^^^^^^^^^^

The H--Cl interaction is represented by the Morse potential

.. math::
   V(r)=D_e\left[1-\exp\left(-a(r-r_e)\right)\right]^2,

where :math:`D_e` is the Morse well depth, :math:`r_e` is the equilibrium
bond length, and :math:`a` determines the width and curvature of the potential well.

Finite-difference Hamiltonian representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve the equation numerically, the continuous wavefunction is replaced by
a vector of grid values, and the derivative is replaced by a finite-difference
formula.

At an interior point :math:`r_i`, the second derivative is approximated using
the central finite-difference expression

.. math::

   \left.
   \frac{d^2\psi}{dr^2}
   \right|_{r_i}
   \approx
   \frac{
   \psi_{i+1}-2\psi_i+\psi_{i-1}
   }{(\Delta r)^2}.

Substitution into the kinetic-energy operator gives

.. math::

   -\frac{\hbar^2}{2\mu}
   \left.
   \frac{d^2\psi}{dr^2}
   \right|_{r_i}
   \approx
   -C\psi_{i-1}
   +2C\psi_i
   -C\psi_{i+1},

where

.. math::

   C
   =
   \frac{\hbar^2}
        {2\mu(\Delta r)^2}.

Adding the potential-energy contribution gives

.. math::

   (H\psi)_i
   =
   -C\psi_{i-1}
   +
   \left(2C+V_i\right)\psi_i
   -
   C\psi_{i+1}.

This linear relation is mathematically equivalent to multiplication by the
tridiagonal matrix

.. math::

   H
   =
   \begin{pmatrix}
   2C+V_1 & -C & 0 & \cdots & 0\\
   -C & 2C+V_2 & -C & \ddots & \vdots\\
   0 & -C & 2C+V_3 & \ddots & 0\\
   \vdots & \ddots & \ddots & \ddots & -C\\
   0 & \cdots & 0 & -C & 2C+V_N
   \end{pmatrix}.

The diagonal elements contain both kinetic- and potential-energy
contributions:

.. math::

   H_{ii}=2C+V_i.

The immediately adjacent off-diagonal elements arise from the
finite-difference kinetic-energy operator:

.. math::

   H_{i,i-1}=H_{i,i+1}=-C.

All remaining elements are zero. Therefore,

.. math::

   H_{ij}
   =
   \begin{cases}
   2C+V_i, & i=j,\\
   -C, & |i-j|=1,\\
   0, & \text{otherwise}.
   \end{cases}

Boundary conditions
~~~~~~~~~~~~~~~~~~~

The coordinate interval is

.. math::

   r_{\min}=0.50\ \mathrm{\AA},
   \qquad
   r_{\max}=2.00\ \mathrm{\AA}.

The code stores ``N_grid = 100`` interior points.  The spacing is

.. math::

   \Delta r
   =
   \frac{r_{\max}-r_{\min}}{N_{\mathrm{grid}}+1}
   =
   \frac{2.00-0.50}{101}\ \mathrm{\AA}
   \approx0.0148515\ \mathrm{\AA}.

The stored coordinate runs from

.. math::

   r_1=r_{\min}+\Delta r

to

.. math::

   r_{N_{\mathrm{grid}}}=r_{\max}-\Delta r.

The wavefunction values are stored only at the interior grid points. Values
outside the interior grid are taken to be zero, which imposes Dirichlet
boundary conditions:

.. math::

   \psi(r_{\min})=0,

.. math::

   \psi(r_{\max})=0.

.. note::

   Higher vibrational states extend farther toward larger bond distances and
   contain more nodes. Therefore, ``r_max`` and ``N_grid`` should be increased
   to ensure that the wavefunctions vanish near the boundary and remain
   adequately resolved. Convergence should be checked by repeating the
   calculation with larger values.
   
   
Helper functions
----------------

All helper functions are defined before the main calculation.

Array and grid initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   def zero_array(length: ℕ): ℝ[m]:
       values: ℝ[length] = for i: ℕ(length) -> i * 0.0
       return values

   def zero_matrix(rows: ℕ, columns: ℕ): ℝ[m,n]:
       values: ℝ[rows,columns] = for i: ℕ(rows) -> for j: ℕ(columns) -> (i + j) * 0.0
       return values

   def linspace(start: ℝ, end: ℝ, number: ℕ): ℝ[m]:
       values: ℝ[number] = zero_array(number)
       spacing: ℝ = (end - start) / (number - 1)
       for i: ℕ(0, number):
           values[i] = start + i * spacing
       return values

Integration and vector operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   def integrate(values: ℝ[m], grid_spacing: ℝ, number: ℕ): ℝ:
       integral: ℝ = 0.0
       for i: ℕ(0, number):
           integral = integral + values[i] * grid_spacing
       return integral

   def dot_product(first: ℝ[m], second: ℝ[n], number: ℕ): ℝ:
       value: ℝ = 0.0
       for i: ℕ(0, number):
           value = value + first[i] * second[i]
       return value

   def normalize_vector(values: ℝ[m], number: ℕ): ℝ[m]:
       result: ℝ[number] = zero_array(number)
       norm: ℝ = sqrt(dot_product(values, values, number))
       for i: ℕ(0, number):
           result[i] = values[i] / norm
       return result

The ``integrate`` function approximates a uniform-grid integral as

.. math::

   \int f(r)\,dr \approx \sum_i f_i\,\Delta r.

The ``dot_product`` function calculates the Euclidean inner product,

.. math::

   \mathbf{a}^{T}\mathbf{b}=\sum_i a_i b_i.

Finally, ``normalize_vector`` divides each vector component by its Euclidean
norm,

.. math::

   \|\mathbf{v}\|_2=\sqrt{\mathbf{v}^{T}\mathbf{v}},

so that the normalized vector satisfies
:math:`\mathbf{v}^{T}\mathbf{v}=1` to numerical precision.

Applying the finite-difference Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   def apply_hamiltonian(wavefunction: ℝ[m], potential: ℝ[n], kinetic_coefficient: ℝ, number: ℕ): ℝ[m]:
       result: ℝ[number] = zero_array(number)
       result[0] = ((2.0 * kinetic_coefficient + potential[0]) * wavefunction[0] - kinetic_coefficient * wavefunction[1])
       for i: ℕ(1, number - 1):
           result[i] = (-kinetic_coefficient * wavefunction[i - 1] + (2.0 * kinetic_coefficient + potential[i]) * wavefunction[i] - kinetic_coefficient * wavefunction[i + 1])
       result[number - 1] = (-kinetic_coefficient * wavefunction[number - 2] + (2.0 * kinetic_coefficient + potential[number - 1]) * wavefunction[number - 1])
       return result

The ``apply_hamiltonian`` function evaluates :math:`H\psi` on the coordinate
grid using the finite-difference Hamiltonian. Because ``potential`` and ``kinetic_coefficient`` are expressed in
electronvolts, the returned array has units of electronvolts multiplied by the
wavefunction units.

Diagonalizing the projected Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final helper works with the projected arrays.

.. code-block:: text

   def jacobi_diagonalize(): ℕ:
       jacobi_not_converged: ℕ = 0
       jacobi_converged: ℕ = 1
       jacobi_converged_local: ℕ = jacobi_not_converged
       for i: ℕ(0, Krylov_dimension):
           Ritz_vectors[i,i] = 1.0
           for j: ℕ(0, Krylov_dimension):
               projected_work_matrix[i,j] = projected_hamiltonian[i,j]
       for rotation: ℕ(0, jacobi_maximum_rotations):
           jacobi_p = 0
           jacobi_q = 1
           jacobi_largest = abs(projected_work_matrix[0,1])
           for i: ℕ(0, Krylov_dimension):
               for j: ℕ(i + 1, Krylov_dimension):
                   if abs(projected_work_matrix[i,j]) > jacobi_largest:
                       jacobi_largest = abs(projected_work_matrix[i,j])
                       jacobi_p = i
                       jacobi_q = j
           if jacobi_converged_local == jacobi_not_converged:
               if jacobi_largest <= jacobi_tolerance:
                   jacobi_converged_local = jacobi_converged
           if jacobi_largest > jacobi_tolerance:
               jacobi_angle = 0.5 * atan2(2.0 * projected_work_matrix[jacobi_p,jacobi_q], projected_work_matrix[jacobi_q,jacobi_q] - projected_work_matrix[jacobi_p,jacobi_p])
               jacobi_cosine = cos(jacobi_angle)
               jacobi_sine = sin(jacobi_angle)
               jacobi_app = (projected_work_matrix[jacobi_p,jacobi_p] + 0.0)
               jacobi_aqq = (projected_work_matrix[jacobi_q,jacobi_q] + 0.0)
               jacobi_apq = (projected_work_matrix[jacobi_p,jacobi_q] + 0.0)
               for k: ℕ(0, Krylov_dimension):
                   if k != jacobi_p:
                       if k != jacobi_q:
                           jacobi_akp = (projected_work_matrix[k,jacobi_p] + 0.0)
                           jacobi_akq = (projected_work_matrix[k,jacobi_q] + 0.0)
                           projected_work_matrix[k,jacobi_p] = (jacobi_cosine * jacobi_akp - jacobi_sine * jacobi_akq)
                           projected_work_matrix[jacobi_p,k] = (projected_work_matrix[k,jacobi_p])
                           projected_work_matrix[k,jacobi_q] = (jacobi_sine * jacobi_akp + jacobi_cosine * jacobi_akq)
                           projected_work_matrix[jacobi_q,k] = (projected_work_matrix[k,jacobi_q])
               projected_work_matrix[jacobi_p,jacobi_p] = (jacobi_cosine * jacobi_cosine * jacobi_app - 2.0 * jacobi_sine * jacobi_cosine * jacobi_apq + jacobi_sine * jacobi_sine * jacobi_aqq)
               projected_work_matrix[jacobi_q,jacobi_q] = (jacobi_sine * jacobi_sine * jacobi_app + 2.0 * jacobi_sine * jacobi_cosine * jacobi_apq + jacobi_cosine * jacobi_cosine * jacobi_aqq)
               projected_work_matrix[jacobi_p,jacobi_q] = 0.0
               projected_work_matrix[jacobi_q,jacobi_p] = 0.0
               for k: ℕ(0, Krylov_dimension):
                   jacobi_vkp = Ritz_vectors[k,jacobi_p] + 0.0
                   jacobi_vkq = Ritz_vectors[k,jacobi_q] + 0.0
                   Ritz_vectors[k,jacobi_p] = (jacobi_cosine * jacobi_vkp - jacobi_sine * jacobi_vkq)
                   Ritz_vectors[k,jacobi_q] = (jacobi_sine * jacobi_vkp + jacobi_cosine * jacobi_vkq)
       for i: ℕ(0, Krylov_dimension):
           Ritz_values[i] = projected_work_matrix[i,i]
       for i: ℕ(0, Krylov_dimension - 1):
           jacobi_minimum = i
           for j: ℕ(i + 1, Krylov_dimension):
               if Ritz_values[j] < Ritz_values[jacobi_minimum]:
                   jacobi_minimum = j
           if jacobi_minimum != i:
               jacobi_temporary_value = Ritz_values[i] + 0.0
               Ritz_values[i] = Ritz_values[jacobi_minimum]
               Ritz_values[jacobi_minimum] = jacobi_temporary_value
               for k: ℕ(0, Krylov_dimension):
                   jacobi_temporary_vector = Ritz_vectors[k,i] + 0.0
                   Ritz_vectors[k,i] = Ritz_vectors[k,jacobi_minimum]
                   Ritz_vectors[k,jacobi_minimum] = jacobi_temporary_vector
       return jacobi_converged_local

The Jacobi diagonalization performs the following operations:

#. It copies the projected operator into a working array and initializes
   ``Ritz_vectors`` as the identity matrix.

#. At each iteration, it searches the upper triangle of the working array for
   the largest off-diagonal magnitude:

   .. math::

      |T_{pq}|
      =
      \max_{i<j}|T_{ij}|.

#. It calculates the rotation angle:

   .. math::

      \theta
      =
      \frac{1}{2}
      \operatorname{atan2}
      \left(
      2T_{pq},
      T_{qq}-T_{pp}
      \right).

   The corresponding plane rotation is applied
   to the selected rows and columns of the working array and to the accumulated
   ``Ritz_vectors``.

#. It considers the calculation as converged when the largest off-diagonal
   magnitude satisfies

   .. math::

      \max_{p<q}|T_{pq}|
      \leq
      10^{-7},

   corresponding to ``jacobi_tolerance = 1.0e-7``.

#. After convergence, it copies the diagonal entries into ``Ritz_values`` and
   sorts the values in ascending energy order. The corresponding columns of
   ``Ritz_vectors`` are reordered in the same way.

After many rotations, the projected operator becomes approximately diagonal:

.. math::

   T
   \longrightarrow
   \begin{pmatrix}
   \varepsilon_0 & 0 & \cdots \\
   0 & \varepsilon_1 & \cdots \\
   \vdots & \vdots & \ddots
   \end{pmatrix}.

The function returns ``1`` when convergence is detected and ``0`` otherwise.

.. note::

   The Ritz values are approximate eigenvalues of the original Hamiltonian.
   The Ritz vectors are eigenvectors in the reduced Krylov basis and must be
   reconstructed on the coordinate grid to obtain the approximate physical
   wavefunctions.

If ``atan2`` is not available, add the following function in ``physika/runtime.py``:

.. code-block:: python

   def atan2(y, x):
       return torch.atan2(y, x)
       
HCl Morse-oscillator numerical calculation
------------------------------------------

With the helper functions now defined, the tutorial proceeds to the main
numerical calculation. The following sections define the physical and numerical
parameters, evaluate the Morse potential, construct the block-Krylov basis,
solve the projected eigenvalue problem, and calculate the HCl vibrational
energies and wavefunctions.


Physical constants and numerical parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following block defines the required constants, 
unit conversion, and solver settings.

.. code-block:: text

   atomic_mass_unit: ℝ = 1.66053906660e-27
   hbar: ℝ = 1.054571817e-34
   planck_constant: ℝ = 6.62607015e-34
   speed_of_light_cm: ℝ = 2.99792458e10
   electron_volt: ℝ = 1.602176634e-19
   mass_H_u: ℝ = 1.00784
   mass_Cl_u: ℝ = 35.45
   mass_H: ℝ = mass_H_u * atomic_mass_unit
   mass_Cl: ℝ = mass_Cl_u * atomic_mass_unit
   reduced_mass_HCl: ℝ = ((mass_H * mass_Cl) / (mass_H + mass_Cl))
   dissociation_energy_eV: ℝ = 4.61907
   dissociation_energy_J: ℝ = (dissociation_energy_eV * electron_volt)
   equilibrium_distance: ℝ = 1.2746e-10
   morse_a: ℝ = 1.868e10
   N_grid: ℕ = 100
   N_levels: ℕ = 2
   block_size: ℕ = 2
   Krylov_dimension: ℕ = 40
   r_min: ℝ = 0.50e-10
   r_max: ℝ = 2.00e-10

``N_levels = 2`` is sufficient for the fundamental transition because only
:math:`E_0` and :math:`E_1` are required.

Constructing the Morse potential and kinetic-energy coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Morse potential and finite-difference kinetic coefficient are then evaluated:

.. code-block:: text

   distance_from_equilibrium: ℝ[N_grid] = (bond_distance - equilibrium_distance)
   morse_exponential: ℝ[N_grid] = exp(-morse_a * distance_from_equilibrium)
   morse_difference: ℝ[N_grid] = (1.0 - morse_exponential)
   potential_J: ℝ[N_grid] = (dissociation_energy_J * morse_difference * morse_difference)
   potential_eV: ℝ[N_grid] = (potential_J / electron_volt)
   kinetic_coefficient_J: ℝ = ((hbar / grid_spacing) * (hbar / (2.0 * reduced_mass_HCl * grid_spacing)))
   kinetic_coefficient_eV: ℝ = (kinetic_coefficient_J / electron_volt)

The factored source expression is algebraically identical to

.. math::

   C=\frac{\hbar^2}{2\mu(\Delta r)^2}.

Initial trial block and working-array initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first trial vector is a Gaussian centred at :math:`r_e`:

.. code-block:: text

   trial_width: ℝ = 0.20e-10
   gaussian_argument: ℝ[N_grid] = (distance_from_equilibrium / trial_width)
   gaussian_envelope: ℝ[N_grid] = exp(-gaussian_argument * gaussian_argument)
   Krylov_basis: ℝ[Krylov_dimension,N_grid] = zero_matrix(Krylov_dimension, N_grid)
   H_Krylov: ℝ[Krylov_dimension,N_grid] = zero_matrix(Krylov_dimension, N_grid)
   projected_hamiltonian: ℝ[Krylov_dimension,Krylov_dimension] = zero_matrix(Krylov_dimension, Krylov_dimension)
   projected_work_matrix: ℝ[Krylov_dimension,Krylov_dimension] = zero_matrix(Krylov_dimension, Krylov_dimension)
   Ritz_vectors: ℝ[Krylov_dimension,Krylov_dimension] = zero_matrix(Krylov_dimension, Krylov_dimension)
   Ritz_values: ℝ[Krylov_dimension] = zero_array(Krylov_dimension)
   jacobi_tolerance: ℝ = 1.0e-7
   jacobi_maximum_rotations: ℕ = 10000

Mathematically,

.. math::

   g_i
   =
   \exp\left[
   -\left(\frac{r_i-r_e}{\sigma}\right)^2
   \right],
   \qquad
   \sigma=0.20\ \mathrm{\AA}.

Each row of ``Krylov_basis`` stores one 100-component grid vector.
``H_Krylov[q]`` stores the Hamiltonian applied to that row.
``projected_hamiltonian``, ``projected_work_matrix``, and ``Ritz_vectors`` are
all :math:`40\times40`; ``Ritz_values`` contains 40 entries.

Building the block-Krylov basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first basis row is filled and normalized:

.. code-block:: text

   for i: ℕ(0, N_grid):
       Krylov_basis[0,i] = gaussian_envelope[i]
   Krylov_basis[0] = normalize_vector(Krylov_basis[0], N_grid)

The remaining rows are generated by

.. code-block:: text

   for q_index: ℕ(1, Krylov_dimension):
       if q_index < block_size:
           for i: ℕ(0, N_grid):
               candidate[i] = (gaussian_argument[i] * Krylov_basis[q_index - 1,i])
       else:
           candidate = apply_hamiltonian(Krylov_basis[q_index - block_size], potential_eV, kinetic_coefficient_eV, N_grid)
       for orthogonalization_pass: ℕ(0, 2):
           for lower: ℕ(0, q_index):
               overlap = dot_product(Krylov_basis[lower], candidate, N_grid)
               for i: ℕ(0, N_grid):
                   candidate[i] = (candidate[i] - overlap * Krylov_basis[lower,i])
       candidate_norm = sqrt(dot_product(candidate, candidate, N_grid))
       for i: ℕ(0, N_grid):
           Krylov_basis[q_index,i] = (candidate[i] / candidate_norm)

Because ``block_size = 2``, ``q_index = 1`` creates the second starting vector

.. math::

   q_1^{\mathrm{candidate}}(r_i)
   =
   \frac{r_i-r_e}{\sigma}q_0(r_i).

The first vector is Gaussian-like, while the second changes sign at
:math:`r_e`.  This gives the initial block overlap with different nodal
characters.

For ``q_index >= 2``, the candidate is generated from

.. math::

   q_j^{\mathrm{candidate}}=Hq_{j-2}.

This interleaves two Krylov chains:

.. math::

   q_0,\ Hq_0,\ H^2q_0,\ldots

and

.. math::

   q_1,\ Hq_1,\ H^2q_1,\ldots.

Every candidate is orthogonalized against all accepted vectors twice.  For an
existing vector :math:`q_k`, the overlap

.. math::

   s_k=q_k^Tq_j^{\mathrm{candidate}}

is removed:

.. math::

   q_j^{\mathrm{candidate}}
   \leftarrow
   q_j^{\mathrm{candidate}}-s_kq_k.

The second complete Gram--Schmidt pass reduces numerical loss of
orthogonality.  Finally, the candidate is normalized and stored.  Ideally,

.. math::

   q_i^Tq_j=\delta_{ij}.

Projecting and solving the eigenproblem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, the Hamiltonian is applied to every basis vector and then the projected elements are calculated:

.. code-block:: text

   for q_index: ℕ(0, Krylov_dimension):
       H_Krylov[q_index] = apply_hamiltonian(Krylov_basis[q_index], potential_eV, kinetic_coefficient_eV, N_grid)
   for row: ℕ(0, Krylov_dimension):
       for column: ℕ(row, Krylov_dimension):
           projected_hamiltonian[row,column] = dot_product(Krylov_basis[row], H_Krylov[column], N_grid)
           projected_hamiltonian[column,row] = (projected_hamiltonian[row,column])

Therefore,

.. math::

   T_{ij}=q_i^THq_j.

Only the upper triangle is calculated explicitly.  The value is copied to the
opposite triangle because the projected Hamiltonian is real and symmetric.
The resulting :math:`40\times40` projected operator describes the action of
the vibrational Hamiltonian inside the selected Krylov subspace.

The solver is invoked with

.. code-block:: text

   jacobi_diagonalize()

It solves

.. math::

   T y_n=\varepsilon_n y_n.

The sorted ``Ritz_values`` :math:`\varepsilon_n` approximate the vibrational
energies.  Column ``n`` of ``Ritz_vectors`` contains :math:`y_n`, the
coefficients of state ``n`` in the Krylov basis.

Reconstructing the grid wavefunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two lowest states are reconstructed:

.. code-block:: text

   vibrational_energies_eV: ℝ[N_levels] = zero_array(N_levels)
   psi_raw: ℝ[N_levels,N_grid] = zero_matrix(N_levels, N_grid)

   for n: ℕ(0, N_levels):
       vibrational_energies_eV[n] = Ritz_values[n]
       for i: ℕ(0, N_grid):
           for q_index: ℕ(0, Krylov_dimension):
               psi_raw[n,i] = (psi_raw[n,i] + Krylov_basis[q_index,i] * Ritz_vectors[q_index,n])
       psi_raw[n] = normalize_vector(psi_raw[n], N_grid)

In mathematical notation,

.. math::

   \psi_n^{\mathrm{raw}}(r_i)
   =
   \sum_{q=0}^{K-1}Q_{qi}(y_n)_q,

where :math:`K=40`.  Although the coefficients come from a 40-dimensional
projected eigenproblem, each reconstructed wavefunction has 100 grid values.
The final line gives the reconstructed array Euclidean normalization before
the physical grid normalization is applied.

Continuous wavefunction normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The normalization condition is

.. math::

   \int|\psi_n(r)|^2dr=1.

It is applied using

.. code-block:: text

   normalization_factor: ℝ[N_levels] = zero_array(N_levels)
   psi: ℝ[N_levels,N_grid] = zero_matrix(N_levels, N_grid)

   for n: ℕ(0, N_levels):
       normalization_factor[n] = sqrt(integrate(psi_raw[n] * psi_raw[n], grid_spacing, N_grid))
       for i: ℕ(0, N_grid):
           psi[n,i] = (psi_raw[n,i] / normalization_factor[n])

The normalization factor is

.. math::

   A_n
   =
   \sqrt{\sum_i|\psi_n^{\mathrm{raw}}(r_i)|^2\Delta r},

and the final wavefunction is

.. math::

   \psi_n(r_i)=\frac{\psi_n^{\mathrm{raw}}(r_i)}{A_n}.

It consequently satisfies

.. math::

   \sum_i|\psi_n(r_i)|^2\Delta r\approx1.

Because :math:`r` is in metres at this stage, ``psi`` has units of
:math:`\mathrm{m}^{-1/2}`.

Fundamental transition
~~~~~~~~~~~~~~~~~~~~~~

The fundamental transition is :math:`v=0\rightarrow1`.

.. code-block:: text

   transition_eV: ℝ = (vibrational_energies_eV[1] - vibrational_energies_eV[0])
   transition_J: ℝ = transition_eV * electron_volt
   wavenumber: ℝ = (transition_J / (planck_constant * speed_of_light_cm))
   wavelength_micrometer: ℝ = (10000.0 / wavenumber)

The equations are

.. math::

   \Delta E_{01}=E_1-E_0,

.. math::

   \widetilde{\nu}_{01}
   =
   \frac{\Delta E_{01}}{hc},

and

.. math::

   \lambda_{\mu\mathrm m}
   =
   \frac{10000}{\widetilde{\nu}_{01}}.

Since ``speed_of_light_cm`` is in :math:`\mathrm{cm\,s^{-1}}`,
``wavenumber`` is in :math:`\mathrm{cm^{-1}}`.  The factor 10000 converts the
reciprocal wavenumber in centimetres into micrometres.

Results
-------

Calculated results
~~~~~~~~~~~~~~~~~~

The expected output corresponds, in order, to:

#. Jacobi convergence flag
#. Ground-state energy, :math:`E_0` (eV)
#. First-excited-state energy, :math:`E_1` (eV)
#. Fundamental transition energy, :math:`\Delta E_{01}` (eV)
#. Fundamental wavenumber, :math:`\widetilde{\nu}_{01}` (:math:`\mathrm{cm^{-1}}`)
#. Fundamental wavelength, :math:`\lambda_{01}` (:math:`\mu\mathrm{m}`)


.. code-block:: text

   physika_print(vibrational_energies_eV[0])
   physika_print(vibrational_energies_eV[1])
   physika_print(transition_eV)
   physika_print(wavenumber)
   physika_print(wavelength_micrometer)

.. admonition:: Results
   :class: note

   .. code-block:: text

      1 ∈ ℝ
      0.18333028256893158 ∈ ℝ
      0.5384687781333923 ∈ ℝ
      0.35513848066329956 ∈ ℝ
      2864.384765625 ∈ ℝ
      3.4911510944366455 ∈ ℝ

.. figure:: ../_static/tutorial_files/HCl_morse_oscillator.png
   :alt: Morse potential and vibrational wavefunctions of HCl
   :width: 70%
   :align: center

   **Figure: Morse-potential representation of the HCl molecule.** The upper panel
   shows the potential-energy curve as a function of the H--Cl internuclear
   distance, together with vibrational energy levels,
   equilibrium bond length :math:`r_e`, and Morse well depth :math:`D_e`.
   The lower panel displays the normalized vibrational wavefunctions for the
   ground (:math:`v=0`) and first excited (:math:`v=1`) states.


Analytical Morse energies
~~~~~~~~~~~~~~~~~~~~~~~~~

The Morse oscillator has an analytical expression for its bound-state
energies indexed by the vibrational quantum number :math:`v`.

.. math::

   E_v^{\mathrm{analytical}}
   =
   \hbar\omega_e\left(v+\frac{1}{2}\right)
   -
   \frac{(\hbar\omega_e)^2}{4D_e}
   \left(v+\frac{1}{2}\right)^2,

where

.. math::

   \omega_e
   =
   a\sqrt{\frac{2D_e}{\mu}}.

For the parameters used in this tutorial,

.. math::

   D_e = 4.61907\ \mathrm{eV},

.. math::

   a = 1.868\times10^{10}\ \mathrm{m^{-1}},

and

.. math::

   \mu
   =
   \frac{m_{\mathrm{H}}m_{\mathrm{Cl}}}
        {m_{\mathrm{H}}+m_{\mathrm{Cl}}}
   \approx 0.9799793\ \mathrm{u},

the harmonic energy spacing is

.. math::

   \hbar\omega_e \approx 0.370815\ \mathrm{eV}.

The first two analytical Morse energies are

.. math::

   E_0^{\mathrm{analytical}}
   \approx 0.183547\ \mathrm{eV},

and

.. math::

   E_1^{\mathrm{analytical}}
   \approx 0.539477\ \mathrm{eV}.

Therefore, the analytical fundamental transition energy is

.. math::

   \Delta E_{01}^{\mathrm{analytical}}
   =
   E_1^{\mathrm{analytical}}
   -
   E_0^{\mathrm{analytical}}
   \approx 0.355930\ \mathrm{eV}.

The corresponding analytical wavenumber is

.. math::

   \widetilde{\nu}_{01}^{\mathrm{analytical}}
   =
   \frac{\Delta E_{01}^{\mathrm{analytical}}}{hc}
   \approx 2870.77\ \mathrm{cm^{-1}}.

Comparison
~~~~~~~~~~

The calculated fundamental transition in this tutorial is compared below with
the analytical Morse result and the experimental HCl value.

.. list-table:: Comparison of the HCl fundamental transition
   :header-rows: 1
   :widths: 38 25 25
   :align: center

   * - Method
     - Transition energy,
       :math:`\Delta E_{01}` (eV)
     - Wavenumber,
       :math:`\widetilde{\nu}_{01}` (:math:`\mathrm{cm^{-1}}`)
   * - Numerical block-Krylov
     - 0.355138
     - 2864.38
   * - Analytical Morse
     - 0.355930
     - 2870.77
   * - Experiment
     - 0.357806
     - 2885.90

The numerical wavenumber differs from the analytical Morse result by

.. math::

   \left|
   2864.38-2870.77
   \right|
   \approx 6.39\ \mathrm{cm^{-1}},

corresponding to a relative error of approximately

.. math::

   \frac{6.39}{2870.77}\times100
   \approx 0.223\%.

.. note::

   Although this tutorial calculates only the two lowest vibrational states,
   higher states can be obtained by increasing ``N_levels`` and adjusting the
   numerical parameters as needed to ensure convergence. These states
   can be used to calculate overtone transitions
   :math:`0\rightarrow n` (:math:`n\geq2`) and hot-band transitions
   :math:`1\rightarrow n` (:math:`n\geq2`).
   
.. admonition:: Try it yourself: Calculate the first overtone
   :class: important

   Increase ``N_levels`` and the required convergence parameters to calculate
   :math:`E_2`. Then calculate

   .. math::

      \widetilde{\nu}_{02}=\frac{E_2-E_0}{hc}

   Compare your result with :math:`2\widetilde{\nu}_{01}` and with the
   first-overtone value reported by `Holmes and Shay
   <https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/13%3A_Molecular_Spectroscopy/13.05%3A_Vibrational_Overtones>`_.

   **Hint:** If the current settings do not produce a converged result, adjust
   ``Krylov_dimension`` and, if necessary, ``N_grid`` and
   ``jacobi_maximum_rotations`` until :math:`E_2` is converged. For an anharmonic
   Morse oscillator, :math:`\widetilde{\nu}_{02}<2\widetilde{\nu}_{01}` because
   successive vibrational levels become more closely spaced.

Plotting (Optional)
-------------------

The coordinate is converted from metres to angstroms in ``HCl_morse_oscillator.phyk``:

.. code-block:: text

   bond_distance_angstrom: ℝ[N_grid] = (bond_distance / 1.0e-10)
   equilibrium_distance_angstrom: ℝ = (equilibrium_distance / 1.0e-10)

Add the following custom ``physika_plot`` function to ``physika/runtime.py``:

.. code-block:: python

   def physika_plot(bond_distance, potential, psi, vibrational_energies_eV, dissociation_energy, equilibrium_distance, number_of_energy_levels=10, number_of_wavefunctions=2):
       import numpy as np
       import matplotlib.pyplot as plt
       def to_numpy(value):
           if hasattr(value, "detach"):
               return value.detach().cpu().numpy()
           return np.asarray(value)
       bond_distance = to_numpy(bond_distance).reshape(-1)
       potential = to_numpy(potential).reshape(-1)
       psi = to_numpy(psi)
       vibrational_energies_eV = to_numpy(vibrational_energies_eV).reshape(-1)
       dissociation_energy = float(np.asarray(to_numpy(dissociation_energy)).squeeze())
       equilibrium_distance = float(np.asarray(to_numpy(equilibrium_distance)).squeeze())
       if psi.ndim != 2:
           raise ValueError(f"psi must be a 2D array, but received shape {psi.shape}")
       if psi.shape[0] != len(bond_distance):
           if psi.shape[1] == len(bond_distance):
               psi = psi.T
           else:
               raise ValueError(f"Neither dimension of psi matches bond_distance: bond_distance={len(bond_distance)}, psi={psi.shape}")
       bound_energies = vibrational_energies_eV[vibrational_energies_eV < dissociation_energy]
       number_of_energy_levels = min(int(number_of_energy_levels), len(bound_energies))
       number_of_wavefunctions = min(int(number_of_wavefunctions), psi.shape[1], len(vibrational_energies_eV))
       fig, (ax_energy, ax_wavefunction) = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]})

       # Upper panel: Morse potential and vibrational energies
       ax_energy.plot(bond_distance, potential, color="black", linewidth=1.8, label=r"$V(r)$")
       for n in range(number_of_energy_levels):
           energy = float(bound_energies[n])
           mask = potential <= energy
           if n == 0:
               color = "tab:blue"
               linewidth = 1.5
               label = rf"$E_0={energy:.3f}\ \mathrm{{eV}}$"
           elif n == 1:
               color = "tab:orange"
               linewidth = 1.5
               label = rf"$E_1={energy:.3f}\ \mathrm{{eV}}$"
           else:
               color = "black"
               linewidth = 0.5
               label = "_nolegend_"
           ax_energy.plot(bond_distance[mask], np.full_like(bond_distance[mask], energy, dtype=float), color=color, linestyle="--", linewidth=linewidth, label=label)
       ax_energy.axhline(dissociation_energy, color="tab:red", linestyle=":", linewidth=1.5, label=(rf"$D_e={dissociation_energy:.2f}\ \mathrm{{eV}}$"))
       ax_energy.axvline(equilibrium_distance, color="gray", linestyle=":", linewidth=1.0, label=(rf"$r_e={equilibrium_distance:.3f}\ \mathrm{{\AA}}$"))
       ax_energy.set_ylabel("Energy (eV)")
       ax_energy.set_ylim(0.0, max(5.2, 1.05 * dissociation_energy))
       ax_energy.legend(frameon=False, fontsize=8, loc="lower right", labelspacing=0)

       # Lower panel: normalized vibrational wavefunctions
       colors = ["tab:blue", "tab:orange"]
       for n in range(number_of_wavefunctions):
           ax_wavefunction.plot(bond_distance, psi[:, n], color=colors[n % len(colors)], linewidth=1.8, label=rf"$\psi_{n}(r)$")
       ax_wavefunction.axhline(0.0, color="gray", linewidth=0.8,)
       ax_wavefunction.axvline(equilibrium_distance, color="gray", linestyle=":", linewidth=1.0)
       ax_wavefunction.set_xlabel(r"H-Cl bond distance, $r$ ($\mathrm{\AA}$)")
       ax_wavefunction.set_ylabel(r"Wavefunction, $\psi_n(r)$ " r"($\mathrm{\AA}^{-1/2}$)")
       ax_wavefunction.legend(frameon=False, fontsize=9)
       ax_wavefunction.set_xlim(0.7, 2.5)
       plt.tight_layout()
       plt.savefig("HCl_morse_oscillator.png", dpi=300, bbox_inches="tight")
       # plt.show()
       plt.close()

Source code
-----------

The complete Physika implementation is also provided in
``tutorials/HCl_morse_oscillator.phyk``

.. code-block:: text

   # HCl vibration using a Morse potential

   def zero_array(length: ℕ): ℝ[m]:
       values: ℝ[length] = for i: ℕ(length) -> i * 0.0
       return values

   def zero_matrix(rows: ℕ, columns: ℕ): ℝ[m,n]:
       values: ℝ[rows,columns] = for i: ℕ(rows) -> for j: ℕ(columns) -> (i + j) * 0.0
       return values

   def linspace(start: ℝ, end: ℝ, number: ℕ): ℝ[m]:
       values: ℝ[number] = zero_array(number)
       spacing: ℝ = (end - start) / (number - 1)
       for i: ℕ(0, number):
           values[i] = start + i * spacing
       return values

   def integrate(values: ℝ[m], grid_spacing: ℝ, number: ℕ): ℝ:
       integral: ℝ = 0.0
       for i: ℕ(0, number):
           integral = integral + values[i] * grid_spacing
       return integral

   def dot_product(first: ℝ[m], second: ℝ[n], number: ℕ): ℝ:
       value: ℝ = 0.0
       for i: ℕ(0, number):
           value = value + first[i] * second[i]
       return value

   def normalize_vector(values: ℝ[m], number: ℕ): ℝ[m]:
       result: ℝ[number] = zero_array(number)
       norm: ℝ = sqrt(dot_product(values, values, number))
       for i: ℕ(0, number):
           result[i] = values[i] / norm
       return result

   def apply_hamiltonian(wavefunction: ℝ[m], potential: ℝ[n], kinetic_coefficient: ℝ, number: ℕ): ℝ[m]:
       result: ℝ[number] = zero_array(number)
       result[0] = ((2.0 * kinetic_coefficient + potential[0]) * wavefunction[0] - kinetic_coefficient * wavefunction[1])
       for i: ℕ(1, number - 1):
           result[i] = (-kinetic_coefficient * wavefunction[i - 1] + (2.0 * kinetic_coefficient + potential[i]) * wavefunction[i] - kinetic_coefficient * wavefunction[i + 1])
       result[number - 1] = (-kinetic_coefficient * wavefunction[number - 2] + (2.0 * kinetic_coefficient + potential[number - 1]) * wavefunction[number - 1])
       return result

   def jacobi_diagonalize(): ℕ:
       jacobi_not_converged: ℕ = 0
       jacobi_converged: ℕ = 1
       jacobi_converged_local: ℕ = jacobi_not_converged
       for i: ℕ(0, Krylov_dimension):
           Ritz_vectors[i,i] = 1.0
           for j: ℕ(0, Krylov_dimension):
               projected_work_matrix[i,j] = projected_hamiltonian[i,j]
       for rotation: ℕ(0, jacobi_maximum_rotations):
           jacobi_p = 0
           jacobi_q = 1
           jacobi_largest = abs(projected_work_matrix[0,1])
           for i: ℕ(0, Krylov_dimension):
               for j: ℕ(i + 1, Krylov_dimension):
                   if abs(projected_work_matrix[i,j]) > jacobi_largest:
                       jacobi_largest = abs(projected_work_matrix[i,j])
                       jacobi_p = i
                       jacobi_q = j
           if jacobi_converged_local == jacobi_not_converged:
               if jacobi_largest <= jacobi_tolerance:
                   jacobi_converged_local = jacobi_converged
           if jacobi_largest > jacobi_tolerance:
               jacobi_angle = 0.5 * atan2(2.0 * projected_work_matrix[jacobi_p,jacobi_q], projected_work_matrix[jacobi_q,jacobi_q] - projected_work_matrix[jacobi_p,jacobi_p])
               jacobi_cosine = cos(jacobi_angle)
               jacobi_sine = sin(jacobi_angle)
               jacobi_app = (projected_work_matrix[jacobi_p,jacobi_p] + 0.0)
               jacobi_aqq = (projected_work_matrix[jacobi_q,jacobi_q] + 0.0)
               jacobi_apq = (projected_work_matrix[jacobi_p,jacobi_q] + 0.0)
               for k: ℕ(0, Krylov_dimension):
                   if k != jacobi_p:
                       if k != jacobi_q:
                           jacobi_akp = (projected_work_matrix[k,jacobi_p] + 0.0)
                           jacobi_akq = (projected_work_matrix[k,jacobi_q] + 0.0)
                           projected_work_matrix[k,jacobi_p] = (jacobi_cosine * jacobi_akp - jacobi_sine * jacobi_akq)
                           projected_work_matrix[jacobi_p,k] = (projected_work_matrix[k,jacobi_p])
                           projected_work_matrix[k,jacobi_q] = (jacobi_sine * jacobi_akp + jacobi_cosine * jacobi_akq)
                           projected_work_matrix[jacobi_q,k] = (projected_work_matrix[k,jacobi_q])
               projected_work_matrix[jacobi_p,jacobi_p] = (jacobi_cosine * jacobi_cosine * jacobi_app - 2.0 * jacobi_sine * jacobi_cosine * jacobi_apq + jacobi_sine * jacobi_sine * jacobi_aqq)
               projected_work_matrix[jacobi_q,jacobi_q] = (jacobi_sine * jacobi_sine * jacobi_app + 2.0 * jacobi_sine * jacobi_cosine * jacobi_apq + jacobi_cosine * jacobi_cosine * jacobi_aqq)
               projected_work_matrix[jacobi_p,jacobi_q] = 0.0
               projected_work_matrix[jacobi_q,jacobi_p] = 0.0
               for k: ℕ(0, Krylov_dimension):
                   jacobi_vkp = Ritz_vectors[k,jacobi_p] + 0.0
                   jacobi_vkq = Ritz_vectors[k,jacobi_q] + 0.0
                   Ritz_vectors[k,jacobi_p] = (jacobi_cosine * jacobi_vkp - jacobi_sine * jacobi_vkq)
                   Ritz_vectors[k,jacobi_q] = (jacobi_sine * jacobi_vkp + jacobi_cosine * jacobi_vkq)
       for i: ℕ(0, Krylov_dimension):
           Ritz_values[i] = projected_work_matrix[i,i]
       for i: ℕ(0, Krylov_dimension - 1):
           jacobi_minimum = i
           for j: ℕ(i + 1, Krylov_dimension):
               if Ritz_values[j] < Ritz_values[jacobi_minimum]:
                   jacobi_minimum = j
           if jacobi_minimum != i:
               jacobi_temporary_value = Ritz_values[i] + 0.0
               Ritz_values[i] = Ritz_values[jacobi_minimum]
               Ritz_values[jacobi_minimum] = jacobi_temporary_value
               for k: ℕ(0, Krylov_dimension):
                   jacobi_temporary_vector = Ritz_vectors[k,i] + 0.0
                   Ritz_vectors[k,i] = Ritz_vectors[k,jacobi_minimum]
                   Ritz_vectors[k,jacobi_minimum] = jacobi_temporary_vector
       return jacobi_converged_local

   atomic_mass_unit: ℝ = 1.66053906660e-27
   hbar: ℝ = 1.054571817e-34
   planck_constant: ℝ = 6.62607015e-34
   speed_of_light_cm: ℝ = 2.99792458e10
   electron_volt: ℝ = 1.602176634e-19
   mass_H_u: ℝ = 1.00784
   mass_Cl_u: ℝ = 35.45
   mass_H: ℝ = mass_H_u * atomic_mass_unit
   mass_Cl: ℝ = mass_Cl_u * atomic_mass_unit
   reduced_mass_HCl: ℝ = ((mass_H * mass_Cl) / (mass_H + mass_Cl))
   dissociation_energy_eV: ℝ = 4.61907
   dissociation_energy_J: ℝ = (dissociation_energy_eV * electron_volt)
   equilibrium_distance: ℝ = 1.2746e-10
   morse_a: ℝ = 1.868e10
   N_grid: ℕ = 100 #150
   N_levels: ℕ = 2
   block_size: ℕ = 2
   Krylov_dimension: ℕ = 40
   r_min: ℝ = 0.50e-10
   r_max: ℝ = 2.00e-10
   grid_spacing: ℝ = ((r_max - r_min) / (N_grid + 1))
   r_start: ℝ = r_min + grid_spacing
   r_end: ℝ = r_max - grid_spacing
   bond_distance: ℝ[N_grid] = linspace(r_start, r_end, N_grid)
   distance_from_equilibrium: ℝ[N_grid] = (bond_distance - equilibrium_distance)
   morse_exponential: ℝ[N_grid] = exp(-morse_a * distance_from_equilibrium)
   morse_difference: ℝ[N_grid] = (1.0 - morse_exponential)
   potential_J: ℝ[N_grid] = (dissociation_energy_J * morse_difference * morse_difference)
   potential_eV: ℝ[N_grid] = (potential_J / electron_volt)
   kinetic_coefficient_J: ℝ = ((hbar / grid_spacing) * (hbar / (2.0 * reduced_mass_HCl * grid_spacing)))
   kinetic_coefficient_eV: ℝ = (kinetic_coefficient_J / electron_volt)
   trial_width: ℝ = 0.20e-10
   gaussian_argument: ℝ[N_grid] = (distance_from_equilibrium / trial_width)
   gaussian_envelope: ℝ[N_grid] = exp(-gaussian_argument * gaussian_argument)
   Krylov_basis: ℝ[Krylov_dimension,N_grid] = zero_matrix(Krylov_dimension, N_grid)
   H_Krylov: ℝ[Krylov_dimension,N_grid] = zero_matrix(Krylov_dimension, N_grid)
   projected_hamiltonian: ℝ[Krylov_dimension,Krylov_dimension] = zero_matrix(Krylov_dimension, Krylov_dimension)
   projected_work_matrix: ℝ[Krylov_dimension,Krylov_dimension] = zero_matrix(Krylov_dimension, Krylov_dimension)
   Ritz_vectors: ℝ[Krylov_dimension,Krylov_dimension] = zero_matrix(Krylov_dimension, Krylov_dimension)
   Ritz_values: ℝ[Krylov_dimension] = zero_array(Krylov_dimension)
   jacobi_tolerance: ℝ = 1.0e-7
   jacobi_maximum_rotations: ℕ = 10000
   jacobi_largest: ℝ = 0.0
   jacobi_angle: ℝ = 0.0
   jacobi_cosine: ℝ = 0.0
   jacobi_sine: ℝ = 0.0
   jacobi_app: ℝ = 0.0
   jacobi_aqq: ℝ = 0.0
   jacobi_apq: ℝ = 0.0
   jacobi_akp: ℝ = 0.0
   jacobi_akq: ℝ = 0.0
   jacobi_vkp: ℝ = 0.0
   jacobi_vkq: ℝ = 0.0
   jacobi_temporary_value: ℝ = 0.0
   jacobi_temporary_vector: ℝ = 0.0
   jacobi_p: ℕ = 0
   jacobi_q: ℕ = 1
   jacobi_minimum: ℕ = 0
   candidate: ℝ[N_grid] = zero_array(N_grid)
   overlap: ℝ = 0.0
   candidate_norm: ℝ = 0.0

   for i: ℕ(0, N_grid):
       Krylov_basis[0,i] = gaussian_envelope[i]
   Krylov_basis[0] = normalize_vector(Krylov_basis[0], N_grid)

   for q_index: ℕ(1, Krylov_dimension):
       if q_index < block_size:
           for i: ℕ(0, N_grid):
               candidate[i] = (gaussian_argument[i] * Krylov_basis[q_index - 1,i])
       else:
           candidate = apply_hamiltonian(Krylov_basis[q_index - block_size], potential_eV, kinetic_coefficient_eV, N_grid)
       for orthogonalization_pass: ℕ(0, 2):
           for lower: ℕ(0, q_index):
               overlap = dot_product(Krylov_basis[lower], candidate, N_grid)
               for i: ℕ(0, N_grid):
                   candidate[i] = (candidate[i] - overlap * Krylov_basis[lower,i])
       candidate_norm = sqrt(dot_product(candidate, candidate, N_grid))
       for i: ℕ(0, N_grid):
           Krylov_basis[q_index,i] = (candidate[i] / candidate_norm)
   for q_index: ℕ(0, Krylov_dimension):
       H_Krylov[q_index] = apply_hamiltonian(Krylov_basis[q_index], potential_eV, kinetic_coefficient_eV, N_grid)

   for row: ℕ(0, Krylov_dimension):
       for column: ℕ(row, Krylov_dimension):
           projected_hamiltonian[row,column] = dot_product(Krylov_basis[row], H_Krylov[column], N_grid)
           projected_hamiltonian[column,row] = (projected_hamiltonian[row,column])

   jacobi_diagonalize()
   vibrational_energies_eV: ℝ[N_levels] = zero_array(N_levels)
   psi_raw: ℝ[N_levels,N_grid] = zero_matrix(N_levels, N_grid)

   for n: ℕ(0, N_levels):
       vibrational_energies_eV[n] = Ritz_values[n]
       for i: ℕ(0, N_grid):
           for q_index: ℕ(0, Krylov_dimension):
               psi_raw[n,i] = (psi_raw[n,i] + Krylov_basis[q_index,i] * Ritz_vectors[q_index,n])
       psi_raw[n] = normalize_vector(psi_raw[n], N_grid)

   normalization_factor: ℝ[N_levels] = zero_array(N_levels)
   psi: ℝ[N_levels,N_grid] = zero_matrix(N_levels, N_grid)

   for n: ℕ(0, N_levels):
       normalization_factor[n] = sqrt(integrate(psi_raw[n] * psi_raw[n], grid_spacing, N_grid))
       for i: ℕ(0, N_grid):
           psi[n,i] = (psi_raw[n,i] / normalization_factor[n])

   transition_eV: ℝ = (vibrational_energies_eV[1] - vibrational_energies_eV[0])
   transition_J: ℝ = transition_eV * electron_volt
   wavenumber: ℝ = (transition_J / (planck_constant * speed_of_light_cm))
   wavelength_micrometer: ℝ = (10000.0 / wavenumber)

   physika_print(vibrational_energies_eV[0])
   physika_print(vibrational_energies_eV[1])
   physika_print(transition_eV)
   physika_print(wavenumber)
   physika_print(wavelength_micrometer)

   bond_distance_angstrom: ℝ[N_grid] = (bond_distance / 1.0e-10)
   equilibrium_distance_angstrom: ℝ = (equilibrium_distance / 1.0e-10)
   psi_angstrom: ℝ[N_levels,N_grid] = (sqrt(1.0e-10) * psi)

   # Use the below function to plot the results (Optional)
   #physika_plot(bond_distance_angstrom, potential_eV, psi_angstrom, vibrational_energies_eV, dissociation_energy_eV, equilibrium_distance_angstrom, N_levels, N_levels)

References
----------

* Kelly, P. (2023). *5.3: The harmonic oscillator approximates
  vibrations*. `Chemistry LibreTexts
  <https://chem.libretexts.org/Courses/Pacific_Union_College/Quantum_Chemistry/05%3A_The_Harmonic_Oscillator_and_the_Rigid_Rotor/5.03%3A_The_Harmonic_Oscillator_Approximates_Vibrations>`_.

* Holmes, A., & Shay, H. T. *Vibrational overtones*.
  `Chemistry LibreTexts
  <https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/13%3A_Molecular_Spectroscopy/13.05%3A_Vibrational_Overtones>`_.

* Hanson, D. M., Harvey, E., Sweeney, R., & Zielinski, T. J.
  *The harmonic oscillator and infrared spectra*.
  `Chemistry LibreTexts
  <https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Physical_Chemistry_(LibreTexts)/05%3A_The_Harmonic_Oscillator_and_the_Rigid_Rotor/5.05%3A_The_Harmonic_Oscillator_and_Infrared_Spectra>`_.

* Nasser, I., Abdelmonem, M. S., Bahlouli, H., & Alhaidari, A. D. (2007).
  The rotating Morse potential model for diatomic molecules in the
  tridiagonal J-matrix representation: I. Bound states.
  *Journal of Physics B: Atomic, Molecular and Optical Physics*, *40*,
  4245–4257. `https://doi.org/10.1088/0953-4075/40/21/011
  <https://doi.org/10.1088/0953-4075/40/21/011>`_.

* Saad, Y. (2011). *Numerical methods for large eigenvalue problems*
  (Rev. ed.). Society for Industrial and Applied Mathematics.
  `https://doi.org/10.1137/1.9781611970739
  <https://doi.org/10.1137/1.9781611970739>`_.

* Wikipedia contributors. Jacobi rotation. In *Wikipedia*.
  `https://en.wikipedia.org/wiki/Jacobi_rotation
  <https://en.wikipedia.org/wiki/Jacobi_rotation>`_.

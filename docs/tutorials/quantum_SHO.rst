Quantum Simple Harmonic Oscillator
==================================

Overview
--------

The quantum harmonic oscillator is one of the most important exactly solvable
problems in quantum mechanics. This tutorial uses Physika to calculate analytical
wavefunctions for a particle moving in the one-dimensional harmonic potential.
It then applies a finite-difference representation of the Hamiltonian to each
wavefunction and calculates its energy from an expectation-value integral. The user
chooses the number of states with ``N_levels``. The tutorial then calculates

.. math::

   \psi_0(x),\ \psi_1(x),\ \ldots,\
   \psi_{N_{\mathrm{levels}}-1}(x)

and their corresponding energies.

1. Definition of the problem
----------------------------

Consider a particle of mass :math:`m` that can move along one cartesian
coordinate :math:`x`. The particle experiences a restoring force that pulls it
toward the equilibrium position :math:`x=0`.

For a harmonic oscillator, the restoring force is proportional to the
displacement:

.. math::

   F(x)=-kx,

where :math:`k` is the force constant.

The corresponding potential energy is

.. math::

   V(x)=\frac{1}{2}kx^2=\frac{1}{2}m\omega^2x^2,

where :math:`\omega` is the angular frequency.

This potential has a parabolic shape. Its minimum is at :math:`x=0`, where
:math:`V(0)=0`, and it increases symmetrically on both sides of the origin.

.. admonition:: Problem statement

   Calculate the analytical wavefunctions of the 1D simple
   harmonic oscillator. Then apply the Hamiltonian to each wavefunction
   and calculate its energy from the expectation value

   .. math::

      E_n =
      \frac{
      \int \psi_n(x)\,\hat{H}\psi_n(x)\,dx
      }{
      \int |\psi_n(x)|^2\,dx
      }.

2. Physical model
-----------------

2.1 The time-independent Schrödinger equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a particle moving in one dimension, the time-independent Schrödinger
equation is

.. math::

   \hat{H}\psi_n(x)=E_n\psi_n(x),

where

* :math:`\hat{H}` is the Hamiltonian operator,
* :math:`\psi_n(x)` is the wavefunction of state :math:`n`,
* :math:`E_n` is the energy of that state, and
* :math:`n=0,1,2,\ldots` is the quantum number.

The Hamiltonian is the sum of kinetic and potential energy:

.. math::

   \hat{H}
   =
   -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}
   +V(x).

Substituting the harmonic potential gives

.. math::

   \left[
   -\frac{\hbar^2}{2m}\frac{d^2}{dx^2}
   +\frac{1}{2}m\omega^2x^2
   \right]\psi_n(x)
   =
   E_n\psi_n(x).

2.2 Dimensionless units
~~~~~~~~~~~~~~~~~~~~~~~

The general harmonic-oscillator formulas contain :math:`\hbar`, :math:`m`,
and :math:`\omega`. To keep the implementation simple, this tutorial uses
dimensionless units:

.. math::

   \hbar=1,\qquad m=1,\qquad \omega=1.

With these choices, the potential becomes

.. math::

   V(x)=\frac{x^2}{2},

and the Schrödinger equation becomes

.. math::

   \left[
   -\frac{1}{2}\frac{d^2}{dx^2}
   +\frac{x^2}{2}
   \right]\psi_n(x)
   =
   E_n\psi_n(x).

Using dimensionless units does not change the important physics. It removes
conversion factors and allows the tutorial to focus on the connection between
the equations, wavefunctions, and code.


2.3 The ground-state wavefunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In dimensionless units, the normalized ground-state wavefunction is

.. math::

   \psi_0(x)
   =
   \frac{1}{\pi^{1/4}}
   \exp\left(-\frac{x^2}{2}\right).

The normalization constant is

.. math::

   A=\frac{1}{\pi^{1/4}}.

It ensures that the total probability of finding the particle somewhere is
one:

.. math::

   \int_{-\infty}^{+\infty}|\psi_0(x)|^2\,dx=1.

2.4 Higher-state wavefunctions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first excited state can be obtained directly from the ground state:

.. math::

   \psi_1(x)=\sqrt{2}\,x\,\psi_0(x).

The factor :math:`x` makes :math:`\psi_1(0)=0`. It also changes sign across
the origin, so :math:`\psi_1` is an odd function.

Every higher normalized wavefunction can be generated from the preceding two
wavefunctions using a Hermite-function recurrence relation:

.. math::

   \psi_{n+1}(x)
   =
   \sqrt{\frac{2}{n+1}}\,x\,\psi_n(x)
   -
   \sqrt{\frac{n}{n+1}}\,\psi_{n-1}(x).

For example, setting :math:`n=1` gives

.. math::

   \psi_2(x)
   =
   \sqrt{\frac{2}{2}}\,x\,\psi_1(x)
   -
   \sqrt{\frac{1}{2}}\,\psi_0(x).

Once :math:`\psi_0` and :math:`\psi_1` have been calculated, this recurrence
relation produces :math:`\psi_2`. The next iteration uses :math:`\psi_1` and
:math:`\psi_2` to produce :math:`\psi_3`, and the process continues for any
requested number of levels.

2.5 Symmetry and nodes
~~~~~~~~~~~~~~~~~~~~~~

Because the potential is symmetric,

.. math::

   V(-x)=V(x),

the wavefunctions have definite parity:

.. math::

   \psi_n(-x)=
   \begin{cases}
   \phantom{-}\psi_n(x), & n=0,2,4,\ldots \quad \text{(even)},\\
   -\psi_n(x), & n=1,3,5,\ldots \quad \text{(odd)}.
   \end{cases}

The :math:`n`-th harmonic-oscillator wavefunction has exactly :math:`n` nodes.

2.6 Applying the Hamiltonian on the grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a wavefunction :math:`\psi_n`, the Hamiltonian acts as

.. math::

   \hat H\psi_n(x)
   =
   -\frac{\hbar^2}{2m}\frac{d^2\psi_n(x)}{dx^2}
   +V(x)\psi_n(x).

The potential term can be evaluated by ordinary multiplication at each grid
point. The kinetic term contains a second derivative, which must be
approximated.

Using the centered three-point finite-difference formula,

.. math::

   \left.\frac{d^2\psi_n}{dx^2}\right|_{x_i}
   \approx
   \frac{
   \psi_n(x_{i+1})-2\psi_n(x_i)+\psi_n(x_{i-1})
   }{(\Delta x)^2},

where :math:`\Delta x=x_{i+1}-x_i` is the uniform grid spacing. This expression
uses the wavefunction values at :math:`x_i` and its two neighboring grid points,
:math:`x_{i-1}` and :math:`x_{i+1}`. It is applied only at interior points,
while the boundary values are taken to be zero.

2.7 Calculating energy from an expectation value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An eigenfunction satisfies :math:`\hat H\psi_n=E_n\psi_n`. Its energy can be
calculated using the following expectation-value expression:

.. math::

   E_n
   =
   \frac{
   \displaystyle\int\psi_n^*(x)\hat H\psi_n(x)\,dx
   }{
   \displaystyle\int\psi_n^*(x)\psi_n(x)\,dx
   }.

The wavefunctions in this tutorial are real, so
:math:`\psi_n^*=\psi_n`. On the uniform grid, the integrals become sums:

.. math::

   E_n
   \approx
   \frac{
   \displaystyle\sum_{i=1}^{N_{\mathrm{grid}}-2}
   \psi_n(x_i)(\hat H\psi_n)_i\,\Delta x
   }{
   \displaystyle\sum_{i=1}^{N_{\mathrm{grid}}-2}
   \psi_n(x_i)^2\,\Delta x
   }.

Each term is an integrand value multiplied by the grid spacing
:math:`\Delta x`. Summing these terms gives a rectangular-rule approximation
to the integral. The denominator accounts for small normalization errors
caused by representing the wavefunctions on a finite grid.

3. Explanation of the Physika code
----------------------------------

3.1 Physical constants and numerical parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Physical constants in dimensionless units
   hbar: ℝ = 1.0
   mass: ℝ = 1.0
   angular_frequency: ℝ = 1.0
   pi: ℝ = 3.141592653589793

   # Numerical parameters
   N_levels: ℕ = 5
   x_max: ℝ = 6.0
   N_grid: ℕ = 601
   dx: ℝ = (2.0 * x_max) / (N_grid - 1)

The annotation ``ℝ`` declares a real number. Setting :math:`\hbar`, :math:`m`,
and :math:`\omega` to one selects the dimensionless units introduced in
Section 2.2. ``pi`` is needed for the normalization constant of the
ground-state wavefunction.

The annotation ``ℕ`` declares a natural number. ``N_levels = 5`` requests the
states :math:`n=0,1,2,3,4`. ``x_max`` sets the finite domain from
:math:`-x_{\max}` to :math:`+x_{\max}`, and ``N_grid`` is the number of points
used to represent that interval.

Because ``N_grid`` points contain ``N_grid - 1`` intervals, the grid spacing is

.. math::

   \Delta x=\frac{2x_{\max}}{N_{\mathrm{grid}}-1}
   =\frac{12}{600}=0.02.

3.2 Position grid and harmonic potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Position grid and harmonic potential
   position: ℝ[N_grid] = for i: ℕ(N_grid) -> -x_max + i * dx
   potential: ℝ[N_grid] = for i: ℕ(N_grid) -> (0.5 * mass * angular_frequency
       * angular_frequency * position[i] * position[i])

For each grid index :math:`i`, the first line stores

.. math::

   x_i=-x_{\max}+i\Delta x.

Thus ``position[0]`` is :math:`-6`, ``position[300]`` is :math:`0`, and
``position[600]`` is :math:`+6`. The second line evaluates the harmonic
potential at every position:

.. math::

   V(x_i)=\frac{1}{2}m\omega^2x_i^2.


3.3 Initialize all result arrays and scalar work variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initialize all result arrays
   wavefunctions: ℝ[N_levels,N_grid] = (for n: ℕ(N_levels) -> for i: ℕ(N_grid) -> (n + i) * 0.0)
   hamiltonian_wavefunctions: ℝ[N_levels,N_grid] = (for n: ℕ(N_levels) ->
      for i: ℕ(N_grid) -> (n + i) * 0.0)
   energies: ℝ[N_levels] = for n: ℕ(N_levels) -> n * 0.0

   # Initialize scalar work variables
   x: ℝ = 0.0
   gaussian: ℝ = 0.0
   n_real: ℝ = 0.0
   next_n_real: ℝ = 0.0
   first_coefficient: ℝ = 0.0
   second_coefficient: ℝ = 0.0
   second_derivative: ℝ = 0.0
   kinetic_part: ℝ = 0.0
   potential_part: ℝ = 0.0
   energy_numerator: ℝ = 0.0
   normalization_integral: ℝ = 0.0

All result arrays are created before the calculations begin.
``wavefunctions[n,i]`` will store :math:`\psi_n(x_i)`, while
``hamiltonian_wavefunctions[n,i]`` will store
:math:`(\hat H\psi_n)(x_i)`. Each two-dimensional array therefore has
``N_levels`` rows and ``N_grid`` columns. ``energies[n]`` will store the
calculated energy of state :math:`n`.

3.4 Calculate the analytical ground-state wavefunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate the analytical ground-state wavefunction
   normalization_constant: ℝ = 1.0 / (pi ** 0.25)
   for i: ℕ(N_grid):
       x = position[i]
       gaussian = exp(-0.5 * x * x)
       wavefunctions[0,i] = normalization_constant * gaussian

The normalization constant is :math:`\pi^{-1/4}`. At every grid
position, the loop evaluates the Gaussian :math:`e^{-x_i^2/2}` and stores

.. math::

   \psi_0(x_i)=\pi^{-1/4}e^{-x_i^2/2}.

After the loop, the first row of ``wavefunctions`` contains the complete
ground-state curve on the selected grid.

3.5 Generate the first excited-state wavefunction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate the first excited-state wavefunction
   one_level: ℕ = 1
   if N_levels > one_level:
       for i: ℕ(N_grid):
           x = position[i]
           wavefunctions[1,i] = (sqrt(2.0) * x * wavefunctions[0,i])

The first excited state is calculated from

.. math::

   \psi_1(x_i)=\sqrt{2}\,x_i\psi_0(x_i).

The condition is required because the row ``wavefunctions[1,i]`` exists only
when at least two states are requested. If ``N_levels = 1``, the program needs
only :math:`\psi_0` and skips this block.

3.6 Generate all remaining states using the recurrence relation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate all remaining states using the recurrence relation
   two_levels: ℕ = 2
   if N_levels > two_levels:
       for n: ℕ(1, N_levels - 1):
           n_real = n * 1.0
           next_n_real = n_real + 1.0
           first_coefficient = sqrt(2.0 / next_n_real)
           second_coefficient = sqrt(n_real / next_n_real)
           for i: ℕ(N_grid):
               x = position[i]
               wavefunctions[n + 1,i] = (first_coefficient * x * wavefunctions[n,i]
                   - second_coefficient * wavefunctions[n - 1,i])

Once :math:`\psi_0` and :math:`\psi_1` are available, the three-term
recurrence relation generates every higher normalized harmonic-oscillator
wavefunction:

.. math::

   \psi_{n+1}(x)
   =
   \sqrt{\frac{2}{n+1}}x\psi_n(x)
   -
   \sqrt{\frac{n}{n+1}}\psi_{n-1}(x).

``n_real`` converts the natural-number loop index to a real number for division
and square roots. ``first_coefficient`` and ``second_coefficient`` represent
the two square-root factors in the equation. The outer loop begins at
:math:`n=1`, so its first iteration generates :math:`\psi_2`; subsequent
iterations generate :math:`\psi_3`, :math:`\psi_4`, and so on.

3.7 Apply the finite-difference Hamiltonian to each state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Apply the finite-difference Hamiltonian to each state
   for n: ℕ(N_levels):
       for i: ℕ(1, N_grid - 1):
           second_derivative = (wavefunctions[n,i + 1] - 2.0 * wavefunctions[n,i]
               + wavefunctions[n,i - 1]) / (dx * dx)
           kinetic_part = (-(hbar * hbar) / (2.0 * mass) * second_derivative)
           potential_part = (potential[i] * wavefunctions[n,i])
           hamiltonian_wavefunctions[n,i] = (kinetic_part + potential_part)

The outer loop selects a state, and the inner loop visits every interior grid
point. The centered finite-difference expression approximates
:math:`d^2\psi_n/dx^2`. The range begins at ``1`` and stops before
``N_grid - 1`` so that every evaluated point has both ``i - 1`` and ``i + 1``
neighbors.

The kinetic and potential contributions are then added:

.. math::

   (\hat H\psi_n)(x_i)
   =
   -\frac{\hbar^2}{2m}
   \left.\frac{d^2\psi_n}{dx^2}\right|_{x_i}
   +V(x_i)\psi_n(x_i).

The result is stored in ``hamiltonian_wavefunctions[n,i]``. Its two boundary
entries remain at their initialized value of zero.

3.8 Calculate each energy from its expectation value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate each energy from its expectation value
   for n: ℕ(N_levels):
       energy_numerator = 0.0
       normalization_integral = 0.0
       for i: ℕ(1, N_grid - 1):
           energy_numerator = (energy_numerator
               + wavefunctions[n,i] * hamiltonian_wavefunctions[n,i] * dx)
           normalization_integral = (normalization_integral
               + wavefunctions[n,i] * wavefunctions[n,i] * dx)
       energies[n] = (energy_numerator / normalization_integral)
       physika_print(energies[n])

Before integrating a new state, both accumulators are reset to zero. The inner
loop performs rectangular numerical integration. Its first update accumulates

.. math::

   \psi_n(x_i)(\hat H\psi_n)(x_i)\Delta x,

and its second update accumulates

.. math::

   \psi_n(x_i)^2\Delta x.

Their ratio gives the expectation value of the energy and remains reliable
even if a grid-represented wavefunction is not perfectly normalized.
``physika_print`` displays each energy after it is calculated.

3.9 Plotting (optional)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plotting (optional)
   physika_plot(position, wavefunctions, energies, N_levels, potential)

The plotting call passes the position grid, wavefunctions, calculated energies,
number of levels, and potential to the plotting helper. Add the custom
``physika_plot`` helper to ``runtime.py`` to plot the results.

.. code-block:: python

   def physika_plot(position: object, wavefunctions: object, energies: object, N_levels: object, potential: object) -> None:
       import matplotlib.pyplot as plt
       import numpy as np
       import torch
       x = position.detach().cpu().numpy() if isinstance(position, torch.Tensor) else np.asarray(position)
       psi = wavefunctions.detach().cpu().numpy() if isinstance(wavefunctions, torch.Tensor) else np.asarray(wavefunctions)
       energy_values = energies.detach().cpu().numpy() if isinstance(energies, torch.Tensor) else np.asarray(energies)
       potential_values = potential.detach().cpu().numpy() if isinstance(potential, torch.Tensor) else np.asarray(potential)
       number_of_levels = int(N_levels.detach().cpu().item()) if isinstance(N_levels, torch.Tensor) else int(N_levels)
       x = np.asarray(x, dtype=float).reshape(-1)
       psi = np.asarray(psi, dtype=float)
       energy_values = np.asarray(energy_values, dtype=float).reshape(-1)
       potential_values = np.asarray(potential_values, dtype=float).reshape(-1)
       psi = psi[:number_of_levels]
       energy_values = energy_values[:number_of_levels]
       if number_of_levels > 1:
           positive_differences = np.diff(np.sort(energy_values))
           positive_differences = positive_differences[positive_differences > 0.0]
           energy_spacing = float(np.min(positive_differences)) if positive_differences.size else 1.0
       else:
           energy_spacing = max(1.0, abs(float(energy_values[0])))

       maximum_wavefunction = float(np.max(np.abs(psi)))
       wavefunction_scale = 0.35 * energy_spacing / maximum_wavefunction if maximum_wavefunction > 0.0 else 1.0
       maximum_energy = float(np.max(energy_values))
       minimum_energy = float(np.min(energy_values))
       maximum_displayed_energy = maximum_energy + 0.75 * energy_spacing
       minimum_displayed_energy = min(0.0, minimum_energy - 0.50 * energy_spacing)
       visible_indices = np.where(potential_values <= maximum_displayed_energy)[0]
       if visible_indices.size:
           padding = max(2, int(0.05 * x.size))
           first_visible = max(0, int(visible_indices[0]) - padding)
           last_visible = min(x.size - 1, int(visible_indices[-1]) + padding)
           x_min, x_max = float(x[first_visible]), float(x[last_visible])
       else:
           x_min, x_max = float(np.min(x)), float(np.max(x))
       fig, ax = plt.subplots(figsize=(5, 4))
       colors = plt.cm.viridis(np.linspace(0.05, 0.90, number_of_levels))
       ax.plot(x, potential_values, color="black", linewidth=2.0, label=r"$V(x)=x^2/2$", zorder=1)
       for level in range(number_of_levels):
           energy = float(energy_values[level])
           color = colors[level]
           ax.hlines(energy, x_min, x_max, color=color, linewidth=1.0, linestyle="--", alpha=0.60, zorder=2)
           ax.plot(x, energy + wavefunction_scale * psi[level], color=color, linewidth=2.2, zorder=3)
           ax.text(0.97, energy, rf"$E_{{{level}}}={energy:.1f}$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=color, fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.5}, zorder=4)
       ax.set_xlim(x_min - 1.0, x_max + 2.5)
       ax.set_ylim(minimum_displayed_energy, maximum_displayed_energy + 0.5)
       ax.set_xlabel(r"Position, $x$", fontsize=10)
       ax.set_ylabel(r"Energy, $E/(\hbar\omega)$, and wavefunctions, $\psi_n(x)$", fontsize=10)
       ax.legend(loc="upper right", frameon=True, fontsize=8)
       fig.tight_layout()
       # plt.savefig("quantum_SHO.png", dpi=300, bbox_inches="tight")
       plt.show()
       plt.close(fig)

4. Expected results
-------------------

For ``N_levels = 5``,

.. admonition:: Results

   .. code-block:: text

      ✓ No type errors found
      0.49998700618743896 ∈ ℝ
      1.4999382495880127 ∈ ℝ
      2.4998371601104736 ∈ ℝ
      3.4996869564056396 ∈ ℝ
      4.499486446380615 ∈ ℝ

The first line confirms that the Physika code passed type checking. Each
subsequent line gives the calculated dimensionless energy
:math:`E_n/(\hbar\omega)` for one harmonic-oscillator state:

- :math:`0.499987`: ground-state energy, :math:`E_0`
- :math:`1.499938`: first excited-state energy, :math:`E_1`
- :math:`2.499837`: second excited-state energy, :math:`E_2`
- :math:`3.499687`: third excited-state energy, :math:`E_3`
- :math:`4.499486`: fourth excited-state energy, :math:`E_4`

These numerical values closely agree with the exact energy relation

.. math::

   \frac{E_n}{\hbar\omega}=n+\frac{1}{2}.
   
.. figure:: ../_static/tutorial_files/quantum_SHO.png
   :align: center
   :width: 70%
   :alt: Harmonic-oscillator energy levels, wavefunctions, and parabolic potential

   **Figure.** The calculated harmonic-oscillator energies and wavefunctions,
   plotted with the dimensionless potential :math:`V(x)/(\hbar\omega)=x^2/2`.

The black parabola represents the harmonic potential. The colored horizontal
lines show the calculated energy levels, while the colored curves show the
wavefunctions shifted to their corresponding energies. The wavefunctions
alternate between even and odd parity, and each state :math:`\psi_n` has exactly
:math:`n` nodes. Their decay to zero near :math:`x=\pm 6` supports the finite-domain
boundary approximation.

Source code
-----------

The complete Physika implementation is provided in
``tutorials/quantum_SHO.phyk``.

.. code-block:: python

   # Physical constants in dimensionless units
   hbar: ℝ = 1.0
   mass: ℝ = 1.0
   angular_frequency: ℝ = 1.0
   pi: ℝ = 3.141592653589793

   # Numerical parameters
   N_levels: ℕ = 5
   x_max: ℝ = 6.0
   N_grid: ℕ = 601
   dx: ℝ = (2.0 * x_max) / (N_grid - 1)

   # Position grid and harmonic potential
   position: ℝ[N_grid] = for i: ℕ(N_grid) -> -x_max + i * dx
   potential: ℝ[N_grid] = for i: ℕ(N_grid) -> (0.5 * mass * angular_frequency
       * angular_frequency * position[i] * position[i])

   # Initialize all result arrays
   wavefunctions: ℝ[N_levels,N_grid] = (for n: ℕ(N_levels) -> for i: ℕ(N_grid) -> (n + i) * 0.0)
   hamiltonian_wavefunctions: ℝ[N_levels,N_grid] = (for n: ℕ(N_levels) ->
      for i: ℕ(N_grid) -> (n + i) * 0.0)
   energies: ℝ[N_levels] = for n: ℕ(N_levels) -> n * 0.0

   # Initialize scalar work variables
   x: ℝ = 0.0
   gaussian: ℝ = 0.0
   n_real: ℝ = 0.0
   next_n_real: ℝ = 0.0
   first_coefficient: ℝ = 0.0
   second_coefficient: ℝ = 0.0
   second_derivative: ℝ = 0.0
   kinetic_part: ℝ = 0.0
   potential_part: ℝ = 0.0
   energy_numerator: ℝ = 0.0
   normalization_integral: ℝ = 0.0

   # Calculate the analytical ground-state wavefunction
   normalization_constant: ℝ = 1.0 / (pi ** 0.25)
   for i: ℕ(N_grid):
       x = position[i]
       gaussian = exp(-0.5 * x * x)
       wavefunctions[0,i] = normalization_constant * gaussian

   # Generate the first excited-state wavefunction
   one_level: ℕ = 1
   if N_levels > one_level:
       for i: ℕ(N_grid):
           x = position[i]
           wavefunctions[1,i] = (sqrt(2.0) * x * wavefunctions[0,i])

   # Generate all remaining states using the recurrence relation
   two_levels: ℕ = 2
   if N_levels > two_levels:
       for n: ℕ(1, N_levels - 1):
           n_real = n * 1.0
           next_n_real = n_real + 1.0
           first_coefficient = sqrt(2.0 / next_n_real)
           second_coefficient = sqrt(n_real / next_n_real)
           for i: ℕ(N_grid):
               x = position[i]
               wavefunctions[n + 1,i] = (first_coefficient * x * wavefunctions[n,i]
                   - second_coefficient * wavefunctions[n - 1,i])

   # Apply the finite-difference Hamiltonian to each state
   for n: ℕ(N_levels):
       for i: ℕ(1, N_grid - 1):
           second_derivative = (wavefunctions[n,i + 1] - 2.0 * wavefunctions[n,i]
               + wavefunctions[n,i - 1]) / (dx * dx)
           kinetic_part = (-(hbar * hbar) / (2.0 * mass) * second_derivative)
           potential_part = (potential[i] * wavefunctions[n,i])
           hamiltonian_wavefunctions[n,i] = (kinetic_part + potential_part)

   # Calculate each energy from its expectation value
   for n: ℕ(N_levels):
       energy_numerator = 0.0
       normalization_integral = 0.0
       for i: ℕ(1, N_grid - 1):
           energy_numerator = (energy_numerator
               + wavefunctions[n,i] * hamiltonian_wavefunctions[n,i] * dx)
           normalization_integral = (normalization_integral
               + wavefunctions[n,i] * wavefunctions[n,i] * dx)
       energies[n] = (energy_numerator / normalization_integral)
       physika_print(energies[n])

   # Plotting (optional)
   # physika_plot(position, wavefunctions, energies, N_levels, potential)

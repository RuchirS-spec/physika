Parameter Learning for the Hodgkin–Huxley Neuron Model
======================================================

In this tutorial we learn the maximal channel conductances of the **Hodgkin–Huxley
model** (Hodgkin & Huxley, 1952) — the founding conductance-based description of the
nerve action potential, and the work that earned the 1963 Nobel Prize in Physiology
or Medicine. It is a direct sibling of the :doc:`/tutorials/fitzhugh_nagumo` tutorial
(and of the Lotka–Volterra and SIR tutorials): the scaffolding — RK4 stepper,
trajectory solver, full-trajectory adjoint — is identical, and only the model changes.
As in the SIR tutorial we fit from a **partial observation**: we recover the
conductances from the membrane voltage :math:`V(t)` alone, the one quantity an
electrophysiologist actually records.

The idea worth reading carefully here is one of **conditioning**. The smooth systems
in the sibling tutorials recover their parameters essentially exactly. Hodgkin–Huxley
does not — and that is the point. It is *stiff*, and its conductances *compensate* one
another: gradient descent drives the voltage fit down beautifully,
yet leaves all three conductances meaningfully off. A low loss does not
imply recovered parameters. Crucially, the gradients Physika computes are **exact**
(they agree with finite differences to float precision); the difficulty lives in the
geometry of the loss, not in the automatic differentiation. Learning to tell those two
things apart is the whole lesson.


The Equations
-------------

The membrane potential :math:`V` is driven by three ionic currents — sodium, potassium
and a leak — each a conductance times a driving force :math:`(V - E_\mathrm{ion})`. Three
dimensionless *gating variables* :math:`m, h, n \in [0,1]` open and close the channels:

.. math::

   C_m \frac{dV}{dt} = I_\mathrm{app}
     - \bar{g}_{\mathrm{Na}}\, m^3 h\, (V - E_{\mathrm{Na}})
     - \bar{g}_{\mathrm{K}}\, n^4\, (V - E_{\mathrm{K}})
     - \bar{g}_{\mathrm{L}}\, (V - E_{\mathrm{L}})

   \frac{dm}{dt} = \alpha_m(V)\,(1-m) - \beta_m(V)\,m

   \frac{dh}{dt} = \alpha_h(V)\,(1-h) - \beta_h(V)\,h

   \frac{dn}{dt} = \alpha_n(V)\,(1-n) - \beta_n(V)\,n

The voltage-dependent rates (in the standard convention, :math:`V` in mV, resting near
:math:`-65`) are

.. math::

   \alpha_m = \frac{0.1\,(V+40)}{1 - e^{-(V+40)/10}}, \quad
   \beta_m = 4\,e^{-(V+65)/18}, \quad
   \alpha_h = 0.07\,e^{-(V+65)/20},

   \beta_h = \frac{1}{1 + e^{-(V+35)/10}}, \quad
   \alpha_n = \frac{0.01\,(V+55)}{1 - e^{-(V+55)/10}}, \quad
   \beta_n = 0.125\,e^{-(V+65)/80}.

We fix the reversal potentials and capacitance
(:math:`C_m = 1`, :math:`E_{\mathrm{Na}} = 50`, :math:`E_{\mathrm{K}} = -77`,
:math:`E_{\mathrm{L}} = -54.387` mV) and a constant drive
:math:`I_\mathrm{app} = 10\ \mu\mathrm{A/cm}^2` that produces repetitive firing. The
parameters we learn are the maximal conductances
:math:`\theta = [\bar{g}_{\mathrm{Na}}, \bar{g}_{\mathrm{K}}, \bar{g}_{\mathrm{L}}]`,
with true values :math:`[120, 36, 0.3]\ \mathrm{mS/cm}^2`.

.. note::
   :math:`\alpha_m` and :math:`\alpha_n` each contain a removable :math:`0/0`
   singularity (at :math:`V=-40` and :math:`V=-55`). We return the analytic limit
   there (:math:`\alpha_m \to 1`, :math:`\alpha_n \to 0.1`), so the rate functions are
   safe to evaluate and to differentiate everywhere.


Why conductances are hard to identify
-------------------------------------

Taken one at a time, the conductances matter enormously: raising
:math:`\bar{g}_{\mathrm{Na}}` alone by 1.7% already shifts the voltage trace by 3.6 mV
RMSE, and the minimum along any single axis is sharp. The difficulty is that the
conductances **compensate** one another. Slightly less sodium current can be offset by
slightly less potassium current and a slightly larger leak, leaving an almost identical
spike train — with :math:`\bar{g}_{\mathrm{Na}}` 8% low, a compensating
:math:`\bar{g}_{\mathrm{K}}` and :math:`\bar{g}_{\mathrm{L}}` bring the error back from
18.7 mV to 0.8 mV RMSE. The loss therefore has a long, narrow, shallow valley: a curved
family of conductance combinations that all fit the voltage almost equally well. This
degeneracy is a genuine, well-known property of conductance-based neuron models —
not an artefact of Physika. The leak :math:`\bar{g}_{\mathrm{L}}` and the two spike
conductances are also on very different scales (:math:`0.3` vs :math:`120`), so we use
**Adam** rather than plain gradient descent, exactly as in the repressilator tutorial.


Helper functions
----------------

We reuse the same dynamic-array helpers as the sibling tutorials
(``zero_1d_array`` / ``get_1d_array_length`` / ``append``); Physika arrays are
fixed-shape, so ``append`` allocates a new, one-longer array and copies into it:

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def append(x: ℝ[m], var: ℝ): ℝ[n]:
        new_length: ℝ = get_1d_array_length(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results


Step 1: Rate functions and the ODE
----------------------------------

Each rate is a small helper. The two with a removable singularity guard it with a
statement-``if`` that returns the analytic limit; because the branch is chosen on the
concrete value of :math:`V`, this differentiates cleanly. Note the ``exp(0.0 - x)``
spelling: it avoids a leading unary minus, which the parser does not accept.

.. code-block:: text

    Cm: ℝ = 1.0
    ENa: ℝ = 50.0
    EK: ℝ = -77.0
    EL: ℝ = -54.387
    Iapp: ℝ = 10.0

    def alpha_m(V: ℝ): ℝ:
        x: ℝ = V + 40.0
        if abs(x) < 0.0001:
            return 1.0
        else:
            return 0.1 * x / (1.0 - exp(0.0 - x / 10.0))

    def beta_m(V: ℝ): ℝ:
        return 4.0 * exp(0.0 - (V + 65.0) / 18.0)

    def alpha_h(V: ℝ): ℝ:
        return 0.07 * exp(0.0 - (V + 65.0) / 20.0)

    def beta_h(V: ℝ): ℝ:
        return 1.0 / (1.0 + exp(0.0 - (V + 35.0) / 10.0))

    def alpha_n(V: ℝ): ℝ:
        x: ℝ = V + 55.0
        if abs(x) < 0.0001:
            return 0.1
        else:
            return 0.01 * x / (1.0 - exp(0.0 - x / 10.0))

    def beta_n(V: ℝ): ℝ:
        return 0.125 * exp(0.0 - (V + 65.0) / 80.0)

``f`` unpacks the four-dimensional state ``[V, mg, hg, ng]`` (the gating variables are
named ``mg``/``hg``/``ng`` because a bare ``m`` is already used as an array-length
variable) and returns the derivatives. The channel powers :math:`m^3` and :math:`n^4`
are written as explicit products:

.. code-block:: text

    def f(state: ℝ[4], θ: ℝ[3]): ℝ[4]:
        V: ℝ = state[0]
        mg: ℝ = state[1]
        hg: ℝ = state[2]
        ng: ℝ = state[3]
        gNa: ℝ = θ[0]
        gK: ℝ = θ[1]
        gL: ℝ = θ[2]
        am: ℝ = alpha_m(V)
        bm: ℝ = beta_m(V)
        ah: ℝ = alpha_h(V)
        bh: ℝ = beta_h(V)
        an: ℝ = alpha_n(V)
        bn: ℝ = beta_n(V)
        iNa: ℝ = gNa * mg * mg * mg * hg * (V - ENa)
        iK: ℝ = gK * ng * ng * ng * ng * (V - EK)
        iL: ℝ = gL * (V - EL)
        dV: ℝ = (Iapp - iNa - iK - iL) / Cm
        dmg: ℝ = am * (1.0 - mg) - bm * mg
        dhg: ℝ = ah * (1.0 - hg) - bh * hg
        dng: ℝ = an * (1.0 - ng) - bn * ng
        return [dV, dmg, dhg, dng]


Step 2: Build the RK4 Solver
----------------------------

We integrate with the classic fourth-order Runge–Kutta method — identical to the
sibling tutorials, over the four-dimensional state:

.. code-block:: text

    def rk4_step(state: ℝ[4], θ: ℝ[3]): ℝ[4]:
        k1: ℝ[4] = f(state, θ)
        k2_state: ℝ[4] = state + 0.5 * dt * k1
        k2: ℝ[4] = f(k2_state, θ)
        k3_state: ℝ[4] = state + 0.5 * dt * k2
        k3: ℝ[4] = f(k3_state, θ)
        k4_state: ℝ[4] = state + dt * k3
        k4: ℝ[4] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


Step 3: Build the Trajectory Solver
-----------------------------------

We integrate forward 400 steps at :math:`dt = 0.05` ms (a 20 ms window, enough for a
couple of spikes) from rest, starting the gates at their steady-state values for
:math:`V = -65` mV, and collect all four trajectories:

.. code-block:: text

    dt: ℝ = 0.05
    timesteps: ℝ = 400

    def solver(θ: ℝ[3]): ℝ[4, m]:
        state: ℝ[4] = [-65.0, 0.0529, 0.5961, 0.3177]
        V_array: ℝ[1] = [-65.0]
        mg_array: ℝ[1] = [0.0529]
        hg_array: ℝ[1] = [0.5961]
        ng_array: ℝ[1] = [0.3177]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            V = results[0]
            mg = results[1]
            hg = results[2]
            ng = results[3]
            V_array = append(V_array, V)
            mg_array = append(mg_array, mg)
            hg_array = append(hg_array, hg)
            ng_array = append(ng_array, ng)
            state = results
        return [V_array, mg_array, hg_array, ng_array]


Step 4: Generate Ground Truth Data
----------------------------------

We simulate the standard squid-axon conductances and keep the voltage trace as the
data to fit:

.. code-block:: text

    true_theta: ℝ[3] = [120.0, 36.0, 0.3]
    true_results: ℝ[4, m] = solver(true_theta)
    true_V: ℝ[m] = true_results[0]


Step 5: Adjoint Gradient from the Observed Voltage
--------------------------------------------------

We fit the **voltage trace only**, with a mean-squared-error loss over the :math:`V`
samples,

.. math::

    \mathcal{L}(\theta) = \frac{1}{2m} \sum_{k=0}^{m-1}
        \left( V_k - V_k^{\mathrm{true}} \right)^2 ,

and compute its gradient with the adjoint (reverse-mode) method. The co-state is
seeded at the terminal step and propagated backwards with a per-step *running cost* —
but the residual lives **only on the** :math:`V` **component**; the ``m``, ``h``,
``n`` gating entries are ``0.0`` because those states are never observed:

.. math::

    s_k = \Big[\,\tfrac{1}{m}\big(V_k - V_k^{\mathrm{true}}\big),\; 0,\; 0,\; 0\,\Big]
          + s_{k+1}\, J_{\mathrm{state}}(y_k),

where the RK4 Jacobians come from ``grad`` and the parameter gradient accumulates
:math:`L \mathrel{+}= s\,J_\theta` along the sweep:

.. code-block:: text

    def adjoint_grad(θ: ℝ[3]): ℝ[n]:
        states: ℝ[4, m] = solver(θ)
        V_array: ℝ[m] = states[0]
        mg_array: ℝ[m] = states[1]
        hg_array: ℝ[m] = states[2]
        ng_array: ℝ[m] = states[3]
        m: ℝ = get_1d_array_length(V_array)
        s: ℝ[4] = [
            (V_array[m-1] - true_V[m-1]) / m,
            0.0,
            0.0,
            0.0
        ]
        L: ℝ[3] = zero_1d_array(3)
        for i:ℕ(m-1):
            idx = m - 2 - i
            V = V_array[idx]
            mg = mg_array[idx]
            hg = hg_array[idx]
            ng = ng_array[idx]
            state = [V, mg, hg, ng]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [(V_array[idx] - true_V[idx]) / m, 0.0, 0.0, 0.0]
            s = residual + (s @ J_state)
        return L

This hand-rolled adjoint agrees with autograd (and with finite differences) to float
precision — the sensitivities through the stiff, exponential rate functions are
computed correctly.


Step 6: Train with Adam
-----------------------

Because the conductances span more than two orders of magnitude, we hand-roll
bias-corrected **Adam** (as in the repressilator tutorial); the ``t_adam`` counter is
the bias-correction step (the loop variable ``i`` cannot be used in arithmetic):

.. code-block:: text

    θ: ℝ[3] = [100.0, 30.0, 0.5]
    learning_rate: ℝ = 0.2
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[3] = [0.0, 0.0, 0.0]
    v_adam: ℝ[3] = [0.0, 0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 1

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        θ = θ - learning_rate * mhat / (sqrt(vhat) + eps_adam)

    pred_results = solver(θ)

.. note::
   The committed ``tutorials/hodgkin_huxley.phyk`` sets ``epochs = 1`` so the test
   suite runs quickly. Raise it (e.g. ``1500``) to actually fit the model.


Step 7: Results
---------------

After 1500 Adam steps the learned voltage trace is nearly indistinguishable from the
truth (fit RMSE :math:`\approx 0.8` mV, and the loss falls ~156×). Yet the recovered
conductances — :math:`[108.6, 33.1, 0.32]` against the true :math:`[120, 36, 0.3]` —
tell a different story:

.. figure:: /_static/tutorial_files/output_hodgkin_huxley.png
   :alt: Hodgkin-Huxley voltage fit is excellent yet conductances remain off
   :align: center
   :width: 900px

   Left (A): the learned trajectory (orange, dashed) lands on top of the truth
   (black) while the initial guess (grey) is clearly wrong. Middle (B): the
   trajectory-MSE loss on :math:`V(t)` falls by roughly two orders of magnitude.
   Right (C): despite that, the recovered conductances remain ~8–9% away from their
   true values — they have drifted along the compensating valley of the loss.

This is the take-away, and it is the opposite of the SIR tutorial's happy ending: a
partial observation is *not* always enough. The voltage fit is superb and the loss is
small, yet all three conductances are still ~8–9% out, because their errors compensate
one another: a 1.7% error in :math:`\bar{g}_{\mathrm{Na}}` *on its own* costs 3.6 mV
RMSE, while the learned set — more than five times further off in
:math:`\bar{g}_{\mathrm{Na}}` — costs only 0.8 mV because the other two conductances
have moved to absorb it. Pushing the learning rate to force faster progress makes the tiny
leak conductance overshoot below zero and the integrator diverge — a reminder that the
system is stiff. The remedy is a matter of *experimental design and conditioning*
(richer stimuli, multiple current levels, observing more than voltage, or priors on
the conductances), not of the differentiation, which is exact throughout. Telling
"good fit" apart from "recovered parameters" is the skill this model teaches.

To visualise the fit yourself, add a plotting helper to ``physika/runtime.py`` as in
the FitzHugh–Nagumo tutorial and plot ``true_results`` against ``pred_results``.


Full Code
---------

.. code-block:: text

    def zero_1d_array(len: ℝ): ℝ[m]:
        results: ℝ[len] = for i: ℕ(len) -> i*0
        return results

    def get_1d_array_length(x: ℝ[m]): ℝ:
        total: ℝ = 0
        temp: ℝ = 0
        for i:
            temp = x[i]
            total += 1
        return total

    def append(x: ℝ[m], var: ℝ): ℝ[n]:
        new_length: ℝ = get_1d_array_length(x) + 1
        results: ℝ[new_length] = zero_1d_array(new_length)
        len_x: ℕ = get_1d_array_length(x)
        for i:ℕ(new_length):
            if i<len_x:
                results[i] = x[i]
            else:
                results[i] = var
        return results

    Cm: ℝ = 1.0
    ENa: ℝ = 50.0
    EK: ℝ = -77.0
    EL: ℝ = -54.387
    Iapp: ℝ = 10.0

    def alpha_m(V: ℝ): ℝ:
        x: ℝ = V + 40.0
        if abs(x) < 0.0001:
            return 1.0
        else:
            return 0.1 * x / (1.0 - exp(0.0 - x / 10.0))

    def beta_m(V: ℝ): ℝ:
        return 4.0 * exp(0.0 - (V + 65.0) / 18.0)

    def alpha_h(V: ℝ): ℝ:
        return 0.07 * exp(0.0 - (V + 65.0) / 20.0)

    def beta_h(V: ℝ): ℝ:
        return 1.0 / (1.0 + exp(0.0 - (V + 35.0) / 10.0))

    def alpha_n(V: ℝ): ℝ:
        x: ℝ = V + 55.0
        if abs(x) < 0.0001:
            return 0.1
        else:
            return 0.01 * x / (1.0 - exp(0.0 - x / 10.0))

    def beta_n(V: ℝ): ℝ:
        return 0.125 * exp(0.0 - (V + 65.0) / 80.0)

    def f(state: ℝ[4], θ: ℝ[3]): ℝ[4]:
        V: ℝ = state[0]
        mg: ℝ = state[1]
        hg: ℝ = state[2]
        ng: ℝ = state[3]
        gNa: ℝ = θ[0]
        gK: ℝ = θ[1]
        gL: ℝ = θ[2]
        am: ℝ = alpha_m(V)
        bm: ℝ = beta_m(V)
        ah: ℝ = alpha_h(V)
        bh: ℝ = beta_h(V)
        an: ℝ = alpha_n(V)
        bn: ℝ = beta_n(V)
        iNa: ℝ = gNa * mg * mg * mg * hg * (V - ENa)
        iK: ℝ = gK * ng * ng * ng * ng * (V - EK)
        iL: ℝ = gL * (V - EL)
        dV: ℝ = (Iapp - iNa - iK - iL) / Cm
        dmg: ℝ = am * (1.0 - mg) - bm * mg
        dhg: ℝ = ah * (1.0 - hg) - bh * hg
        dng: ℝ = an * (1.0 - ng) - bn * ng
        return [dV, dmg, dhg, dng]

    def rk4_step(state: ℝ[4], θ: ℝ[3]): ℝ[4]:
        k1: ℝ[4] = f(state, θ)
        k2_state: ℝ[4] = state + 0.5 * dt * k1
        k2: ℝ[4] = f(k2_state, θ)
        k3_state: ℝ[4] = state + 0.5 * dt * k2
        k3: ℝ[4] = f(k3_state, θ)
        k4_state: ℝ[4] = state + dt * k3
        k4: ℝ[4] = f(k4_state, θ)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    dt: ℝ = 0.05
    timesteps: ℝ = 400

    def solver(θ: ℝ[3]): ℝ[4, m]:
        state: ℝ[4] = [-65.0, 0.0529, 0.5961, 0.3177]
        V_array: ℝ[1] = [-65.0]
        mg_array: ℝ[1] = [0.0529]
        hg_array: ℝ[1] = [0.5961]
        ng_array: ℝ[1] = [0.3177]
        for i:ℕ(timesteps):
            results = rk4_step(state, θ)
            V = results[0]
            mg = results[1]
            hg = results[2]
            ng = results[3]
            V_array = append(V_array, V)
            mg_array = append(mg_array, mg)
            hg_array = append(hg_array, hg)
            ng_array = append(ng_array, ng)
            state = results
        return [V_array, mg_array, hg_array, ng_array]

    true_theta: ℝ[3] = [120.0, 36.0, 0.3]
    true_results: ℝ[4, m] = solver(true_theta)
    true_V: ℝ[m] = true_results[0]

    def adjoint_grad(θ: ℝ[3]): ℝ[n]:
        states: ℝ[4, m] = solver(θ)
        V_array: ℝ[m] = states[0]
        mg_array: ℝ[m] = states[1]
        hg_array: ℝ[m] = states[2]
        ng_array: ℝ[m] = states[3]
        m: ℝ = get_1d_array_length(V_array)
        s: ℝ[4] = [
            (V_array[m-1] - true_V[m-1]) / m,
            0.0,
            0.0,
            0.0
        ]
        L: ℝ[3] = zero_1d_array(3)
        for i:ℕ(m-1):
            idx = m - 2 - i
            V = V_array[idx]
            mg = mg_array[idx]
            hg = hg_array[idx]
            ng = ng_array[idx]
            state = [V, mg, hg, ng]
            J_state = grad(rk4_step(state, θ), state)
            J_theta = grad(rk4_step(state, θ), θ)
            L += s @ J_theta
            residual = [(V_array[idx] - true_V[idx]) / m, 0.0, 0.0, 0.0]
            s = residual + (s @ J_state)
        return L

    θ: ℝ[3] = [100.0, 30.0, 0.5]
    learning_rate: ℝ = 0.2
    beta1: ℝ = 0.9
    beta2: ℝ = 0.999
    eps_adam: ℝ = 0.00000001
    m_adam: ℝ[3] = [0.0, 0.0, 0.0]
    v_adam: ℝ[3] = [0.0, 0.0, 0.0]
    t_adam: ℝ = 0.0
    epochs: ℕ = 1

    for i:ℕ(epochs):
        g = adjoint_grad(θ)
        t_adam = t_adam + 1.0
        m_adam = beta1 * m_adam + (1.0 - beta1) * g
        v_adam = beta2 * v_adam + (1.0 - beta2) * (g * g)
        mhat = m_adam / (1.0 - beta1 ** t_adam)
        vhat = v_adam / (1.0 - beta2 ** t_adam)
        θ = θ - learning_rate * mhat / (sqrt(vhat) + eps_adam)

    pred_results = solver(θ)


References
----------

- A. L. Hodgkin and A. F. Huxley, *A quantitative description of membrane current and its application to conduction and excitation in nerve*, J. Physiol. 117, 500–544 (1952).
- `Hodgkin–Huxley model — Wikipedia <https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model>`_

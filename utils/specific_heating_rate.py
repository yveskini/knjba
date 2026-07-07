"""
Rosswog & Korobkin 2022 nuclear heating-rate fit, JAX/GPU-ready, standalone.
arXiv:2208.14026, Eq. 2 and Tables 1-2.

Unit / base conventions (verified against Table 1, not assumed):
  - E0 grid stores log10(eps0 / 1e18)         -> eps0 = 10**(E0 + 18)
  - C1, C2, C3 grids store NATURAL log(Ci)    -> Ci   = exp(Ci_grid)
    (E0 uses base 10, C1/C2/C3 use base e - this asymmetry is real, not
    a transcription bug: 10**(E0+18) at Ye=0.05,v=0.05c gives exactly the
    tabulated eps0x1e18=10.0, while 10**C1 would overshoot the plotted
    heating curve by ~8 orders of magnitude at t~tau1; exp(C1) does not.)
  - tau1 tabulated in units of 1e3 s, tau2 & tau3 in units of 1e5 s.
  - alpha, t0, sigma, alpha1, t1, sigma1 are dimensionless / seconds, as-is.

This module returns the RAW ("naked") specific heating rate dε/dt
[erg/g/s], no thermalisation efficiency applied - multiply by your own
e_th(t, mej, vej) downstream if needed.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def _grid(flat):
    return np.asarray(flat, dtype=np.float64).reshape(len(V_GRID_RAW), len(YE_GRID_RAW), order='F')


_EPS = 1e-30  # keeps x**alpha gradients finite at x -> 0 (needed for jax.grad)
YE_GRID_RAW = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
V_GRID_RAW = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


E0_GRID = _grid([1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                 1.000, 1.000, 1.041, 1.041, 1.041, 1.041,
                 1.146, 1.000, 1.041, 1.041, 1.041, 1.041,
                 1.146, 1.000, 1.000, 1.000, 1.041, 1.041,
                 1.301, 1.398, 1.602, 1.580, 1.763, 1.845,
                 0.785, 1.255, 1.673, 1.673, 1.874, 1.874,
                 0.863, 0.845, 1.212, 1.365, 1.635, 2.176,
                 -2.495, -2.495, -2.097, -2.155, -2.046, -1.824,
                 -0.699, -0.699, -0.222, 0.176, 0.176, 0.176,
                 -0.398, 0.000, 0.301, 0.477, 0.477, 0.477,])

ALP_GRID = _grid([1.37, 1.38, 1.41, 1.41, 1.41, 1.41,
                1.41, 1.38, 1.37, 1.37, 1.37, 1.37,
                1.41, 1.38, 1.37, 1.37, 1.37, 1.37,
                1.36, 1.25, 1.32, 1.32, 1.34, 1.34,
                1.44, 1.40, 1.46, 1.66, 1.60, 1.60,
                1.36, 1.33, 1.33, 1.33, 1.374, 1.374,
                1.40, 1.358, 1.384, 1.384, 1.384, 1.344,
                1.80, 1.80, 2.10, 2.10, 1.90, 1.90,
                8.00, 8.00, 7.00, 7.00, 7.00, 7.00,
                1.40, 1.40, 1.40, 1.60, 1.60, 1.60])

T0_GRID = _grid([1.80, 1.40, 1.20, 1.20, 1.20, 1.20,
                1.40, 1.00, 0.85, 0.85, 0.85, 0.85,
                1.00, 0.80, 0.65, 0.65, 0.61, 0.61,
                0.85, 0.60, 0.45, 0.45, 0.45, 0.45,
                0.65, 0.38, 0.22, 0.18, 0.12, 0.095,
                0.540, 0.31, 0.18, 0.13, 0.095, 0.081,
                0.385, 0.235, 0.1, 0.06, 0.035, 0.025,
                26.0, 26.0, 0.4, 0.4, 0.12, -20.0,
                0.20, 0.12, 0.05, 0.03, 0.025, 0.021,
                0.16, 0.08, 0.04, 0.02, 0.018, 0.016,])

SIG_GRID = _grid([0.08, 0.08, 0.095, 0.095, 0.095, 0.095,
                0.10, 0.08, 0.070, 0.070, 0.070, 0.070,
                0.07, 0.08, 0.070, 0.065, 0.070, 0.070,
                0.040, 0.030, 0.05, 0.05, 0.05, 0.050,
                0.05, 0.030, 0.025, 0.045, 0.05, 0.05,
                0.11, 0.04, 0.021, 0.021, 0.017, 0.017,
                0.10, 0.094, 0.068, 0.05, 0.03, 0.01,
                45.0, 45.0, 45.0, 45.0, 25.0, 40.0,
                0.20, 0.12, 0.05, 0.03, 0.025, 0.021,
                0.03, 0.015, 0.007, 0.01, 0.009, 0.007])

ALP1_GRID = _grid([7.50, 7.50, 7.50, 7.50, 7.50, 7.50,
                9.00, 9.00, 7.50, 7.50, 7.00, 7.00,
                8.00, 8.00, 7.50, 7.50, 7.00, 7.00,
                8.00, 8.00, 7.50, 7.50, 7.00, 7.00,
                8.00, 8.00, 5.00, 7.50, 7.00, 6.50,
                4.5, 3.8, 4.0, 4.0, 4.0, 4.0,
                2.4, 3.8, 3.8, 3.21, 2.91, 3.61,
                -1.55, -1.55, -0.75, -0.75, -2.50, -5.00,
                -1.55, -1.55, -1.55, -1.55, -1.55, -1.55,
                3.00, 3.00, 3.00, 3.00, 3.00, 3.00])

T1_GRID = _grid([0.040, 0.025, 0.014, 0.010, 0.008, 0.006,
                0.040, 0.035, 0.020, 0.012, 0.010, 0.008,
                0.080, 0.040, 0.020, 0.012, 0.012, 0.009,
                0.080, 0.040, 0.030, 0.018, 0.012, 0.009,
                0.080, 0.060, 0.065, 0.028, 0.020, 0.015,
                0.14, 0.123, 0.089, 0.060, 0.045, 0.031,
                0.264, 0.1, 0.07, 0.055, 0.042, 0.033,
                1.0, 1.0, 1.0, 1.0, 0.02, 0.01,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.04, 0.02, 0.01, 0.002, 0.002, 0.002])

SIG1_GRID = _grid([0.250, 0.120, 0.045, 0.028, 0.020, 0.015,
                0.250, 0.060, 0.035, 0.020, 0.016, 0.012,
                0.170, 0.090, 0.035, 0.020, 0.012, 0.009,
                0.170, 0.070, 0.035, 0.015, 0.012, 0.009,
                0.170, 0.070, 0.050, 0.025, 0.020, 0.020,
                0.065, 0.067, 0.053, 0.032, 0.032, 0.024,
                0.075, 0.044, 0.03, 0.02, 0.02, 0.014,
                10.0, 10.0, 10.0, 10.0, 0.02, 0.01,
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                0.01, 0.005, 0.002, 1e-4, 1e-4, 1e-4])

C1_GRID = _grid([27.2, 27.8, 28.2, 28.2, 28.2, 28.2,
                28.0, 27.8, 27.8, 27.8, 27.8, 27.8,
                27.5, 27.0, 27.8, 27.8, 27.8, 27.8,
                28.8, 28.1, 27.8, 27.8, 27.5, 27.5,
                28.5, 28.0, 27.5, 28.5, 29.2, 29.0,
                25.0, 27.5, 25.8, 20.9, 29.3, 1.0,
                28.7, 27.0, 28.0, 28.0, 27.4, 25.3,
                28.5, 29.1, 29.5, 30.1, 30.4, 29.9,
                20.4, 20.6, 20.8, 20.9, 20.9, 21.0,
                29.9, 30.1, 30.1, 30.2, 30.3, 30.3])

TAU1_GRID = _grid([4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                4.07, 4.07, 4.07, 4.07, 4.07, 4.07,
                4.77, 4.77, 4.77, 4.77, 4.07, 4.07,
                4.77, 4.77, 28.2, 1.03, 0.613, 1.0,
                3.4, 14.5, 11.4, 14.3, 13.3, 13.3,
                2.52, 2.52, 2.52, 2.52, 2.52, 2.52,
                1.02, 1.02, 1.02, 1.02, 1.02, 1.02,
                0.22, 0.22, 0.22, 0.22, 0.22, 0.22])

C2_GRID = _grid([21.5, 21.5, 22.1, 22.1, 22.1, 22.1,
                22.3, 21.5, 21.5, 21.8, 21.8, 21.8,
                22.0, 21.5, 21.5, 22.0, 21.8, 21.8,
                23.5, 22.5, 22.1, 22.0, 22.2, 22.2,
                22.0, 22.8, 23.0, 23.0, 23.5, 23.5,
                10.0, 0.0, 0.0, 19.8, 22.0, 21.0,
                26.2, 14.1, 18.8, 19.1, 23.8, 19.2,
                25.4, 25.4, 25.8, 26.0, 26.0, 25.8,
                18.4, 18.4, 18.6, 18.6, 18.6, 18.6,
                27.8, 28.0, 28.2, 28.2, 28.3, 28.3])

TAU2_GRID = _grid([4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                4.62, 4.62, 4.62, 4.62, 4.62, 4.62,
                5.62, 5.62, 5.62, 5.62, 4.62, 4.62,
                5.62, 5.18, 5.18, 34.7, 8.38, 22.6,
                0.15, 4.49, 95.0, 95.0, 0.95, 146.,
                0.12, 0.12, 0.12, 0.12, 0.12, 0.14,
                0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
                0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

C3_GRID = _grid([19.4, 19.8, 20.1, 20.1, 20.1, 20.1,
                20.0, 19.8, 19.8, 19.8, 19.8, 19.8,
                19.9, 19.8, 19.8, 19.8, 19.8, 19.8,
                5.9, 9.8, 23.5, 23.5, 23.5, 23.5,
                27.3, 26.9, 26.6, 27.4, 25.8, 25.8,
                27.8, 26.9, 18.9, 25.4, 24.8, 25.8,
                22.8, 17.9, 18.9, 25.4, 24.8, 25.5,
                20.6, 20.2, 19.8, 19.2, 19.5, 18.4,
                12.6, 13.1, 14.1, 14.5, 14.5, 14.5,
                24.3, 24.2, 24.0, 24.0, 24.0, 23.9])

TAU3_GRID = _grid([18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
                18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
                18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
                18.2, 18.2, 0.62, 0.62, 0.62, 0.62,
                0.18, 0.18, 0.18, 0.18, 0.32, 0.32,
                0.12, 0.18, 50.8, 0.18, 0.32, 0.32,
                2.4, 51.8, 50.8, 0.18, 0.32, 0.32,
                3.0, 2.5, 2.4, 2.4, 2.4, 60.4,
                200., 200., 200., 200., 200., 200.,
                8.76, 8.76, 8.76, 8.76, 8.76, 8.76])

# ----------------------------------------------------------------------
# Assemble into a single (n_Ye, n_v, 13) JAX array, coefficient order:
# [E0, alpha, t0, sigma, alpha1, t1, sigma1, C1, tau1, C2, tau2, C3, tau3]
# tau1/tau2/tau3 unit scale factors (1e3 s, 1e5 s, 1e5 s) are baked in
# here so the interpolator/heating_rate function never has to remember them.
# ----------------------------------------------------------------------
_stack = np.stack([E0_GRID, ALP_GRID, T0_GRID, SIG_GRID, ALP1_GRID, T1_GRID, SIG1_GRID,
                   C1_GRID, TAU1_GRID * 1e3, C2_GRID, TAU2_GRID * 1e5, C3_GRID, TAU3_GRID * 1e5,], axis=-1) # (n_v, n_ye, 13)


_stack = np.transpose(_stack, (1, 0, 2))        # (n_ye, n_v, 13)

YE_GRID = jnp.asarray(YE_GRID_RAW)
V_GRID = jnp.asarray(V_GRID_RAW)
COEFF_GRID = jnp.asarray(_stack)


# ----------------------------------------------------------------------
# Interpolation + heating rate
# ----------------------------------------------------------------------
def _interp_coeffs_scalar(Ye, v_ej, Ye_grid=YE_GRID, v_grid=V_GRID, coeffs=COEFF_GRID):
    """Bilinear interpolation of the 13 raw fit coefficients for scalar
    Ye, v_ej. Clamped (constant) extrapolation outside the grid.
    Returns shape (13,) in the order documented above the COEFF_GRID assembly.
    """
    i = jnp.clip(jnp.searchsorted(Ye_grid, Ye, side="right") - 1, 0, Ye_grid.size - 2)
    j = jnp.clip(jnp.searchsorted(v_grid, v_ej, side="right") - 1, 0, v_grid.size - 2)

    Ye0, Ye1 = Ye_grid[i], Ye_grid[i + 1]
    v0, v1 = v_grid[j], v_grid[j + 1]
    tY = jnp.clip((Ye - Ye0) / (Ye1 - Ye0), 0.0, 1.0)
    tv = jnp.clip((v_ej - v0) / (v1 - v0), 0.0, 1.0)

    c00, c01 = coeffs[i, j], coeffs[i, j + 1]
    c10, c11 = coeffs[i + 1, j], coeffs[i + 1, j + 1]
    c0 = c00 * (1 - tv) + c01 * tv
    c1 = c10 * (1 - tv) + c11 * tv
    return c0 * (1 - tY) + c1 * tY


@jax.jit
def heating_rate_rosswogkorobkin24(t, Ye, v_ej):
    """
    Raw ("naked") specific nuclear heating rate dε/dt [erg/g/s],
    Eq. 2 of Rosswog & Korobkin 2022, evaluated from the grid baked
    into this module (Tables 1-2). No thermalisation efficiency applied.

    t    : array_like, rest-frame time [s], any shape
    Ye   : scalar electron fraction
    v_ej : scalar ejecta velocity [c]

    Returns: array, same shape as t
    """
    (e0, alpha, t0, sigma, alpha1, t1, sigma1, c1, tau1, c2, tau2, c3, tau3) = _interp_coeffs_scalar(Ye, v_ej)

    #print(e0, alpha, t0, sigma, alpha1, t1, sigma1, c1, tau1, c2, tau2, c3, tau3)

    eps0 = 10.0 ** (e0 + 18.0)
    C1, C2, C3 = jnp.exp(c1), jnp.exp(c2), jnp.exp(c3)

    rise = jnp.clip(0.5 - jnp.arctan((t - t0) / sigma) / jnp.pi, _EPS, None)
    fall = jnp.clip(0.5 + jnp.arctan((t - t1) / sigma1) / jnp.pi, _EPS, None)

    power_law = eps0 * rise ** alpha * fall ** alpha1
    exp_terms = (C1 * jnp.exp(-t / tau1) + C2 * jnp.exp(-t / tau2) + C3 * jnp.exp(-t / tau3))
    return power_law + exp_terms


@jax.jit
def heating_rate_rosswogkorobkin24_batched(t, Ye, v_ej):
    """
    Vectorized over a batch of (Ye, v_ej) pairs sharing a common time
    grid, e.g. the blue/purple/red components of a multi-component
    kilonova, or a batch of posterior samples.

    t       : (T,)  shared rest-frame time grid [s]
    Ye, v_ej: (N,)  batch of parameters
    Returns : (N, T)
    """
    return jax.vmap(heating_rate, in_axes=(None, 0, 0))(t, Ye, v_ej)

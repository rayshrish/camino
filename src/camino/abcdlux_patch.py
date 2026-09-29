# abcdlux_patch.py
"""
Patched ABCD / LCT / MFT code for camino.
Authors: Louis, Hayden, Shrish.
"""

from __future__ import annotations

import jax.numpy as np
from jax import Array
import dLux.utils as dlu

# ============================================================
# ABCD matrices
# ============================================================


def abcd_surface_power(power: float | Array) -> Array:
    p = np.asarray(power).reshape(())
    M = np.eye(2, dtype=p.dtype)
    M = M.at[1, 0].set(-p)
    return M


def abcd_lens(focal_length: float | Array) -> Array:
    f = np.asarray(focal_length).reshape(())
    return abcd_surface_power(1.0 / f)


def abcd_mirror(radius: float) -> Array:
    return abcd_surface_power(2.0 / radius)


def abcd_free_space(z: float | Array) -> Array:
    z = np.asarray(z).reshape(())  # force scalar (0-d)
    M = np.eye(2, dtype=z.dtype)
    M = M.at[0, 1].set(z)
    return M


def abcd_fraunhofer(focal_length: float) -> Array:
    return np.array([[0.0, focal_length], [-1.0 / focal_length, 0.0]])


def compose_abcd(matrices: list | tuple) -> Array:
    M = np.eye(2)
    for Mi in matrices:
        M = Mi @ M
    return M


# ============================================================
# Coord specs (single-file version of coords.py)
# ============================================================


def unpack_size(N: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(N, tuple):
        Nx, Ny = N
        return int(Nx), int(Ny)
    return int(N), int(N)


def unpack_scale(d: float | tuple) -> tuple[float, float]:
    if isinstance(d, tuple):
        dx, dy = d
        return float(dx), float(dy)
    val = float(d)
    return val, val


def _is_size_like(obj) -> bool:
    if isinstance(obj, int):
        return True
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and all(isinstance(v, int) for v in obj)
    ):
        return True
    if hasattr(obj, "shape") and getattr(obj, "shape") == ():
        return True
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and all(hasattr(v, "shape") and v.shape == () for v in obj)
    ):
        return True
    return False


def _is_scale_like(obj) -> bool:
    if isinstance(obj, (int, float)):
        return True
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and all(isinstance(v, (int, float)) for v in obj)
    ):
        return True
    if hasattr(obj, "shape") and getattr(obj, "shape") == ():
        return True
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and all(hasattr(v, "shape") and v.shape == () for v in obj)
    ):
        return True
    return False


def unpack_coords(coords: Array | tuple) -> tuple[Array, Array]:
    # Avoid isinstance(coords, Array) fragility: just check "ndim"
    if hasattr(coords, "ndim"):
        if coords.ndim != 1:
            raise ValueError(
                f"unpack_coords: Array coords must be 1D, got shape={coords.shape}."
            )
        return coords, coords
    x, y = coords
    return x, y


def unpack_coord_spec(spec: Array | tuple) -> tuple[Array, Array]:
    # Case: explicit array -> symmetric x=y
    if hasattr(spec, "ndim"):
        return unpack_coords(spec)

    if isinstance(spec, tuple) and len(spec) == 2:
        a, b = spec

        # Uniform spec: (N, d) or ((Nx, Ny), (dx, dy))
        if _is_size_like(a) and _is_scale_like(b):
            Nx, Ny = unpack_size(a)
            dx, dy = unpack_scale(b)
            x = dlu.nd_coords(Nx, dx)
            y = dlu.nd_coords(Ny, dy)
            return x, y

        # Explicit (x, y)
        if not _is_size_like(a) and not _is_scale_like(b):
            return unpack_coords(spec)

    raise TypeError(f"unpack_coord_spec: unsupported spec {type(spec)} value={spec}")


# ============================================================
# Curvature helpers (single-file subset of curvature.py)
# ============================================================


def r2_coords(spec_in: Array | tuple) -> Array:
    x, y = unpack_coord_spec(spec_in)
    return (y**2)[:, None] + (x**2)[None, :]


def quad_phase(coords: Array | tuple, lam: float, curv: float | Array) -> Array:
    return np.exp(1j * np.pi * curv * r2_coords(coords) / lam)


def apply_curv(
    u: Array, coords: Array | tuple, lam: float, curv: float | Array
) -> Array:
    return u * quad_phase(coords, lam, curv)


def remove_curv(
    u: Array, coords: Array | tuple, lam: float, curv: float | Array
) -> Array:
    return u * quad_phase(coords, lam, curv).conj()


def propagate_curv(ABCD: Array, curv_in: float | Array) -> Array:
    a, b, c, d = ABCD.flatten()
    return (c + d * curv_in) / (a + b * curv_in)


def residual_abcd(ABCD: Array, curv_in: float, curv_out: float) -> Array:
    a, b, c, d = ABCD.flatten()
    a_res = a + b * curv_in
    d_res = d - b * curv_out
    c_res = (a_res * d_res - 1.0) / b
    return np.array([[a_res, b], [c_res, d_res]])


def residual_curv_cancel(ABCD: Array):
    a, b, c, d = ABCD.flatten()
    curv_in = -a / b
    curv_out = d / b
    return curv_in, curv_out


def factorise_curv(
    ABCD: Array,
    curv_in: float | Array | None,
    curv_out: float | Array | None,
    mode: str = "physical",
) -> tuple[Array, float | Array, float | Array]:
    if mode == "physical":
        if curv_in is None:
            curv_in = 0.0
        if curv_out is None:
            curv_out = propagate_curv(ABCD, curv_in)
    elif mode == "cancel":
        curv_in, curv_out = residual_curv_cancel(ABCD)
    elif mode == "manual":
        if curv_in is None or curv_out is None:
            raise ValueError("mode='manual' requires curv_in and curv_out.")
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    ABCD_res = residual_abcd(ABCD, curv_in, curv_out)
    return ABCD_res, curv_in, curv_out


# ============================================================
# MFT (single-file version of mft.py)
# ============================================================


def _mft_kernel_1d(x_in: Array, x_out: Array, alpha: float, weight: float) -> Array:
    phase = alpha * (x_out[:, None] * x_in[None, :])
    return np.exp(1j * phase) * weight


def mft(
    u: Array, Kx: Array, Ky: Array, left_conj: bool = False, right_conj: bool = False
) -> Array:
    Kx_eff = Kx.conj() if right_conj else Kx
    Ky_eff = Ky.conj() if left_conj else Ky
    tmp = u @ Kx_eff.T
    return Ky_eff @ tmp


def mft_kernels(
    spec_in: Array | tuple,
    spec_out: Array | tuple,
    alpha: float,
    weight: float | tuple = 1.0,
) -> tuple[Array, Array]:
    x_in, y_in = unpack_coord_spec(spec_in)
    x_out, y_out = unpack_coord_spec(spec_out)

    if isinstance(weight, (int, float)):
        wx = wy = float(weight)
    else:
        wx, wy = weight

    Kx = _mft_kernel_1d(x_in, x_out, alpha, wx)
    Ky = _mft_kernel_1d(y_in, y_out, alpha, wy)
    return Kx, Ky


# ===================================================================
# LCT / Collins integral (single-file version of Hayden's lct.py core)
# ===================================================================


import jax.numpy as jnp


def lct_sampling_quick(x_in, x_out, lam, ABCD, eps=1e-30):
    """
    Quick sampling diagnostics for Collins LCT on separable grids.

    Returns Nyquist ratios (>=1 is safe-ish):
      - p_kernel: sampling of exp(-i 2π x x' /(λ b)) kernel
      - p_pre:    sampling of input chirp exp(i π a x^2 /(λ b))
      - p_post:   sampling of output chirp exp(i π d x'^2 /(λ b))
    """
    a, b, c, d = [ABCD.reshape(-1)[i] for i in range(4)]

    dx_in = x_in[1] - x_in[0]
    dx_out = x_out[1] - x_out[0]
    X_in = 0.5 * (x_in[-1] - x_in[0])
    X_out = 0.5 * (x_out[-1] - x_out[0])

    # Kernel phase: exp(-i 2π x x' /(λ b))
    # worst phase slope in x is at max |x'|
    dphi_in_max = (2 * jnp.pi / (lam * jnp.abs(b) + eps)) * X_out * dx_in
    dphi_out_max = (2 * jnp.pi / (lam * jnp.abs(b) + eps)) * X_in * dx_out
    p_kernel = jnp.pi / (jnp.maximum(dphi_in_max, dphi_out_max) + eps)

    # Pre/post chirps: phase ~ π a x^2 /(λ b), slope ~ 2π a x /(λ b)
    dphi_pre_max = (2 * jnp.pi * jnp.abs(a) / (lam * jnp.abs(b) + eps)) * X_in * dx_in
    dphi_post_max = (
        (2 * jnp.pi * jnp.abs(d) / (lam * jnp.abs(b) + eps)) * X_out * dx_out
    )
    p_pre = jnp.pi / (dphi_pre_max + eps)
    p_post = jnp.pi / (dphi_post_max + eps)

    return {"p_kernel": p_kernel, "p_pre": p_pre, "p_post": p_post}


def lct_kernels(
    spec_in: Array | tuple, spec_out: Array | tuple, lam: float, ABCD: Array
) -> tuple:
    x_in, y_in = unpack_coord_spec(spec_in)
    x_out, y_out = unpack_coord_spec(spec_out)

    a, b, c, d = ABCD.flatten()

    r2_in = r2_coords((x_in, y_in))
    r2_out = r2_coords((x_out, y_out))

    pre = np.exp(1j * np.pi * a * r2_in / (lam * b))
    post = np.exp(1j * np.pi * d * r2_out / (lam * b))

    dx_in = x_in[1] - x_in[0]
    dy_in = y_in[1] - y_in[0]
    dx_out = x_out[1] - x_out[0]
    dy_out = y_out[1] - y_out[0]

    alpha = -2.0 * np.pi / (lam * b)

    Kx, Ky = mft_kernels(
        spec_in=spec_in, spec_out=spec_out, alpha=alpha, weight=(dx_in, dy_in)
    )

    pref = 1.0 / (1j * lam * b)
    scale = np.sqrt((dx_out * dy_out) / (dx_in * dy_in))
    return pre, Kx, Ky, post, pref, scale


def lct_kernel_prop(
    u_in: Array,
    pre: Array,
    Kx: Array,
    Ky: Array,
    post: Array,
    pref: complex,
    scale: float,
) -> Array:
    u_tmp = pre * u_in
    u_mft = mft(u_tmp, Kx, Ky)
    return pref * scale * u_mft * post


def lct_prop_basic(
    u_in: Array,
    spec_in: Array | tuple,
    spec_out: Array | tuple,
    lam: float,
    ABCD: Array,
    output_phase: bool = True,
) -> Array:
    pre, Kx, Ky, post, pref, scale = lct_kernels(spec_in, spec_out, lam, ABCD)
    if not output_phase:
        # The output chirp is a pure phase; skip it when only |u|^2 is needed.
        post = 1.0
    return lct_kernel_prop(u_in, pre, Kx, Ky, post, pref, scale)


def lct_prop(
    u_in: Array,
    spec_in: Array | tuple,
    spec_out: Array | tuple,
    lam: float,
    ABCD: Array,
    curv_in: float | None = None,
    curv_out: float | None = None,
    mode: str = "physical",
    strip_input: bool = True,
    return_residual: bool = False,
    output_phase: bool = True,
) -> Array:
    """Collins/LCT propagation. output_phase=False omits every output-plane
    phase factor (residual chirp and curv_out); the field is then only
    correct in modulus, which is all a PSF needs."""
    ABCD_res, curv_in, curv_out = factorise_curv(ABCD, curv_in, curv_out, mode)

    u_res_in = remove_curv(u_in, spec_in, lam, curv_in)

    u_res_out = lct_prop_basic(
        u_res_in, spec_in, spec_out, lam, ABCD_res, output_phase=output_phase
    )

    if return_residual or not output_phase:
        return u_res_out

    return apply_curv(u_res_out, spec_out, lam, curv_out)


# ============================================================
# A helper for model class (ABCD version)
# ============================================================


def propagate_mono_abcd(self, wavelength, offset=np.zeros(2), return_wf=False):
    import dLux as dl  # keep local

    wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength).tilt(offset)

    for layer in list(self.layers.values()):
        wf *= layer

    u_in = wf.phasor

    fl = self.fnumber * self.diameter  # meters

    # If your existing code uses self.defocus in nm, convert:
    z_defocus = self.defocus * 1e-9

    abcd = compose_abcd([abcd_lens(fl), abcd_free_space(fl + z_defocus)])

    # Input sampling (pupil plane)
    N_in = self.wf_npixels
    dx_in = self.diameter / self.wf_npixels
    x_in = dlu.nd_coords(N_in, dx_in)

    # Output sampling: match your existing psf_pixel_scale (angular) via x = f * theta
    true_pixel_scale = self.psf_pixel_scale / self.oversample  # arcsec/pix
    theta_pix = dlu.arcsec2rad(true_pixel_scale)  # rad/pix

    N_out = self.psf_npixels * self.oversample
    dx_out = fl * theta_pix  # meters/pix at focal plane

    x_out = dlu.nd_coords(N_out, dx_out)

    u_out = lct_prop_basic(u_in, x_in, x_out, wavelength, abcd)

    wf_out = dl.Wavefront(N_out, N_out * dx_out, wavelength).set(
        ["amplitude", "phase"], [np.abs(u_out), np.angle(u_out)]
    )

    if return_wf:
        return wf_out
    return wf_out.psf

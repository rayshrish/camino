# camino: Computational Aberration Modelling and Inference for NIRCam Observations
from __future__ import annotations

print("Loading camino module...")


### Basic imports
###############################################################################
plot_flag = 0

from jax import Array

import jax

jax.config.update("jax_enable_x64", True)
import astropy.io.fits as fits

from importlib.resources import files

import dLux as dl

if plot_flag == 1:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps, colors

from jax.flatten_util import ravel_pytree
from importlib.resources import files

import pandas as pd

# matplotlib inline
if plot_flag == 1:
    plt.rcParams["image.cmap"] = "inferno"
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["image.origin"] = "lower"
    plt.rcParams["figure.dpi"] = 120
    inferno = colormaps["inferno"]
    seismic = colormaps["seismic"]
    inferno.set_bad("k", 0.5)
    seismic.set_bad("k", 0.5)

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

import dLux.utils as dlu

from scipy.ndimage import gaussian_filter, shift as ndi_shift, center_of_mass

import re
import os
import time
import glob
import numpy as onp
from astropy.io import fits

from functools import partial

###############################################################################


def apply_shear(psf, shx, shy=0.0):
    """
    Apply a small shear to a 2D PSF image in pixel coordinates.

    shx, shy are small dimensionless coefficients:
        x' = x + shx * y
        y' = y + shy * x
    We will only actually use shx for now (1D shear).
    """
    H, W = psf.shape
    y, x = jnp.indices((H, W))

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    x0 = x - cx
    y0 = y - cy

    x_in = x0 + shx * y0 + cx
    y_in = y0 + shy * x0 + cy

    coords = jnp.stack([y_in, x_in])  # (2, H, W)
    psf_warped = map_coordinates(psf, coords, order=1, mode="nearest")
    return psf_warped


def apply_pupil_shear(img, shx, shy=0.0, order=1, cval=0.0):
    """
    Pupil-plane shear implemented as a coordinate warp of a pupil-plane array.

    x' = x + shx*y
    y' = y + shy*x

    order=1 (bilinear) is usually nicer for gradients than order=0 (nearest).
    For a strictly binary pupil you *can* use order=0, but gradients may be ugly.
    """
    img = jnp.asarray(img)
    H, W = img.shape
    y, x = jnp.indices((H, W))

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    x0 = x - cx
    y0 = y - cy

    # input coords to sample from
    x_in = x0 + shx * y0 + cx
    y_in = y0 + shy * x0 + cy

    coords = jnp.stack([y_in, x_in])  # (2, H, W)
    return map_coordinates(img, coords, order=order, mode="constant", cval=cval)


def eval_poly_log10(x, coeffs):
    """
    Shear helper function

    Evaluate p(x) = c0 + c1 x + c2 x^2 + ... (log10-space polynomial)
    x: (...,) array
    coeffs: (n_coeffs,)
    Returns p(x) with same shape as x.
    """
    # Horner's rule, from highest to lowest power
    w = 0.0
    for c in coeffs[::-1]:
        w = w * x + c
    return w  # this is log10 intensity


def radial_zoom(psf: jnp.ndarray, scale: float) -> jnp.ndarray:
    """
    Radially rescale a 2D PSF array by a factor `scale` around its centre.
    scale > 1 → rings move outward, scale < 1 → rings move inward.
    """
    h, w = psf.shape
    yc = (h - 1) / 2.0
    xc = (w - 1) / 2.0

    # coordinates relative to centre
    y = jnp.arange(h) - yc
    x = jnp.arange(w) - xc
    yy, xx = jnp.meshgrid(y, x, indexing="ij")

    # map output coords back to input coords
    yy_in = yy / scale + yc
    xx_in = xx / scale + xc

    coords = jnp.stack([yy_in, xx_in], axis=0)  # shape (2, H, W)

    # bilinear interpolation
    psf_scaled = map_coordinates(psf, coords, order=1, mode="constant", cval=0.0)
    return psf_scaled


def apply_pupil_curvature(wf, R_m):
    """
    Apply spherical curvature as a quadratic phase at the pupil.

    Parameters
    ----------
    wf : dLux.Wavefront
    R_m : float
        Radius of curvature in meters. Sign matters.
        Use large |R| for almost-flat (e.g. 1e9).
    """
    coords = dlu.pixel_coords(wf.npixels, wf.diameter)  # (2, N, N) in meters
    r2 = (coords**2).sum(0)  # m^2

    # phase = (pi / (lambda * R)) * r^2   [radians]
    phase = jnp.pi * r2 / (wf.wavelength * R_m)

    return wf.set("phase", wf.phase + phase)


def tv_norm(x):
    """
    Pupil Delta Helper Function 1
    """
    # anisotropic TV
    dx = x[1:, :] - x[:-1, :]
    dy = x[:, 1:] - x[:, :-1]
    return jnp.sum(jnp.abs(dx)) + jnp.sum(jnp.abs(dy))


def l2_smooth(x):
    """
    Pupil Delta Helper Function 2
    """
    # Laplacian-like smoothness (optional alternative)
    dx = x[1:, :] - x[:-1, :]
    dy = x[:, 1:] - x[:, :-1]
    return jnp.sum(dx**2) + jnp.sum(dy**2)


def _gaussian_otf(ny, nx, sigma_pix):
    """
    Returns the Optical Transfer Function (OTF) of a 2D Gaussian blur
    with std dev sigma_pix (in pixels), for FFT convolution.
    """
    fy = jnp.fft.fftfreq(ny)[:, None]  # cycles/pix
    fx = jnp.fft.fftfreq(nx)[None, :]
    # Gaussian MTF: exp(-2*pi^2*sigma^2*(fx^2+fy^2))
    return jnp.exp(-2.0 * (jnp.pi**2) * (sigma_pix**2) * (fx * fx + fy * fy))


def gaussian_blur_fft(img, sigma_pix):
    """
    Applies the Guassian blur on the OPD map
    """
    ny, nx = img.shape
    otf = _gaussian_otf(ny, nx, sigma_pix)
    F = jnp.fft.fft2(img)
    out = jnp.fft.ifft2(F * otf).real
    return out


def _opd_obsid_to_jw_stem(obs_id: str) -> str:
    # Works for any number of zeros after 'P'
    m = re.match(r"^V(\d{5})(\d{6})P0+(\d{5})$", obs_id.strip())
    if not m:
        raise ValueError(f"Can't parse OPD OBS_ID: {obs_id}")
    prop = m.group(1)  # 07464
    visit = m.group(2)  # 177001
    act = m.group(3)  # 03104
    return f"{prop}{visit}_{act}"


def _get_pupil_keyword(cal_fits_path: str):
    """
    Return PUPIL keyword from a CAL FITS file (best-effort).
    Usually in primary header, but we also check ext 1 defensively.
    """
    for ext in (0, 1):
        try:
            hdr = fits.getheader(cal_fits_path, ext)
            pup = hdr.get("PUPIL")
            if pup is not None:
                pup = str(pup).strip()
                if pup != "":
                    return pup
        except Exception:
            pass
    return None


def _stage(msg: str, tprev: float | None):
    """
    Print stage header + timing since previous stage.
    Returns new timestamp.
    """
    now = time.perf_counter()
    if tprev is None:
        print(f"\n== {msg} ==")
    else:
        print(f"\n== {msg} ==  (+{now - tprev:.2f}s)")
    return now


import numpy as onp
from scipy.ndimage import gaussian_filter, center_of_mass


def extract_cutout(img, center, size, fill_value=None):
    """
    Extract a square cutout of given size centered on (y, x).
    Pads with fill_value if the box goes off the image edge.

    Returns
    -------
    cut : 2D array
        The extracted cutout of shape (size, size)
    origin : tuple
        (y1, x1) of the requested box in the parent image coordinates
        before clipping
    """
    img = onp.asarray(img)
    y0, x0 = center
    y0i, x0i = int(round(y0)), int(round(x0))
    half = size // 2

    y1, y2 = y0i - half, y0i + half
    x1, x2 = x0i - half, x0i + half

    if fill_value is None:
        fill_value = onp.nanmedian(img)

    pad_y1 = max(0, -y1)
    pad_x1 = max(0, -x1)
    pad_y2 = max(0, y2 - img.shape[0])
    pad_x2 = max(0, x2 - img.shape[1])

    y1c = max(0, y1)
    y2c = min(img.shape[0], y2)
    x1c = max(0, x1)
    x2c = min(img.shape[1], x2)

    cut = img[y1c:y2c, x1c:x2c]

    if any(p > 0 for p in [pad_y1, pad_y2, pad_x1, pad_x2]):
        cut = onp.pad(
            cut,
            ((pad_y1, pad_y2), (pad_x1, pad_x2)),
            mode="constant",
            constant_values=fill_value,
        )

    return cut, (y1, x1)


def estimate_center(img, smooth_sigma=2.0, thresh_sigma=5.0):
    """
    Estimate a robust flux-weighted center in a 2D image.
    Returns (y, x).
    """
    img = onp.asarray(img)

    med = onp.nanmedian(img)
    mad = onp.nanmedian(onp.abs(img - med)) + 1e-12
    sigma = 1.4826 * mad

    sm = gaussian_filter(onp.nan_to_num(img - med, nan=0.0), smooth_sigma)

    mask = sm > (thresh_sigma * sigma)
    if mask.sum() < 50:
        mask = sm > (3.0 * sigma)

    if mask.sum() < 10:
        y0, x0 = onp.unravel_index(onp.argmax(sm), sm.shape)
    else:
        weights = onp.where(mask, sm, 0.0)
        y0, x0 = center_of_mass(weights)

    return y0, x0


def cutout_around_defocused_psf_multi(
    img,
    size=128,
    recenter_sizes=(1024, 512, 256),
    smooth_sigma=2.0,
    thresh_sigma=5.0,
    return_history=False,
):
    """
    Multiscale centering for a defocused PSF, followed by one final extraction.

    Parameters
    ----------
    img : 2D array
        Input image.
    size : int
        Final cutout size.
    recenter_sizes : tuple of int
        Sizes used for iterative recentering. The final `size` is extraction only.
    smooth_sigma : float
        Gaussian smoothing sigma for robust centroiding.
    thresh_sigma : float
        Threshold in units of robust sigma above background.
    return_history : bool
        If True, also return the centering history.

    Returns
    -------
    cut : 2D array
        Final cutout of shape (size, size)
    center : tuple
        (y, x) center in full-image coordinates
    history : list, optional
        Returned only if return_history=True
    """
    img = onp.asarray(img)

    # Initial estimate on full frame
    y, x = estimate_center(img, smooth_sigma=smooth_sigma, thresh_sigma=thresh_sigma)
    history = [("full", img.shape, (y, x))]

    # Recenter only while shrinking
    for rec_size in recenter_sizes:
        cut, (y1, x1) = extract_cutout(img, (y, x), rec_size)
        yc, xc = estimate_center(
            cut, smooth_sigma=smooth_sigma, thresh_sigma=thresh_sigma
        )
        y, x = y1 + yc, x1 + xc
        history.append((rec_size, cut.shape, (y, x)))

    # Final extraction only
    final_cut, _ = extract_cutout(img, (y, x), size)

    if return_history:
        return final_cut, (y, x), history
    return final_cut, (y, x)


def cutout_around_defocused_psf(img, size=128, smooth_sigma=2.0, thresh_sigma=5.0):
    """
    Find donut-ish PSF center and return a size x size cutout centered on it.

    img: 2D numpy array
    size: cutout size (e.g. 128)
    smooth_sigma: Gaussian smoothing for robust centroiding
    thresh_sigma: threshold relative to background MAD (robust)
    """
    img = onp.asarray(img)

    # Robust background estimate (median + MAD)
    med = onp.nanmedian(img)
    mad = onp.nanmedian(onp.abs(img - med)) + 1e-12
    sigma = 1.4826 * mad

    # Smooth to suppress speckles/hot pixels
    sm = gaussian_filter(onp.nan_to_num(img - med, nan=0.0), smooth_sigma)

    # Mask: keep only significant flux
    mask = sm > (thresh_sigma * sigma)

    # If threshold is too strict, relax it
    if mask.sum() < 50:
        mask = sm > (3.0 * sigma)
    if mask.sum() < 10:
        # Fallback: just take brightest pixel in smoothed image
        y0, x0 = onp.unravel_index(onp.argmax(sm), sm.shape)
    else:
        # Weighted centroid on masked region
        weights = onp.where(mask, sm, 0.0)
        y0, x0 = center_of_mass(weights)

    # Integer center for cutout indexing
    y0i, x0i = int(round(y0)), int(round(x0))

    half = size // 2
    y1, y2 = y0i - half, y0i + half
    x1, x2 = x0i - half, x0i + half

    # Pad if near edges
    pad_y1 = max(0, -y1)
    pad_x1 = max(0, -x1)
    pad_y2 = max(0, y2 - img.shape[0])
    pad_x2 = max(0, x2 - img.shape[1])

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(img.shape[0], y2)
    x2 = min(img.shape[1], x2)

    cut = img[y1:y2, x1:x2]
    if any(p > 0 for p in [pad_y1, pad_y2, pad_x1, pad_x2]):
        cut = onp.pad(
            cut,
            ((pad_y1, pad_y2), (pad_x1, pad_x2)),
            mode="constant",
            constant_values=med,
        )

    return cut, (y0, x0)


def estimate_best_isolated_center(
    img,
    smooth_sigma=3.0,
    thresh_sigma=5.0,
    aperture_radius=40,
    annulus_inner=70,
    annulus_outer=140,
    min_edge_dist=150,
    min_neighbor_dist=250,
    min_area=20,
    max_candidates=100,
    return_table=False,
):
    """
    First-pass source selector:
    pick a bright, isolated, non-edge source rather than simply the brightest source.

    Returns
    -------
    (y, x) or ((y, x), table)
    """
    img = np.asarray(img)

    med, sig = robust_bg_sigma(img)
    work = img - med
    smooth = gaussian_filter(work, smooth_sigma)

    threshold = thresh_sigma * sig
    mask = np.isfinite(smooth) & (smooth > threshold)

    lab, nlab = label(mask)

    candidates = []

    ny, nx = img.shape

    for lab_id in range(1, nlab + 1):
        pix = lab == lab_id
        area = np.sum(pix)

        if area < min_area:
            continue

        # centroid on smoothed thresholded region
        yc, xc = center_of_mass(smooth, labels=lab, index=lab_id)

        if not np.isfinite(yc) or not np.isfinite(xc):
            continue

        # edge rejection
        dist_edge = min(yc, xc, ny - 1 - yc, nx - 1 - xc)
        if dist_edge < min_edge_dist:
            continue

        inner = aperture_sum(work, yc, xc, aperture_radius)
        ann = annulus_sum(work, yc, xc, annulus_inner, annulus_outer)

        peak = np.nanmax(img[pix])

        # avoid negative weird annulus values causing nonsense
        ann_pos = max(ann, 1e-12)

        isolation = inner / ann_pos
        snr_like = inner / (sig * np.sqrt(np.pi * aperture_radius**2))

        # simple first-pass score
        # high inner flux is good, high annulus flux is bad
        score = snr_like

        candidates.append(
            {
                "y": yc,
                "x": xc,
                "area": area,
                "peak": peak,
                "inner_flux": inner,
                "annulus_flux": ann,
                "isolation": isolation,
                "snr_like": snr_like,
                "score": score,
                "dist_edge": dist_edge,
            }
        )

    coords = np.array([[c["y"], c["x"]] for c in candidates])

    for i, c in enumerate(candidates):
        d = np.sqrt((coords[:, 0] - c["y"]) ** 2 + (coords[:, 1] - c["x"]) ** 2)
        d[i] = np.inf
        c["nearest_neighbor_dist"] = np.min(d)

    min_neighbor_dist = 250  # pixels; try 200–300

    candidates = [
        c for c in candidates if c["nearest_neighbor_dist"] > min_neighbor_dist
    ]

    if len(candidates) == 0:
        raise RuntimeError("No valid isolated source candidates found.")

    # keep only strongest candidates before sorting, mostly for debugging sanity
    candidates = sorted(candidates, key=lambda c: c["snr_like"], reverse=True)
    best = candidates[0]

    if return_table:
        return (best["y"], best["x"]), candidates

    return best["y"], best["x"]


def cutout_around_defocused_psf_multi_isolated(
    img,
    size=128,
    recenter_sizes=(1024, 512, 256),
    smooth_sigma=2.0,
    thresh_sigma=5.0,
    use_isolated_initial=True,
    min_neighbor_dist=250,
    min_edge_dist=150,
    return_history=False,
):
    """
    Multiscale centering for a defocused PSF, followed by one final extraction.

    Initial full-frame selection can use an isolated-source selector to avoid
    choosing crowded bright PSFs.
    """
    img = onp.asarray(img)

    # Initial estimate on full frame
    if use_isolated_initial:
        y, x = estimate_best_isolated_center(
            img,
            smooth_sigma=smooth_sigma,
            thresh_sigma=thresh_sigma,
            min_neighbor_dist=min_neighbor_dist,
            min_edge_dist=min_edge_dist,
        )
        history = [("full_best_isolated", img.shape, (y, x))]
    else:
        y, x = estimate_center(
            img,
            smooth_sigma=smooth_sigma,
            thresh_sigma=thresh_sigma,
        )
        history = [("full_brightest", img.shape, (y, x))]

    # Recenter only while shrinking
    for rec_size in recenter_sizes:
        cut, (y1, x1) = extract_cutout(img, (y, x), rec_size)

        yc, xc = estimate_center(
            cut,
            smooth_sigma=smooth_sigma,
            thresh_sigma=thresh_sigma,
        )

        y, x = y1 + yc, x1 + xc
        history.append((rec_size, cut.shape, (y, x)))

    # Final extraction only
    final_cut, _ = extract_cutout(img, (y, x), size)

    if return_history:
        return final_cut, (y, x), history

    return final_cut, (y, x)


def _find_existing_cals(download_dir: str, want_prefix: str, det_tag: str | None):
    """
    Search download_dir recursively for already-present *_cal.fits matching
    the visit prefix (want_prefix), and optionally detector token.
    Returns list of file paths.
    """
    # Search recursively; astroquery writes into mastDownload/... so this is safest.
    pattern = os.path.join(
        os.path.abspath(download_dir), "**", f"{want_prefix}*_cal.fits"
    )
    hits = glob.glob(pattern, recursive=True)

    if det_tag:
        det_hits = [p for p in hits if f"_{det_tag}_" in os.path.basename(p)]
        if det_hits:
            hits = det_hits

    # de-dupe, keep stable order
    hits = sorted(set(hits))
    return hits


def transfer_fn_old(coords, npixels, wavelength, pscale, distance):
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)
    return _fftshift(jnp.exp(-1.0j * jnp.pi * wavelength * distance * rho_sq))


def transfer_fn_patched(coords, npixels, wavelength, pscale, distance):
    """
    Minimal fix: treat coords/scaling as angular frequency (rad/m) and convert
    to cycles/m by dividing by (2π)^2 in the quadratic phase.
    This brings 'distance' much closer to true meters.
    """
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)  # ~ rad^2 / m^2 (effective)
    rho_sq_cycles = rho_sq / (2.0 * jnp.pi) ** 2  # convert to (cycles/m)^2
    return _fftshift(jnp.exp(-1.0j * jnp.pi * wavelength * distance * rho_sq_cycles))


def transfer_fn(coords, npixels, wavelength, pscale, distance):
    rho_sq = (coords**2).sum(0)
    return _fftshift(jnp.exp(-1.0j * jnp.pi * wavelength * distance * rho_sq))


def transfer(wf, distance, pad=2):
    # coords = dlu.pixel_coords(pad * wf.npixels, pad * wf.diameter)
    npix = pad * wf.npixels
    diam = pad * wf.diameter
    freqs = jnp.fft.fftshift(jnp.fft.fftfreq(npix, diam / npix))
    coords = jnp.array(jnp.meshgrid(freqs, freqs))
    return transfer_fn(
        coords, wf.npixels, wf.wavelength, pad * wf.pixel_scale, distance
    )
    # return transfer_fn_patched(coords, wf.npixels, wf.wavelength, pad * wf.pixel_scale, distance)


def _fft(phasor, pad=2):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return 1 / padded.shape[0] * jnp.fft.fft2(padded)


def _ifft(phasor, pad=1):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return phasor.shape[0] * jnp.fft.ifft2(padded)


def _fftshift(phasor):
    return jnp.fft.fftshift(phasor)


def plane_to_plane(wf, distance, pad=2):
    fft_wf = _fft(wf.phasor, pad=pad)
    tf = transfer(wf, distance, pad=pad)
    phasor = dlu.resize(_ifft(fft_wf * tf), wf.npixels)
    return wf.set(["amplitude", "phase"], [jnp.abs(phasor), jnp.angle(phasor)])


def err_poisson_dn(
    expected_dn=None,
    gain_e_per_dn=1.0,
    read_noise_e=0.0,
    floor_e=1.0,
    fallback_constant=None,
    pupil_mask=None,
    data_dn=None,
):
    """
    Return per-pixel sigma in DN for Poisson + read noise.

    expected_dn: mean counts image in DN (use EXPECTED or your model PSF*flux + bg)
    gain_e_per_dn: electrons per DN
    read_noise_e: read noise RMS in electrons
    floor_e: small extra floor (e-) to prevent σ=0 in very dark regions

    If expected_dn is None, fall back to a constant sigma estimated from
    off-pupil pixels of data_dn (or 'fallback_constant' if provided).
    """
    EPS = 1e-12
    gain = float(max(gain_e_per_dn, EPS))

    if expected_dn is not None:
        expected_dn = onp.asarray(expected_dn, float)
        var_e = (
            onp.clip(expected_dn * gain, 0, None)
            + read_noise_e**2
            + float(max(floor_e, 0.0)) ** 2
        )
        sigma_dn = onp.sqrt(var_e) / gain
    else:
        if fallback_constant is not None:
            sigma_dn = onp.full_like(data_dn, float(fallback_constant), dtype=float)
        else:
            if (pupil_mask is not None) and (data_dn is not None):
                bg = onp.asarray(data_dn)[onp.asarray(pupil_mask) == 0]
                est = float(onp.nanstd(bg)) if bg.size else float(onp.nanstd(data_dn))
            else:
                est = 1.0
            sigma_dn = onp.full_like(data_dn, est, dtype=float)

    sigma_dn = onp.maximum(sigma_dn, 1e-6)
    return sigma_dn


import equinox as eqx
import zodiax as zdx

from dLux.layers.optical_layers import OpticalLayer

from dataclasses import dataclass
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


Array = jax.Array


@dataclass
class Rotate:
    """
    Dependency-light Rotate layer.

    Matches your original approach:
      - coords: centered physical coordinates over `diameter` (default 2)
      - rotate_coords: uses cos(-theta), sin(-theta) (inverse mapping / pull sampling)
      - anisotropy: scales 2nd coordinate after rotation
      - interpolation: map_coordinates (linear by default)
    """

    rotation_deg: float
    anisotropy: float = 1.0
    diameter: float = 2.0
    order: int = 1  # 1 = linear. (JAX map_coordinates commonly supports 0/1 robustly.)
    fill: float = 0.0  # constant fill outside

    def _coords_centered_physical(self, n: int) -> Array:
        """Return coords shaped (2, n, n), centered, spanning `diameter` with pixel-center sampling."""
        step = self.diameter / n
        start = -self.diameter / 2.0 + step / 2.0
        grid_1d = start + step * jnp.arange(n, dtype=jnp.float32)
        # indexing="ij": first axis varies along rows (axis0), second along cols (axis1)
        x, y = jnp.meshgrid(grid_1d, grid_1d, indexing="ij")
        return jnp.stack([x, y], axis=0)

    def _rotate_coords(self, coords: Array, angle_rad: Array) -> Array:
        """Same math as your rotate_coords: uses -angle inside sin/cos."""
        x, y = coords[0], coords[1]
        c = jnp.cos(-angle_rad)
        s = jnp.sin(-angle_rad)
        new_x = c * x + s * y
        new_y = -s * x + c * y
        return jnp.stack([new_x, new_y], axis=0)

    def _physical_to_index(self, coords: Array, n: int) -> Array:
        """
        Convert physical coords (centered, diameter-based) to index coords for map_coordinates.
        For our grid: coord = start + step * idx  -> idx = (coord - start) / step
        """
        step = self.diameter / n
        start = -self.diameter / 2.0 + step / 2.0
        return (coords - start) / step

    def apply(self, PSF_or_array: Any) -> Any:
        """
        If input has `.data` and `.set("data", ...)`, returns PSF.set(...).
        Otherwise returns rotated array.
        """
        # Accept either a PSF-like object or a raw array
        if hasattr(PSF_or_array, "data"):
            img = jnp.asarray(PSF_or_array.data)
        else:
            img = jnp.asarray(PSF_or_array)

        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            raise ValueError(f"Rotate expects a square 2D array, got shape={img.shape}")

        n = img.shape[0]
        angle = jnp.deg2rad(jnp.asarray(self.rotation_deg, dtype=jnp.float32))

        coords = self._coords_centered_physical(n)  # (2,n,n) physical
        rot_coords = self._rotate_coords(coords, angle)  # (2,n,n) physical
        sample_coords = (
            rot_coords
            * jnp.array([1.0, self.anisotropy], dtype=jnp.float32)[:, None, None]
        )

        # map_coordinates wants coordinates in index space, shape (ndim, ...)
        sample_idx = self._physical_to_index(sample_coords, n)  # (2,n,n) index coords

        rotated = map_coordinates(
            img,
            sample_idx.reshape(2, -1),
            order=self.order,
            mode="constant",
            cval=self.fill,
        ).reshape((n, n))

        if hasattr(PSF_or_array, "set") and callable(getattr(PSF_or_array, "set")):
            return PSF_or_array.set("data", rotated)
        return rotated

    # Optional: make it callable like a "layer"
    def __call__(self, x: Any) -> Any:
        return self.apply(x)


class JWSTPrimary(dl.Optic):
    """
    A class used to represent the JWST primary mirror. This is essentially a
    wrapper around the dLux.Optic class that simply enforces the normalisation
    at this plane, and is slightly more efficient than the native
    implementation.
    """

    pixelscale: float

    def __init__(
        self: OpticalLayer,
        transmission: Array = None,
        opd: Array = None,
        pixelscale: float = None,
    ):
        """
        Parameters
        ----------
        transmission: Array = None
            The Array of transmission values to be applied to the input
            wavefront.
        opd : Array, metres = None
            The Array of OPD values to be applied to the input wavefront.
        pixelscale : float, m/pix = None
            Pixel scale of the optical planes
        """
        self.pixelscale = pixelscale
        super().__init__(transmission=transmission, opd=opd, normalise=True)

    def apply(self, wavefront):
        # Apply transmission and normalise
        amplitude = wavefront.amplitude * self.transmission
        amplitude /= np.linalg.norm(amplitude)

        # Apply phase
        phase = wavefront.phase + wavefront.wavenumber * self.opd

        # Update and return
        return wavefront.set(["amplitude", "phase"], [amplitude, phase])


def arr2pix(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return pscale * (coords - shift)


def pix2arr(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return (coords / pscale) + shift


import jax
import jax.numpy as jnp

Array = jax.Array


def map_coordinates_2d(
    image: Array,
    coords: Array,  # shape (2, H, W) or (2, N)
    order: int = 1,  # 0=nearest, 1=bilinear, 3=cubic convolution
    mode: str = "constant",  # only "constant" supported here
    cval: float = 0.0,
) -> Array:
    """
    JAX-friendly 2D coordinate interpolation.

    Supports:
      - image: (H, W)
      - coords: (2, ...) in (row, col) index coordinates
      - mode="constant" only
      - order=0 : nearest-neighbour
      - order=1 : bilinear
      - order=3 : bicubic convolution (Keys kernel, a=-0.5)

    Notes
    -----
    The order=3 implementation is differentiable in JAX and uses a separable
    4x4 cubic-convolution stencil. It is not the same algorithm as SciPy's
    spline-prefiltered ndimage.map_coordinates(order=3), but it provides true
    cubic interpolation while remaining JAX-native.
    """
    if mode != "constant":
        raise NotImplementedError(
            "map_coordinates_2d currently supports only mode='constant'."
        )
    if order not in (0, 1, 3):
        raise NotImplementedError(
            "map_coordinates_2d supports order 0 (nearest), "
            "1 (linear), or 3 (cubic)."
        )

    img = jnp.asarray(image)
    if img.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {img.shape}")

    coords = jnp.asarray(coords)
    if coords.shape[0] != 2:
        raise ValueError(f"coords must have shape (2, ...), got {coords.shape}")

    r = coords[0]
    c = coords[1]
    H, W = img.shape
    cval_arr = jnp.asarray(cval, dtype=img.dtype)

    def sample(rr, cc):
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr_clip = jnp.clip(rr, 0, H - 1)
        cc_clip = jnp.clip(cc, 0, W - 1)
        val = img[rr_clip, cc_clip]
        return jnp.where(valid, val, cval_arr)

    if order == 0:
        rr = jnp.rint(r).astype(jnp.int32)
        cc = jnp.rint(c).astype(jnp.int32)
        return sample(rr, cc).astype(img.dtype)

    r0 = jnp.floor(r).astype(jnp.int32)
    c0 = jnp.floor(c).astype(jnp.int32)

    if order == 1:
        r1 = r0 + 1
        c1 = c0 + 1

        dr = r - r0.astype(r.dtype)
        dc = c - c0.astype(c.dtype)

        v00 = sample(r0, c0)
        v10 = sample(r1, c0)
        v01 = sample(r0, c1)
        v11 = sample(r1, c1)

        w00 = (1.0 - dr) * (1.0 - dc)
        w10 = dr * (1.0 - dc)
        w01 = (1.0 - dr) * dc
        w11 = dr * dc

        out = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
        return out.astype(img.dtype)

    # order == 3: bicubic convolution using the Keys cubic kernel.
    # Four samples are used along each axis: floor(x) + {-1, 0, 1, 2}.
    def cubic_kernel(x, a=-0.5):
        ax = jnp.abs(x)
        ax2 = ax * ax
        ax3 = ax2 * ax

        inner = (a + 2.0) * ax3 - (a + 3.0) * ax2 + 1.0
        outer = a * ax3 - 5.0 * a * ax2 + 8.0 * a * ax - 4.0 * a

        return jnp.where(
            ax <= 1.0,
            inner,
            jnp.where(ax < 2.0, outer, 0.0),
        )

    offsets = (-1, 0, 1, 2)
    out = jnp.zeros_like(r, dtype=img.dtype)

    for i in offsets:
        rr = r0 + i
        wr = cubic_kernel(r - rr.astype(r.dtype))

        for j in offsets:
            cc = c0 + j
            wc = cubic_kernel(c - cc.astype(c.dtype))
            out = out + (wr * wc).astype(img.dtype) * sample(rr, cc)

    return out.astype(img.dtype)


class ApplySensitivities(dl.layers.detector_layers.DetectorLayer):

    FF: jax.Array
    SRF: jax.Array

    def __init__(
        self,
        FF,
        SRF,
    ):
        self.FF = FF
        self.SRF = SRF

    @property
    def sensitivity_map(self):
        oversample = self.SRF.shape[0]
        npix = self.FF.shape[1]
        bc_sens_map = self.SRF[None, :, None, :] * self.FF[:, None, :, None]
        return bc_sens_map.reshape((npix * oversample, npix * oversample))

    def apply(self, PSF):
        return PSF * self.sensitivity_map


class PixelAnisotropy(dl.layers.detector_layers.DetectorLayer):
    transform: dl.CoordTransform
    order: int

    def __init__(self, order=3):
        order = int(order)
        if order not in (0, 1, 3):
            raise ValueError(
                "PixelAnisotropy supports order 0 (nearest), "
                "1 (linear), or 3 (cubic), "
                f"got {order}."
            )

        self.transform = dl.CoordTransform(compression=np.ones(2))
        self.order = order

    def __getattr__(self, key):
        if hasattr(self.transform, key):
            return getattr(self.transform, key)
        raise AttributeError(f"PixelAnisotropy has no attribute {key}")

    def __call__(self, PSF):
        return self.apply(PSF)

    def apply(self, PSF):
        npix = PSF.data.shape[0]
        transformed = self.transform.apply(
            dlu.pixel_coords(npix, npix * PSF.pixel_scale)
        )
        coords = np.roll(
            pix2arr(transformed, PSF.pixel_scale),
            1,
            axis=0,
        )

        data = map_coordinates_2d(
            PSF.data,
            coords,
            order=self.order,
            mode="constant",
            cval=0.0,
        )
        return PSF.set("data", data)


class NIRCamExposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    mjd: str = eqx.field(static=True)
    data: Array
    err: Array
    bad: Array

    fit: object = eqx.field(static=True)

    def __init__(self, filename, name, filter, data, mjd, err, fit, bad):
        """
        Initialise exposure
        """
        self.filename = filename
        self.target = name
        self.filter = filter
        self.data = data
        self.err = err
        self.bad = bad

        self.mjd = mjd

        self.fit = fit

    def get_key(self, param):
        if param == "spectrum":
            return f"{self.filter}"  # shared

        if param == "fluxes":
            return self.key  # per exposure (filename|pupil)

        if param in ("positions", "aberrations", "defocus"):
            return self.key

        raise ValueError(f"Unknown param: {param}")

    def map_param(self, param):
        return self.fit.map_param(self, param)

    @property
    def key(self):
        return self.filename


def exposure_from_defocus_file(fname, fit, threshold=12000, crop=128):
    with fits.open(fname) as hdul:
        sci = hdul["SCI"].data
        err_im = hdul["ERR"].data

        # Multiscale center on SCI, final extraction at `crop`
        data, (yc, xc) = cutout_around_defocused_psf_multi(
            img=sci,
            size=crop,
            recenter_sizes=(1024, 512, 256),
            smooth_sigma=2.0,
            thresh_sigma=5.0,
        )

        # Extract ERR cutout using the exact same final center
        err, _ = extract_cutout(err_im, (yc, xc), crop)

        data = jnp.asarray(data, dtype=float)
        err = jnp.asarray(err, dtype=float)

        print(err.shape, data.shape)

        threshold_map = jnp.where(data > threshold, 1, 0)
        bad_data = jnp.isnan(data)
        bad = bad_data + threshold_map

        err = jnp.where(bad, jnp.nan, err)
        data = jnp.where(bad, jnp.nan, data)

    hdr = fits.getheader(fname, ext=0)
    obs_id = hdr["OBS_ID"]
    pupil = str(hdr.get("PUPIL", "UNKNOWN")).upper()

    # encode pupil into filename (immutable-safe)
    filename = f"{obs_id}|{pupil}"
    name = obs_id  # keep name clean if it exists internally
    filter = "F212N"
    mjd = hdr["DATE"]

    return NIRCamExposure(filename, name, filter, data, mjd, err, fit, bad)


from abc import abstractmethod
from typing import Optional

import jax.numpy as np
from jax import lax
import numpy as onp
import jax.scipy as jsp
import jax.nn as jnn


from functools import partial

# --- keep these names distinct ---
import jax.numpy as jnp
import numpy as onp

# ... your earlier imports (jax, eqx, jsp, etc.) ...


filter_file = str(files("camino_data") / "F212N.dat")


def calc_throughput(file_path, nwavels=1):

    # Load throughput file directly
    wl_array, throughput_array = onp.loadtxt(file_path, unpack=True)

    # Convert to JAX arrays
    wl = jnp.asarray(wl_array)
    tp = jnp.asarray(throughput_array)

    edges = jnp.linspace(wl.min(), wl.max(), nwavels + 1)
    wavels = jnp.linspace(wl.min(), wl.max(), 2 * nwavels + 1)[1::2]

    areas = []

    for i in range(nwavels):
        cond = (edges[i] < wl) & (wl < edges[i + 1])
        throughput = jnp.where(cond, tp, 0.0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl))

    areas = jnp.stack(areas)
    weights = areas / areas.sum()

    wavels = wavels * 1e-10  # Angstrom -> metres

    return wavels, weights


# JAX-friendly poly spectrum, keep as you had it but use jnp, not np
class NonNormalisedClippedPolySpectrum:
    def __init__(self, x: jnp.ndarray, coeffs: jnp.ndarray, clip_nonneg: bool = False):
        self.x = jnp.asarray(x, dtype=float)
        self.coeffs = jnp.asarray(coeffs, dtype=float)
        self.clip_nonneg = clip_nonneg
        if self.coeffs.ndim != 1:
            raise ValueError("Coefficients must be a 1D array.")

    def _poly(self, x):
        w = 0.0
        for c in self.coeffs[::-1]:
            w = w * x + c
        return w

    @property
    def weights(self):
        inten = jnp.power(10.0, jax.vmap(self._poly)(self.x))
        if self.clip_nonneg:
            inten = jnp.clip(inten, 0.0, None)
        return inten


from typing import Optional
import equinox as eqx


class ModelFit(zdx.Base):

    @abstractmethod
    def __call__(self, model, exposure):
        pass

    def get_key(self, exposure, param):
        match param:
            case "positions":
                return exposure.key
            case "aberrations":
                return exposure.key
            case "defocus":
                return exposure.key
            case "pupil_delta":
                return exposure.key
            case "spectrum":
                return f"{exposure.filter}"
            case "fluxes":
                return exposure.key
            case _:
                raise ValueError(f"Parameter {param} has no key")

    def map_param(self, exposure, param):
        """
        currently everything's global so this is just a fallthrough
        """
        if param in [
            "fluxes",
            "positions",
            "spectrum",
            "aberrations",
            "defocus",
        ]:  # , "aberrations", "cold_mask_shift", "cold_mask_rot", "cold_mask_scale", "cold_mask_shear", "primary_rot", "primary_scale", "primary_shear", "breathing", "slope", "spectrum"]:
            return f"{param}.{exposure.get_key(param)}"
        return param

    def update_optics_zernikes(self, model, exposure):
        optics = model.optics
        if "aberrations" in model.params.keys():

            coefficients = model.aberrations[self.get_key(exposure, "aberrations")]

            # Nuke the piston gradient to prevent degeneracy
            fixed_piston = lax.stop_gradient(coefficients[0, 0])
            # print("coeff type:", type(coefficients))
            # print("is jax array:", isinstance(coefficients, jax.Array))
            # coefficients = coefficients.at[0, 0].set(fixed_piston)

            # Stop gradient for science targets
            # if not self.calibrator:
            #    coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil.coefficients", coefficients)

        if "defocus" in model.params.keys():
            disp = model.defocus[self.get_key(exposure, "defocus")]
            optics = optics.set("defocus", disp)

        """if self.fit_reflectivity:
            coefficients = model.reflectivity[self.get_key("reflectivity")]

            # Stop gradient for science targets
            if not self.calibrator:
                coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil_mask.amp_coeffs", coefficients)

        optics = optics.set("defocus", model.defocus[self.get_key("defocus")])"""

        return optics


def update_optics(self, model, exposure):
    optics = model.optics

    if "aberrations" in model.params:
        key = self.get_key(exposure, "aberrations")

        # params are stored in **nm**
        opd_nm = model.aberrations[key]  # nm

        # piston removal can be done in nm (equivalent after scaling)
        pmask = (optics.layers["pupil"].transmission > 0).astype(
            jnp.float64
        )  # opd_nm.dtype
        mean_nm = jnp.sum(opd_nm * pmask) / (jnp.sum(pmask) + 1e-12)
        opd_nm = opd_nm - mean_nm

        # convert to **meters** for optics
        opd_m = opd_nm * 1e-9  # m

        # inject
        optics = optics.set("pupil.opd", opd_m)

    if "defocus" in model.params:
        disp = model.defocus[self.get_key(exposure, "defocus")]
        optics = optics.set("defocus", disp)

    return optics


import jax.numpy as jnp

LOG10 = jnp.log(10.0)


class SinglePointFilterFit(ModelFit):
    """Pixel-basis fitter for point-source PSFs."""

    source: dl.Telescope = eqx.field(static=True)
    nwavels: int = eqx.field(static=True)

    def __init__(self, nwavels: int = 1):
        self.source = dl.PointSource(wavelengths=[1.0])
        self.nwavels = int(nwavels)

    # JAX-safe optics updater

    def update_optics(self, model, exposure):
        optics = model.optics

        # ----------------------------
        # 1) Start from the current pupil transmission
        # ----------------------------
        base_amp = optics.layers["pupil"].transmission  # P0(x)
        amp = base_amp

        # ----------------------------
        # 2) Optional: learn pupil amplitude via pupil_delta
        # ----------------------------
        # --- pupil amplitude correction ---
        if "pupil_delta" in model.params.keys():
            base_amp = optics.layers["pupil"].transmission  # A0(x)
            pupil_mask = base_amp > 0

            delta = model.pupil_delta

            # scale: choose eps so raw delta stays O(1)
            eps_amp = 0.05  # 5% per unit in log-space (tune)
            amp = base_amp * jnp.exp(eps_amp * delta)

            # hard zero outside pupil
            amp = jnp.where(pupil_mask, amp, 0.0)

            # IMPORTANT: renormalize mean amplitude inside pupil to 1
            mean_amp = jnp.sum(amp) / (jnp.sum(pupil_mask) + 1e-12)
            amp = amp / (mean_amp + 1e-12)

            optics = optics.set("pupil.transmission", amp)

        # ----------------------------
        # 3) OPD (aberrations) + pupil-plane shear
        # ----------------------------
        if "aberrations" in model.params.keys():
            opd_map = model.aberrations[self.get_key(exposure, "aberrations")]
        else:
            # if you ever call update_optics without aberrations present
            opd_map = optics.layers["pupil"].opd

        # --- Apply pupil-plane shear as a coordinate warp of pupil-plane arrays ---
        if "pupil_shear" in model.params.keys():
            s_raw = model.pupil_shear

            # scale raw -> physical dimensionless shear coefficient
            # Start small; tune later (1e-4 to 1e-2 are typical exploration ranges)
            eps_shear = 1e-3
            shx = eps_shear * s_raw
            shy = 0.0

            # Shear pupil transmission (smooth is better for gradients)
            # If you want *strict* binary display later, threshold outside optimisation.
            amp = apply_pupil_shear(amp, shx=shx, shy=shy, order=1, cval=0.0)

            # Shear OPD map
            opd_map = apply_pupil_shear(opd_map, shx=shx, shy=shy, order=1, cval=0.0)

        # ----------------------------
        # 4) Piston removal using the (possibly sheared) pupil mask
        # ----------------------------
        pupil_mask = amp > 0
        mean_val = jnp.sum(opd_map * pupil_mask) / (jnp.sum(pupil_mask) + 1e-12)
        opd_map = opd_map - mean_val

        # ----------------------------
        # 5) Write back into optics
        # ----------------------------
        optics = optics.set("pupil.transmission", amp)
        optics = optics.set("pupil.opd", opd_map)

        # ----------------------------
        # 6) Defocus with optional global scale
        # ----------------------------
        if "defocus" in model.params.keys():
            disp = model.defocus[
                self.get_key(exposure, "defocus")
            ]  # nominal defocus param

            scale = 1.0
            if "defocus_scale" in model.params.keys():
                k_raw = model.defocus_scale
                eps_k = 5e-3
                scale = 1.0 + eps_k * k_raw

            optics = optics.set("defocus", disp * scale)

        return optics

    # Forward model
    def __call__(self, model, exposure):
        source = self.source
        nw = self.nwavels

        # 1) Flux
        log_flux = model.get(exposure.fit.map_param(exposure, "fluxes"))
        flux = jnp.exp(log_flux * jnp.log(10.0))
        source = source.set("flux", flux)

        # 2) Position
        pos = model.get(exposure.fit.map_param(exposure, "positions"))
        source = source.set("position", pos * dlu.arcsec2rad(0.031))

        # 3) Polynomial spectrum in log10 space
        wv, filt = calc_throughput(exposure.filter, nwavels=nw)
        wv = jnp.asarray(wv)  # shape (nw,)
        filt = jnp.asarray(filt)
        filt = filt / (jnp.sum(filt) + 1e-12)  # base: normalised filter throughput

        # --- Get polynomial coefficients for this exposure ---
        if "spectrum" in model.params.keys():
            spec_param = exposure.fit.map_param(exposure, "spectrum")
            coeffs = model.get(spec_param)
            coeffs = jnp.atleast_1d(coeffs)  # ensure 1D, handles (1,) or (2,) etc
        else:
            coeffs = jnp.zeros((1,), dtype=jnp.float64)  # default = flat in log10
            # (log10_I = 0 → I = 1)

        # --- Build dimensionless wavelength coordinate ---
        lambda0 = jnp.mean(wv)
        x = (wv - lambda0) / (lambda0 + 1e-12)  # shape (nw,)

        # --- Evaluate log10 intensity p(x) and convert to linear ---
        log10_I = eval_poly_log10(x, coeffs)  # shape (nw,)

        # (Optional, but nice): remove the intercept so polynomial only changes shape,
        # and the overall normalisation is left to the flux parameter.
        log10_I = log10_I - jnp.mean(log10_I)

        I = jnp.power(10.0, log10_I)
        I = jnp.where(jnp.isfinite(I), I, 0.0)  # paranoia against NaN/inf
        I = jnp.clip(I, 0.0, jnp.inf)

        # --- Combine source SED with filter throughput ---
        weights = filt * I
        weights = weights / (jnp.sum(weights) + 1e-12)  # PSF weights sum to 1

        source = source.set("spectrum", dl.Spectrum(wv, weights))

        # 4) Optics, PSF and shear
        optics = self.update_optics(model, exposure)
        psfs = optics.model(source, return_psf=True)
        data = psfs.data

        if data.ndim == 3:
            # shape: (nwavels, ny, nx) -> integrate over wavelength
            psf = data.sum(axis=0)
        elif data.ndim == 2:
            # already a 2D PSF (ny, nx)
            psf = data
        else:
            raise ValueError(
                f"Unexpected PSF data ndim={data.ndim}, shape={data.shape}"
            )

        pixel_scale = psfs.pixel_scale.mean()

        if "jitter_raw" in model.params.keys():
            # map raw -> positive jitter in arcsec (choose a scale that makes raw~O(1))
            # e.g. 1 mas = 1e-3 arcsec
            jitter_arcsec = 1e-3 * jnn.softplus(model.jitter_raw)  # >= 0
            jitter_rad = dlu.arcsec2rad(jitter_arcsec)

            sigma_pix = jitter_rad / (pixel_scale + 1e-30)
            psf = gaussian_blur_fft(psf, sigma_pix)

        # if "primary_shear" in model.params.keys():
        #     shear_raw = model.primary_shear  # scalar
        #     eps_shear = 7.1e-3                 # physical shear = eps_shear * raw
        #     shx = eps_shear * shear_raw
        #     psf = apply_shear(psf, shx=shx, shy=0.0)

        if "radial_scale" in model.params.keys():
            raw = model.radial_scale  # scalar
            eps_rs = 5e-3  # 0.5% per unit
            scale = 1.0 + eps_rs * raw
            psf = radial_zoom(psf, scale)

        psf_obj = dl.PSF(psf, pixel_scale)
        return dlu.downsample(psf_obj.data, 4, mean=False)


def get_pupil(exp):
    if "|" in exp.filename:
        return exp.filename.split("|")[-1].upper()
    raise KeyError(f"No pupil tag in exp.filename={exp.filename!r}")


def get_filter(file_path, nwavels=1):

    filters = {}

    data = onp.loadtxt(file_path, unpack=True)

    filt = "F212N"

    wl_array, throughput_array = np.array(
        onp.loadtxt(file_path, unpack=True)
    )  # the F444W filter is the NIRCam one

    edges = onp.linspace(wl_array.min(), wl_array.max(), nwavels + 1)
    wavels = onp.linspace(wl_array.min(), wl_array.max(), 2 * nwavels + 1)[1::2]

    areas = []
    for i in range(nwavels):
        cond1 = edges[i] < wl_array
        cond2 = wl_array < edges[i + 1]
        throughput = onp.where(cond1 & cond2, throughput_array, 0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl_array))

    areas = onp.array(areas)
    weights = areas / areas.sum()

    wavels *= 1e-10
    spec = onp.array([wavels, weights])

    filters[filt] = onp.array([wavels, weights])

    return filters


def get_filter_test(file):
    flt = np.asarray(pd.read_csv(file, sep=" "))

    wv = flt[:, 0]
    bp = flt[:, 1]

    ebp = bp / (wv / 1e4)

    nebp = ebp / np.sum(ebp) * (np.max(wv) - np.min(wv)) * 0.01
    final = flt.at[:, 1].set(nebp)

    return final


FILTER_FILE = str(files("camino_data") / "F212N.dat")

filter_files = {"F212N": get_filter_test(FILTER_FILE)[:5, :]}


class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):

        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values


import jax.tree_util as jtu


def set_array(pytree):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)


class ModelParams(BaseModeller):

    def __getitem__(self, key):
        return self.params[key]

    def __getattr__(self, key):

        # Make the object act like a real dictionary
        if hasattr(self.params, key):
            return getattr(self.params, key)

        if key in self.params.keys():
            return self.params[key]

        for sub_key, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)

        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set(
            "params", dict([(param, getattr(values, param)) for param in self.keys()])
        )

    def from_model(self, values):
        return self.set(
            "params", dict([(param, values.get(param)) for param in self.keys()])
        )

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree.map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree.map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def map(self, fn):
        return jax.tree.map(lambda x: fn(x), self)

    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(list(self.keys()), list(self.values()))

    def partition(self, params):
        """params can be a model params object or a list of keys"""
        if isinstance(params, ModelParams):
            params = list(params.params.keys())
        return (
            ModelParams({param: self[param] for param in params}),
            ModelParams(
                {param: self[param] for param in self.keys() if param not in params}
            ),
        )

    def combine(self, params2):
        return ModelParams({**self.params, **params2.params})

    def jacfwd(self, fn, n_batch=1):
        X, unravel_fn = ravel_pytree(self)
        Xs = np.array_split(X, n_batch)
        rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
        lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
        starts = np.concatenate([np.array([0]), lens])

        @eqx.filter_jacfwd
        def batched_jac_fn(x, index):
            model_params = unravel_fn(rebuild(x, index))
            return eqx.filter_jit(fn)(model_params)

        return np.concatenate(
            [batched_jac_fn(x, index) for x, index in zip(Xs, starts)], axis=-1
        )


def solve_flux_bg_weighted_jax_nansafe(img, err, bad, m_unit, EPS=1e-12):
    img0 = jnp.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    err0 = jnp.nan_to_num(err, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)

    good = (~bad) & jnp.isfinite(err0) & (err0 > 0) & jnp.isfinite(img0)

    # avoid dividing on bad pixels
    err_safe = jnp.where(good, err0, 1.0)
    w = jnp.where(good, 1.0 / (jnp.maximum(err_safe, EPS) ** 2), 0.0)

    Smm = jnp.sum(w * m_unit * m_unit)
    Smy = jnp.sum(w * m_unit * img0)
    Smb = jnp.sum(w * m_unit)
    Sbb = jnp.sum(w)
    Sby = jnp.sum(w * img0)

    det = Smm * Sbb - Smb * Smb + EPS
    f_star = (Smy * Sbb - Smb * Sby) / det
    b_star = (Smm * Sby - Smb * Smy) / det
    return f_star, b_star


def inject_views_for_pupil(params, pup, exp):
    k_pos = exp.fit.get_key(exp, "positions")
    k_def = exp.fit.get_key(exp, "defocus")
    k_ab = exp.fit.get_key(exp, "aberrations")
    k_flux = exp.fit.get_key(exp, "fluxes")  # unused now, but fine to keep

    # shared OPD view
    params["aberrations"][k_ab] = params["aberrations_shared"]

    if pup == "WLP8":
        params["positions"][k_pos] = params["positions_wlp8"][k_pos]
        params["defocus"][k_def] = params["defocus_wlp8"][k_def]
    elif pup == "WLM8":
        params["positions"][k_pos] = params["positions_wlm8"][k_pos]
        params["defocus"][k_def] = params["defocus_wlm8"][k_def]
    else:
        raise ValueError(pup)

    # IMPORTANT: do NOT set params["fluxes"][k_flux] here
    # flux is solved by LLS inside the loss / diagnostics


def fft_log(x, eps=1e-30):
    X = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x)))
    return jnp.log10(jnp.abs(X) + eps)


def fft_log(x, eps=1e-30):
    X = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x)))
    return jnp.log10(jnp.abs(X) + eps)


def check_convergence_from_file(
    params, model_defocus, pup, fname, nw_list=(5, 10), crop_to=128
):
    """Compare PSFs across different nwavels for a given WLP8/WLM8 file."""
    images = {}

    for nw in nw_list:
        fit = SinglePointFilterFit(nwavels=nw)
        exp = exposure_from_defocus_file(fname, fit)  # new exposure with new fitter

        # ensure model sees correct pos/defocus/flux + shared OPD
        inject_views_for_pupil(params, pup, exp)

        mdl = params.inject(model_defocus)
        img = exp.fit(mdl, exp)

        # match your usual view
        img = dlu.resize(img, crop_to)

        images[nw] = np.asarray(img, dtype=float)

    ref_nw = list(nw_list)[-1]
    ref_img = images[ref_nw]

    results = {}
    for nw in list(nw_list)[:-1]:
        diff = np.sqrt(np.mean((images[nw] - ref_img) ** 2)) / (
            np.mean(ref_img) + 1e-30
        )
        results[(nw, ref_nw)] = diff

    return results, images


def check_poly_vs_mono(model, exposure, nw=20, plot=False, tol=1e-3, label=""):
    """Diagnose whether the current model is using a polynomial spectrum."""
    key = exposure.fit.map_param(exposure, "spectrum")
    try:
        coeffs = model.get(key)
    except Exception:
        print(
            f"[{label}] No 'spectrum' param for key {key} → likely MONO (only flux × filter)."
        )
        return {
            "mode": "mono",
            "reason": "missing spectrum param",
            "key": key,
            "label": label,
        }

    coeffs = jnp.asarray(coeffs)
    ncoef = int(coeffs.size)
    deg_guess = max(0, ncoef - 1)

    print(f"\n[{label}] 'spectrum' present: {key}")
    print(f"[{label}] coeff count = {ncoef}  (degree ≈ {deg_guess})")
    print(f"[{label}] coeffs = {np.asarray(coeffs)}")

    # Build wavelength grid + filter
    wv, filt = calc_throughput(exposure.filter, nwavels=nw)  # wv [m], filt shape (nw,)

    # Evaluate polynomial spectrum on x ∈ [-1, 1]
    x = jnp.linspace(-1.0, 1.0, nw)
    inten = NonNormalisedClippedPolySpectrum(x, coeffs).weights
    # If your implementation expects strictly-positive intensities:
    # inten = 10.0 ** inten

    flat = jnp.ones_like(inten)
    rel_std = float(jnp.std(inten / jnp.mean(inten)))
    is_effectively_flat = rel_std < tol

    print(f"[{label}] relative std of spectrum (flat=0): {rel_std:.3e}")
    mode = "mono_effective" if is_effectively_flat else "poly"

    shaped = np.asarray((inten * filt) / (jnp.max(inten * filt) + 1e-12))

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(wv * 1e6, shaped, marker="o", label="spectrum × filter")
        plt.plot(
            wv * 1e6,
            (flat * filt) / (np.max(filt) + 1e-30),
            lw=2,
            alpha=0.6,
            label="flat × filter (mono ref)",
        )
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Normalized weight")
        plt.title(
            f"{label} {exposure.filter}: {'POLY' if mode=='poly' else 'MONO-like'} (nw={nw})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "label": label,
        "mode": mode,
        "ncoef": ncoef,
        "degree_guess": deg_guess,
        "rel_std_spectrum": rel_std,
        "key": key,
    }


def weights_used_by_fit(params, exposure, nw=20):
    wv, filt = calc_throughput(exposure.filter, nwavels=nw)  # (nw,)

    # If spectrum isn't in params, it's mono/flat for this diagnostic
    if ("spectrum" not in params.params) and (not hasattr(params, "spectrum")):
        shaped = np.asarray(filt)  # flat spectrum × filter
        return wv, shaped

    # Otherwise, read coeffs from the spectrum dict
    k_spec = exposure.fit.get_key(exposure, "spectrum")

    # support both access patterns
    if "spectrum" in params.params:
        coeffs = params.params["spectrum"][k_spec]
    else:
        coeffs = params["spectrum"][k_spec]

    x = jnp.linspace(-1.0, 1.0, nw)
    inten = NonNormalisedClippedPolySpectrum(x, coeffs).weights
    shaped = np.asarray(inten * filt)
    return wv, shaped


def is_curve_not_flat(shaped, tol=1e-3):
    m = shaped.mean()
    if m == 0:
        return False, 0.0
    rel_std = float(np.std(shaped / m))
    return rel_std > tol, rel_std


def solve_flux_bg_weighted_jax_nansafe(img, err, bad, m_unit, EPS=1e-12):
    img0 = jnp.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    err0 = jnp.nan_to_num(err, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)

    good = (~bad) & jnp.isfinite(err0) & (err0 > 0) & jnp.isfinite(img0)

    # avoid dividing on bad pixels
    err_safe = jnp.where(good, err0, 1.0)
    w = jnp.where(good, 1.0 / (jnp.maximum(err_safe, EPS) ** 2), 0.0)

    Smm = jnp.sum(w * m_unit * m_unit)
    Smy = jnp.sum(w * m_unit * img0)
    Smb = jnp.sum(w * m_unit)
    Sbb = jnp.sum(w)
    Sby = jnp.sum(w * img0)

    det = Smm * Sbb - Smb * Smb + EPS
    f_star = (Smy * Sbb - Smb * Sby) / det
    b_star = (Smm * Sby - Smb * Smy) / det
    return f_star, b_star


def scale_poisson_no_bg(img, psf_unit, bad):
    m = jnp.where(bad, 0.0, psf_unit)
    d = jnp.where(bad, 0.0, img)
    num = jnp.sum(d)
    den = jnp.sum(m) + 1e-30
    return num / den


def scale_ls_const_bg_unweighted(img, psf_unit, bad):
    M = jnp.where(bad, 0.0, psf_unit)
    D = jnp.where(bad, 0.0, img)
    Smm = jnp.sum(M * M)
    Sm1 = jnp.sum(M)
    S11 = jnp.sum(~bad).astype(M.dtype)
    Sdm = jnp.sum(D * M)
    Sd1 = jnp.sum(D)
    det = Smm * S11 - Sm1 * Sm1 + 1e-30
    f = (S11 * Sdm - Sm1 * Sd1) / det
    b = (-Sm1 * Sdm + Smm * Sd1) / det
    return f, b


def inject_into_model(model, params_mp):
    return eqx.tree_at(lambda m: m.params, model, params_mp.params)


def inject_views_for_pupil(params_mp, pup, exp):
    suf = "wlp8" if pup.upper() == "WLP8" else "wlm8"

    k_pos = exp.fit.get_key(exp, "positions")
    k_def = exp.fit.get_key(exp, "defocus")
    k_flux = exp.fit.get_key(exp, "fluxes")
    k_ab = exp.fit.get_key(exp, "aberrations")

    p = params_mp

    # positions: storage -> active
    p = eqx.tree_at(
        lambda t: t.params["positions"][k_pos],
        p,
        p.params[f"positions_{suf}"][k_pos],
    )

    # defocus: storage -> active
    p = eqx.tree_at(
        lambda t: t.params["defocus"][k_def],
        p,
        p.params[f"defocus_{suf}"][k_def],
    )

    # flux: optional storage -> active
    storage_flux_key = f"fluxes_{suf}"
    if storage_flux_key in p.params.keys():
        p = eqx.tree_at(
            lambda t: t.params["fluxes"][k_flux],
            p,
            p.params[storage_flux_key][k_flux],
        )

    # aberrations: shared -> active
    p = eqx.tree_at(
        lambda t: t.params["aberrations"][k_ab],
        p,
        p.params["aberrations_shared"],
    )

    return p


def make_gaussian_kernel1d(sigma: float, truncate: float = 4.0, dtype=onp.float64):
    # static (NumPy) kernel
    radius = int(onp.ceil(truncate * sigma))
    x = onp.arange(-radius, radius + 1, dtype=dtype)
    k = onp.exp(-0.5 * (x / sigma) ** 2)
    k = k / onp.sum(k)
    return jnp.asarray(k, dtype=jnp.asarray(0.0, dtype=jnp.float64).dtype)


def separable_gaussian_blur_reflect(img, k1d):
    # img: (H,W)   k1d: (K,)
    img = jnp.asarray(img)
    k1d = jnp.asarray(k1d, dtype=img.dtype)
    r = (k1d.shape[0] - 1) // 2

    # reflect padding
    x = jnp.pad(img, ((r, r), (r, r)), mode="reflect")

    # conv along x then y using lax.conv_general_dilated
    # reshape kernels for separable conv
    kx = k1d[None, :, None, None]  # (1, K, 1, 1)
    ky = k1d[:, None, None, None]  # (K, 1, 1, 1)

    # add N,C dims
    x = x[None, :, :, None]  # (1, H+2r, W+2r, 1)

    # convolve width
    x = lax.conv_general_dilated(
        x,
        kx,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # convolve height
    x = lax.conv_general_dilated(
        x,
        ky,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    return x[0, :, :, 0]  # back to (H,W)


def gaussian_smooth_nan_jax_static(img, sigma=2.0, truncate=4.0):
    """NaN-aware Gaussian smoothing, JIT-safe, sigma fixed."""
    img = jnp.asarray(img, dtype=jnp.float64)

    # sigma=0 behaves like identity
    if sigma <= 0:
        return img

    k1d = make_gaussian_kernel1d(float(sigma), float(truncate), dtype=onp.float64)

    m = jnp.isfinite(img)
    img0 = jnp.where(m, img, 0.0)
    w0 = m.astype(img.dtype)

    img_s = separable_gaussian_blur_reflect(img0, k1d)
    w_s = separable_gaussian_blur_reflect(w0, k1d)

    out = img_s / jnp.maximum(w_s, 1e-12)
    out = jnp.where(m, out, jnp.nan)
    return out


# Functions for monkey patching dict methods to make ModelParams act like a dict for params
def _mp_contains(self, k):
    return k in self.params


def _mp_iter(self):
    return iter(self.params)


def _mp_len(self):
    return len(self.params)


def _mp_keys(self):
    return self.params.keys()


def _mp_items(self):
    return self.params.items()


print("camino module loaded successfully")

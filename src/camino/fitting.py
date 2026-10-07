"""CAMINO single-pair wavefront fitting, from a zero OPD or a previous fit.

Public entry points
-------------------
load_data(...) -> FitProblem: load a WLP8/WLM8 pair and build the model.
fit_data(..., fit_mode="pixel" | "ptt_pixel") -> FitResult: run the fit.
fit_data(..., initial=previous_result) -> FitResult: warm start. Skip the
    initial stages, refit positions, then run only the final L-BFGS-B.
fit_data(ptt_data, initial=pixel_result, reseed="all" | ids) -> FitResult:
    a ptt_pixel fit from a pixel fit (e.g. the previous epoch), re-searching
    the seed grid on all or the listed mirrors.
diagnose_mirror_changes(ptt_data, result, reference): find mirrors that moved.
continue_fit(result, maxiter=2000): continue the final L-BFGS-B stage.
compare_with_mast(result, mast_wss_path): make the notebook comparison figure.

Example (no previous OPD is required or loaded)::

    from camino.fitting import FitConfig, load_data, fit_data
    config = FitConfig.for_mode("ptt_pixel", stage2_maxiter=6000)
    data = load_data("plus.fits", "minus.fits", pupil_path="pupil.fits",
                     filter_path="F212N.dat", config=config)
    result = fit_data(data=data, output_dir="fit_output")
    result.plot_opd()
    # result.continue_fit(maxiter=2000)
    # result.compare_with_mast("official_wss.fits")
    # next_result = fit_data(data=next_data, initial=result)

Numerical conventions retained from the supplied notebooks
---------------------------------------------------------
* Detector cutout size is FitConfig.fit_npix (pixel: 128 by default,
  ptt_pixel: 256); 256-pixel windows work in pixel mode with the same
  stage-1 learning rates.
* pixel: output flip on axis 0, scheduled
  SGD of defocus/positions (60 steps), then illuminated-pixel L-BFGS-B.
  QV is segment-wise on the piston-centred OPD in nm.
* ptt_pixel: pupil flip on axis 0; BFGS positions and defocus,
  sequential 27-seed/local bounded L-BFGS-B per mirror, joint PTT BFGS,
  position/defocus refit, then joint PTT/pixel L-BFGS-B. The pixel residual
  is kept orthogonal to every mirror's piston/tip/tilt, and the per-mirror
  stage propagates only the mirror being fitted (ptt_orthogonal_residual,
  ptt_segment_propagation). QV, L2 and global-slope regularisation act on
  the pixel residual only.
* The two orientations give identical images for the same OPD; only the
  sign of the y position differs, which warm starts account for.
* Both propagate on a 512-pixel pupil. The output field defaults to the
  detector cutout (psf_npixels = fit_npix, times oversample): the MFT
  evaluates each output pixel independently, so a larger field only adds
  pixels that render() crops away.
* The ptt_pixel preset starts its defocus at the source notebook's values,
  +44010282.72 and -43542523.19 nm, and fits it (bfgs_defocus); set
  bfgs_defocus=False to hold them fixed as the notebook did.
* All 18 physical mirrors are fitted in ptt_pixel mode. Split labels share
  one plane and one coordinate centre/scale. No 100-nm selection is applied.
* Pupil amplitude remains frozen; flux/background are solved analytically.
* Continuation starts at the latest parameters with fresh L-BFGS memory.

Dependencies: camino.core and camino.abcdlux_patch, JAX, Equinox, dLux,
Optax, NumPy, SciPy, Astropy, scikit-image, Matplotlib and tqdm. It does
not execute fits, read data or select a Matplotlib backend at import time.
JAX float64 is enabled by load_data, as required by both source notebooks.
The existing CAMINO throughput API needs a temporary module patch; calls
through this module are serialised and the original function is restored.
Run concurrent fits in separate processes, not alongside unrelated CAMINO
calls in another thread.

Validation: syntax, undefined-global checks and dependency-light numerical
comparisons passed, including both regularisation formulas, grouped PTT
planes, SciPy continuation and a synthetic MAST comparison figure. Full
CAMINO/JAX optical fits have not been executed in the build environment.

Source notebooks:
    githubversioncaminofit_lbfgsb(4).ipynb
    fit_real_event_grouped_physical_ptt(2).ipynb
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property, partial
from itertools import product
from pathlib import Path
from threading import RLock
from typing import Any, Callable
import json
import time

import numpy as np
import scipy.optimize as spo
import astropy.io.fits as fits
import dLux as dl
import dLux.utils as dlu
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from jax import Array
from skimage import measure
from tqdm import tqdm  # text bars also render in saved notebooks

from . import abcdlux_patch as abcdlux
from . import core as cam

onp = np
EPS = 1e-30
_FILTER_LOCK = RLock()
PHYSICAL_TO_LABELS = {
    1: (1,),
    2: (2, 4),
    3: (3, 5),
    4: (6,),
    5: (7,),
    6: (8,),
    7: (9,),
    8: (10,),
    9: (11,),
    10: (12,),
    11: (13,),
    12: (14,),
    13: (15,),
    14: (16,),
    15: (17, 18),
    16: (21, 22),
    17: (19,),
    18: (20,),
}

__all__ = [
    "FitConfig",
    "FitProblem",
    "FitResult",
    "StageHistory",
    "load_data",
    "fit_data",
    "continue_fit",
    "compare_with_mast",
    "get_mast_wss_path",
    "diagnose_mirror_changes",
    "PHYSICAL_TO_LABELS",
    "NIRCamFresnelOptics",
    "NRCDetectorLong",
]


@dataclass(frozen=True)
class FitConfig:
    """Fit settings. Use for_mode() so mode-dependent defaults stay together.

    None-valued fields below resolve in __post_init__. Custom values override
    the preset. Loss curves appear after every BFGS/L-BFGS-B invocation;
    show_plots=False suppresses display, but all histories are still retained.
    """

    fit_mode: str = "pixel"
    fit_npix: int | None = None  # detector window: 128 (pixel) / 256 (ptt_pixel)
    orientation: str | None = None
    defocus_nm: tuple[float, float] | None = None
    filter_name: str = "F212N"
    n_wavels: int = 1
    pupil_downsample_factor: int = 2
    psf_npixels: int | None = None  # defaults to fit_npix
    oversample: int = 4
    pixel_scale: float = 0.031
    pixel_pitch: float = 18e-6
    diameter: float = 6.603464
    opd_phase_sign: float = -1.0
    stage1_sgd_steps: int = 60
    # Order: defocus+, position+, defocus-, position-.
    sgd_learning_rates: tuple = (9e4, 8e-8, 9e4, 8e-8)
    sgd_start_steps: tuple = (0, 10, 15, 35)
    sgd_momentum: float = 0.6
    position_maxiter: int = 200
    position_refit_maxiter: int = 300
    position_scale: float = 1e-4
    # Fit defocus alongside positions in the BFGS position stages (default:
    # ptt_pixel only; pixel mode fits defocus by SGD in stage 1).
    bfgs_defocus: bool | None = None
    defocus_scale: float = 1e2  # nm per BFGS unit (matches position curvature)
    skip_grid_init: bool = False
    # Piston, tip, tilt bounds; the seed grid uses [lower, 0, upper].
    ptt_bounds_nm: tuple = ((-3500.0, 3500.0),) * 3
    local_ptt_maxiter: int = 50
    ptt_seed_batch: int = 9  # seeds evaluated per vectorised call
    # Fit each mirror by propagating only its segment over a cached field of
    # the others (monochromatic fits); False uses the full model throughout.
    ptt_segment_propagation: bool = True
    # Keep the ptt_pixel residual orthogonal to every mirror's piston/tip/tilt,
    # so PTT lives only in the coefficients. Otherwise the two trade planes
    # along a nearly flat valley that stalls the final L-BFGS-B.
    ptt_orthogonal_residual: bool = True
    # PTT coefficients enter the final L-BFGS-B vector as c / scale. One
    # coefficient moves thousands of pixels, so unscaled it has far larger
    # curvature than a pixel, which slows L-BFGS-B's scalar initial Hessian.
    ptt_coefficient_scale: float = 1e-2  # ~1/sqrt(pixels per mirror)
    joint_ptt_maxiter: int = 100
    ptt_gtol: float = 1e-3
    stage2_maxiter: int = 2000
    stage2_gtol: float = 1e-4
    stage2_maxcor: int = 20
    stage2_ftol: float | None = None  # pixel: 2.2e-9; ptt_pixel: 1e-8
    stage2_maxls: int = 20
    stage2_maxfun: int = 15000
    lambda_qv: float = 6e-3
    lambda_l1: float = 0.0  # pixel mode; OPD is in metres in this penalty
    lambda_pixel_l2: float | None = None
    lambda_global_plane: float | None = None
    lambda_boundary_mix: float = 0.0
    display_transmission_threshold: float = 0.5
    display_segment_outlines: bool = True
    show_plots: bool = True
    progress: bool = True

    def __post_init__(self):
        if self.fit_mode not in ("pixel", "ptt_pixel"):
            raise ValueError("fit_mode must be 'pixel' or 'ptt_pixel'")
        advanced = self.fit_mode == "ptt_pixel"
        defaults = {
            "fit_npix": 256 if advanced else 128,
            "orientation": "pupil_flip" if advanced else "output_flip",
            "defocus_nm": (44010282.72, -43542523.19) if advanced else (4.3e7, -4.3e7),
            "lambda_pixel_l2": 1e-4 if advanced else 0.0,
            "lambda_global_plane": 1e2 if advanced else 0.0,
            "bfgs_defocus": advanced,
            "stage2_ftol": 1e-8 if advanced else 2.220446049250313e-9,
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        if self.psf_npixels is None:
            object.__setattr__(self, "psf_npixels", self.fit_npix)
        if self.orientation not in ("pupil_flip", "output_flip"):
            raise ValueError("Unknown optical orientation")
        if self.opd_phase_sign not in (-1, 1):
            raise ValueError("opd_phase_sign must be -1 or 1")
        for name in (
            "fit_npix",
            "n_wavels",
            "pupil_downsample_factor",
            "psf_npixels",
            "oversample",
            "stage2_maxiter",
            "stage2_maxcor",
            "stage2_maxls",
            "stage2_maxfun",
            "ptt_seed_batch",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.fit_npix > self.psf_npixels:
            raise ValueError("fit_npix cannot exceed psf_npixels")
        for name in (
            "stage1_sgd_steps",
            "position_maxiter",
            "position_refit_maxiter",
            "local_ptt_maxiter",
            "joint_ptt_maxiter",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "position_scale",
            "defocus_scale",
            "ptt_coefficient_scale",
            "pixel_pitch",
            "diameter",
            "pixel_scale",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.defocus_nm) != 2 or not np.all(np.isfinite(self.defocus_nm)):
            raise ValueError("defocus_nm must contain two finite values in nm")
        if len(self.ptt_bounds_nm) != 3 or any(
            len(b) != 2
            or not np.all(np.isfinite(b))
            or not b[0] <= 0 <= b[1]
            or b[0] >= b[1]
            for b in self.ptt_bounds_nm
        ):
            raise ValueError("Each PTT bound must straddle zero and have lower < upper")
        if len(self.sgd_learning_rates) != 4 or len(self.sgd_start_steps) != 4:
            raise ValueError("Four SGD learning rates and start steps are required")
        if not 0 <= self.display_transmission_threshold <= 1:
            raise ValueError(
                "display_transmission_threshold must be between zero and one"
            )
        for name in (
            "ptt_gtol",
            "stage2_gtol",
            "stage2_ftol",
            "lambda_qv",
            "lambda_l1",
            "lambda_pixel_l2",
            "lambda_global_plane",
            "lambda_boundary_mix",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not advanced and any(
            (self.lambda_pixel_l2, self.lambda_global_plane, self.lambda_boundary_mix)
        ):
            raise ValueError(
                "Pixel L2/global-plane/boundary penalties belong to ptt_pixel mode"
            )
        if advanced and self.lambda_l1:
            raise ValueError("lambda_l1 belongs to pixel mode")

    @classmethod
    def for_mode(cls, fit_mode="pixel", **overrides):
        return cls(fit_mode=fit_mode, **overrides)

    def stage2_options(self, maxiter=None):
        return dict(
            maxiter=self.stage2_maxiter if maxiter is None else maxiter,
            gtol=self.stage2_gtol,
            ftol=self.stage2_ftol,
            maxcor=self.stage2_maxcor,
            maxls=self.stage2_maxls,
            maxfun=self.stage2_maxfun,
        )


def _resolve_config(fit_mode=None, config=None):
    if config is None:
        return FitConfig.for_mode(fit_mode or "pixel")
    if fit_mode is not None and fit_mode != config.fit_mode:
        raise ValueError("fit_mode and config.fit_mode disagree")
    return config


@dataclass
class StageHistory:
    """One optimiser invocation, including its true accepted-iterate losses."""

    name: str
    method: str
    initial_loss: float = np.nan
    losses: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)
    max_abs_grads: list = field(default_factory=list)
    parameter_history: list = field(default_factory=list)
    pixel_rms_nm: list = field(default_factory=list)
    eval_losses: list = field(default_factory=list)
    result: Any = None
    elapsed_seconds: float = 0.0

    def plot(self, show=True):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(len(self.losses) + 1), [self.initial_loss, *self.losses])
        ax.set(xlabel="Iteration (0 = initial)", ylabel="Loss", title=self.name)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


@contextmanager
def _local_filter(filter_path, filter_name):
    """Scope the legacy CAMINO filter redirection to this fit only."""
    with _FILTER_LOCK:
        original = cam.calc_throughput

        def local_throughput(filt, nwavels=1):
            path = Path(filter_path) if str(filt) == filter_name else filt
            return original(path, nwavels=nwavels)

        cam.calc_throughput = local_throughput
        try:
            yield
        finally:
            cam.calc_throughput = original


# Shared optical model, detector, parameter adapter and numerical helpers.
class SegmentField(eqx.Module):
    """Cached image-plane field for fitting one mirror segment at a time.

    Propagation is linear in the pupil field, so with every other segment
    fixed the image field is u_rest + U(segment). u_rest is computed once;
    each evaluation then propagates only the segment's bounding box (a
    fraction of the pupil width). A pupil pixel outside the segment tracks
    the global phase that piston removal introduces when the segment moves.
    Valid only while positions, defocus and the other segments stay fixed.
    """

    u_rest: Array  # image field of the other segments (no output phase)
    mask: Array  # segment mask in the post-layer pupil orientation
    start: Array  # (row, col) of the bounding box, dynamic
    ref: Array  # flat index of a reference pupil pixel outside the segment
    u_ref: Array  # that pixel's pupil field when u_rest was computed
    shape: tuple = eqx.field(static=True)  # common bounding-box shape

    def segment_image_field(self, optics, u_in, x_in, wavelength):
        rows, cols = self.shape
        box = jax.lax.dynamic_slice(u_in * self.mask, tuple(self.start), self.shape)
        x_box = jax.lax.dynamic_slice(x_in, (self.start[1],), (cols,))
        y_box = jax.lax.dynamic_slice(x_in, (self.start[0],), (rows,))
        return optics.propagate_field(box, (x_box, y_box), wavelength)

    def image_field(self, optics, u_in, x_in, wavelength):
        phase = u_in.ravel()[self.ref] / self.u_ref
        return self.u_rest * phase + self.segment_image_field(
            optics, u_in, x_in, wavelength
        )


class NIRCamFresnelOptics(dl.AngularOpticalSystem):
    defocus: jnp.ndarray
    fnumber: jnp.ndarray
    fnominal: jnp.ndarray
    pixel_pitch: float
    opd_phase_sign: float
    orientation: str = eqx.field(static=True)
    segment: SegmentField | None

    def __init__(
        self,
        binary_primary_mirror,
        defocus=0.0,
        opd=None,
        psf_npixels=512,
        oversample=4,
        pixel_scale=0.031,
        fnumber=None,
        pixel_pitch=18e-6,
        diameter=6.603464,
        opd_phase_sign=-1,
        wf_npixels=512,
        orientation="pupil_flip",
    ):
        self.orientation = orientation
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale
        self.defocus = defocus
        self.pixel_pitch = pixel_pitch
        self.opd_phase_sign = opd_phase_sign
        self.segment = None

        theta_native = dlu.arcsec2rad(self.psf_pixel_scale)
        theta_pix = theta_native / self.oversample
        self.fnominal = self.pixel_pitch / theta_native

        expected_native = self.pixel_pitch / self.fnominal
        expected_pix = expected_native / self.oversample

        if fnumber is None:
            fnumber = self.fnominal / self.diameter
        self.fnumber = fnumber

        transmission = binary_primary_mirror
        if opd is None:
            opd = jnp.zeros_like(transmission)

        primary = cam.JWSTPrimary(
            transmission,
            opd=jnp.zeros_like(transmission),
        ).set("opd", opd)

        layers = [("pupil", primary)]
        if orientation == "pupil_flip":
            layers.append(("InvertY", dl.Flip(0)))
        elif orientation != "output_flip":
            raise ValueError("orientation must be 'pupil_flip' or 'output_flip'")
        self.layers = dlu.list2dictionary(layers, ordered=True)

    def pupil_coords(self):
        return dlu.nd_coords(self.wf_npixels, self.diameter / self.wf_npixels)

    def pupil_field(self, wavelength, offset):
        """Complex pupil field after every layer, the phase sign and tilt."""
        wf = dl.Wavefront(wavelength, self.wf_npixels, diameter=self.diameter)

        for layer in self.layers.values():
            wf = layer(wf)

        if self.opd_phase_sign == -1.0:
            wf = wf.set(phasor=wf.phasor.conj())

        return wf.tilt(offset).phasor

    def propagate_field(self, u_in, spec_in, wavelength, output_phase=False):
        """LCT-propagate a pupil field sampled on spec_in to the detector."""
        fl_fit = self.fnumber * self.diameter
        dz = self.defocus * 1e-9
        L = self.fnominal + dz

        abcd = abcdlux.compose_abcd(
            [
                abcdlux.abcd_lens(fl_fit),
                abcdlux.abcd_free_space(L),
            ]
        )

        n_out = self.psf_npixels * self.oversample
        dx_out = self.pixel_pitch / self.oversample
        x_out = dlu.nd_coords(n_out, dx_out)

        return abcdlux.lct_prop(
            u_in=u_in,
            spec_in=spec_in,
            spec_out=x_out,
            lam=wavelength,
            ABCD=abcd,
            mode="physical",
            strip_input=False,
            output_phase=output_phase,
        )

    def propagate(
        self,
        wavelengths,
        offset=onp.zeros(2),
        weights=None,
        return_wf=False,
        return_psf=False,
    ):
        """Polychromatic PSF from |u|^2 directly.

        dLux's propagate always builds Wavefronts (abs/angle and the output
        phase factors) even when only the PSF is used; that path is kept for
        return_wf=True.
        """
        if return_wf and return_psf:
            raise ValueError(
                "return_wf and return_psf cannot both be True. Please choose one."
            )
        if return_wf:
            return super().propagate(wavelengths, offset, weights, return_wf=True)
        wavelengths = jnp.atleast_1d(wavelengths)
        if weights is None:
            weights = jnp.ones_like(wavelengths) / wavelengths.size
        psfs = jax.vmap(lambda wl: self.propagate_mono(wl, jnp.asarray(offset)))(
            wavelengths
        )
        psf = jnp.tensordot(jnp.atleast_1d(weights), psfs, axes=1)
        if return_psf:
            pixel_scale = dlu.arcsec2rad(self.psf_pixel_scale) / self.oversample
            return dl.PSF(psf, pixel_scale)
        return psf

    def propagate_mono(self, wavelength, offset=onp.zeros(2), return_wf=False):
        u_in = self.pupil_field(wavelength, offset)
        x_in = self.pupil_coords()
        if self.segment is None:
            u_out = self.propagate_field(u_in, x_in, wavelength, output_phase=return_wf)
        elif return_wf:
            raise ValueError("Segment propagation returns intensities only")
        else:
            u_out = self.segment.image_field(self, u_in, x_in, wavelength)

        theta_pix = dlu.arcsec2rad(self.psf_pixel_scale) / self.oversample
        if self.orientation == "output_flip":
            u_out = jnp.flip(u_out, axis=0)
        if not return_wf:
            # PSF = |u|^2; avoids abs/angle and the output phase factors.
            return u_out.real**2 + u_out.imag**2
        return dl.Wavefront.from_phasor(u_out, wavelength, pixel_scale=theta_pix)


class NRCDetectorLong(dl.detectors.LayeredDetector):
    dark_current: Array

    def __init__(
        self,
        angle=0.0,
        oversample=4,
        SRF=None,
        FF=None,
        downsample=True,
        npixels_in=256,
        anisotropy=True,
        dark_current=0.0,
    ):
        # Exact 256x256-crop behaviour from the uploaded helper:
        # use a simple flat field with the same size as the fitted detector stamp.
        if FF is None:
            FF = jnp.ones((npixels_in, npixels_in))

        if SRF is None:
            SRF = jnp.ones((oversample, oversample))

        layers = [("rotate", cam.Rotate(angle))]
        if anisotropy:
            layers.append(("anisotropy", cam.PixelAnisotropy(order=3)))

        # Important: the uploaded helper intentionally does NOT apply the old
        # 64x64 sensitivity crop/padding path for the 256x256 fit.
        # layers.append(("sensitivity", cam.ApplySensitivities(FF, SRF)))

        if downsample:
            layers.append(("downsample", dl.Downsample(oversample)))

        self.layers = dlu.list2dictionary(layers, ordered=True)
        self.dark_current = jnp.array(dark_current, float)


class NIRCamModel(cam.BaseModeller):
    filters: dict
    optics: NIRCamFresnelOptics
    detector: NRCDetectorLong

    def __init__(self, exposures, params, optics, detector, filter_files):
        self.optics = optics
        self.detector = detector
        self.params = params
        self.filters = {}

        for filt in [e.filter for e in exposures]:
            spec = filter_files[filt]
            spec = spec.at[:, 0].divide(1e10)
            self.filters[filt] = spec[::5, :]


def get_exposure_keys(exp):
    return {
        "k_pos": exp.fit.get_key(exp, "positions"),
        "k_flux": exp.fit.get_key(exp, "fluxes"),
        "k_def": exp.fit.get_key(exp, "defocus"),
        "k_ab": exp.fit.get_key(exp, "aberrations"),
        "k_spec": exp.fit.get_key(exp, "spectrum"),
    }


def init_params(exposures, optics, defocus_nm):
    poly_order = 2
    opd_shape = optics.layers["pupil"].opd.shape

    params_start = {
        "positions": {},
        "fluxes": {},
        "defocus": {},
        "aberrations": {},
        "spectrum": {},
        "positions_wlp8": {},
        "positions_wlm8": {},
        "defocus_wlp8": {},
        "defocus_wlm8": {},
        "aberrations_shared": jnp.zeros(opd_shape, dtype=jnp.float64),
    }

    keys = {}

    for pup, exp in exposures.items():
        k = get_exposure_keys(exp)
        keys[pup] = k

        if k["k_spec"] not in params_start["spectrum"]:
            params_start["spectrum"][k["k_spec"]] = jnp.zeros(
                (poly_order,), dtype=jnp.float64
            )

        total = jnp.nan_to_num(jnp.nansum(exp.data))
        params_start["fluxes"][k["k_flux"]] = jnp.asarray(
            jnp.log10(total + 1e-6), dtype=jnp.float64
        )
        params_start["positions"][k["k_pos"]] = jnp.array([0.0, 0.0], dtype=jnp.float64)
        params_start["defocus"][k["k_def"]] = jnp.asarray([0.0], dtype=jnp.float64)
        params_start["aberrations"][k["k_ab"]] = params_start["aberrations_shared"]

        if pup == "WLP8":
            params_start["positions_wlp8"][k["k_pos"]] = jnp.array(
                [0.0, 0.0], dtype=jnp.float64
            )
            params_start["defocus_wlp8"][k["k_def"]] = jnp.asarray(
                [defocus_nm[0]], dtype=jnp.float64
            )
        elif pup == "WLM8":
            params_start["positions_wlm8"][k["k_pos"]] = jnp.array(
                [0.0, 0.0], dtype=jnp.float64
            )
            params_start["defocus_wlm8"][k["k_def"]] = jnp.asarray(
                [defocus_nm[1]], dtype=jnp.float64
            )

    params_start["positions_wlp8_xy"] = jnp.array([0.0, 0.0], dtype=jnp.float64)
    params_start["positions_wlm8_xy"] = jnp.array([0.0, 0.0], dtype=jnp.float64)
    params_start["defocus_wlp8_val"] = jnp.array([defocus_nm[0]], dtype=jnp.float64)
    params_start["defocus_wlm8_val"] = jnp.array([defocus_nm[1]], dtype=jnp.float64)
    params_start["pupil_delta"] = jnp.zeros_like(
        optics.layers["pupil"].transmission, dtype=jnp.float64
    )

    return params_start, keys


def build_full_params(train_params, params_template, exposures):
    p = params_template

    k_wlp8 = get_exposure_keys(exposures["WLP8"])
    k_wlm8 = get_exposure_keys(exposures["WLM8"])

    p = eqx.tree_at(
        lambda t: t.params["aberrations_shared"],
        p,
        train_params["aberrations_shared"],
    )
    p = eqx.tree_at(
        lambda t: t.params["pupil_delta"],
        p,
        train_params["pupil_delta"],
    )
    p = eqx.tree_at(
        lambda t: t.params["positions_wlp8"][k_wlp8["k_pos"]],
        p,
        train_params["positions_wlp8_xy"],
    )
    p = eqx.tree_at(
        lambda t: t.params["defocus_wlp8"][k_wlp8["k_def"]],
        p,
        train_params["defocus_wlp8_val"],
    )
    p = eqx.tree_at(
        lambda t: t.params["positions_wlm8"][k_wlm8["k_pos"]],
        p,
        train_params["positions_wlm8_xy"],
    )
    p = eqx.tree_at(
        lambda t: t.params["defocus_wlm8"][k_wlm8["k_def"]],
        p,
        train_params["defocus_wlm8_val"],
    )

    return p


def build_segment_labels(pupil_path: str, downsample_factor: int):
    primary_mirror = np.asarray(fits.getdata(pupil_path))
    segment_binary_native = primary_mirror >= 256.0
    segment_labels_native = measure.label(segment_binary_native, connectivity=2)

    h, w = segment_labels_native.shape
    factor = int(downsample_factor)
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    labels = segment_labels_native[:h2, :w2]

    out = np.zeros((h2 // factor, w2 // factor), dtype=np.int32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            block = labels[
                i * factor : (i + 1) * factor, j * factor : (j + 1) * factor
            ].ravel()
            block = block[block > 0]
            if block.size:
                out[i, j] = np.bincount(block).argmax()
    return out


def ptt_dual_basis(basis, masks):
    """Least-squares dual of the PTT basis: tensordot(dual, opd, 2) gives the
    54 coefficients whose ptt_map best fits opd on each mirror."""
    basis, masks = np.asarray(basis), np.asarray(masks)
    dual = np.zeros_like(basis)
    for i, mask in enumerate(masks):
        b = basis[3 * i : 3 * i + 3][:, mask]  # (3, pixels on this mirror)
        dual[3 * i : 3 * i + 3][:, mask] = np.linalg.solve(b @ b.T, b)
    return jnp.asarray(dual)


def center_opd_on_pupil(opd, mask):
    mask_f = mask.astype(jnp.float64)
    mean = jnp.sum(opd * mask_f) / (jnp.sum(mask_f) + EPS)
    opd_centered = (opd - mean) * mask_f
    return jnp.where(mask, opd_centered, opd), mean


def qv_masked_nm(x_nm, mask):
    mask = mask.astype(jnp.float64)
    dx = x_nm[1:, :] - x_nm[:-1, :]
    mx = mask[1:, :] * mask[:-1, :]
    dy = x_nm[:, 1:] - x_nm[:, :-1]
    my = mask[:, 1:] * mask[:, :-1]
    return jnp.sum((dx * mx) ** 2) + jnp.sum((dy * my) ** 2)


def quadratic_variation_masked(image, mask):
    image = jnp.asarray(image, dtype=jnp.float64)
    mask = jnp.asarray(mask, dtype=jnp.float64)

    dx = image[1:, :] - image[:-1, :]
    mx = mask[1:, :] * mask[:-1, :]

    dy = image[:, 1:] - image[:, :-1]
    my = mask[:, 1:] * mask[:, :-1]

    return jnp.sum((dx * mx) ** 2) + jnp.sum((dy * my) ** 2)


@partial(jax.jit, static_argnums=(2,))
def segmentwise_qv(labeled_array, data, unique_labels):
    """Sum of per-segment quadratic variation over unique_labels.

    A neighbour pair contributes only when both pixels carry the same label
    from unique_labels, so one masked difference replaces a per-segment stack.
    """
    data = jnp.asarray(data, dtype=jnp.float64)
    labeled_array = jnp.asarray(labeled_array, dtype=jnp.int32)
    in_set = jnp.isin(labeled_array, jnp.array(unique_labels, dtype=jnp.int32))

    def same_segment(lo, hi, ok_lo, ok_hi):
        return ((lo == hi) & ok_lo & ok_hi).astype(jnp.float64)

    dx = data[1:, :] - data[:-1, :]
    mx = same_segment(
        labeled_array[1:, :], labeled_array[:-1, :], in_set[1:, :], in_set[:-1, :]
    )
    dy = data[:, 1:] - data[:, :-1]
    my = same_segment(
        labeled_array[:, 1:], labeled_array[:, :-1], in_set[:, 1:], in_set[:, :-1]
    )
    return jnp.sum((dx * mx) ** 2) + jnp.sum((dy * my) ** 2)


def good_bad_boundary_mix_nm(x_nm, good_mask, bad_mask):
    """Penalty that explicitly couples the Stage-3 full pixel map across
    good/bad boundaries. This discourages the merged full-mirror pixel map
    from keeping a hard discontinuity between the bad segment and the rest
    of the pupil.
    """
    good_f = good_mask.astype(jnp.float64)
    bad_f = bad_mask.astype(jnp.float64)

    dx = x_nm[1:, :] - x_nm[:-1, :]
    bx = good_f[1:, :] * bad_f[:-1, :] + bad_f[1:, :] * good_f[:-1, :]

    dy = x_nm[:, 1:] - x_nm[:, :-1]
    by = good_f[:, 1:] * bad_f[:, :-1] + bad_f[:, 1:] * good_f[:, :-1]

    return jnp.sum((dx * bx) ** 2) + jnp.sum((dy * by) ** 2)


def piecewise_start_lr(lr, start_step):
    return lambda step: jnp.where(jnp.asarray(step) < start_step, 0.0, lr)


def make_sgd_with_schedule(lr, start_step, momentum=0.6):
    return optax.chain(
        optax.scale_by_schedule(piecewise_start_lr(lr, start_step)),
        optax.sgd(learning_rate=1.0, momentum=momentum, nesterov=True),
    )


def grad_global_norm(grads):
    return jnp.sqrt(
        sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads) if g is not None)
    )


def build_physical_ptt_basis(segment_labels):
    """Return union masks and [1, x/R, y/R] bases for all 18 mirrors.

    Coordinates are centred and normalised over the full physical mirror,
    including both disconnected components of a split segment.
    """
    labels = np.asarray(segment_labels)
    expected = {x for group in PHYSICAL_TO_LABELS.values() for x in group}
    actual = set(np.unique(labels)) - {0}
    if actual != expected:
        raise ValueError(
            f"PTT mapping expects CAMINO labels 1..22; missing={sorted(expected-actual)}, "
            f"unexpected={sorted(actual-expected)}. Check pupil labelling."
        )
    yy, xx = np.indices(labels.shape, dtype=float)
    masks, basis = [], []
    for group in PHYSICAL_TO_LABELS.values():
        mask = np.isin(labels, group)
        x, y = xx - xx[mask].mean(), yy - yy[mask].mean()
        radius = np.max(np.sqrt(x[mask] ** 2 + y[mask] ** 2))
        if radius <= 0:
            raise ValueError(f"Degenerate physical segment mask: {group}")
        masks.append(mask)
        basis.extend(
            [
                mask.astype(float),
                np.where(mask, x / radius, 0.0),
                np.where(mask, y / radius, 0.0),
            ]
        )
    return jnp.asarray(np.stack(masks)), jnp.asarray(np.stack(basis))


def _image_products(exp, rendered):
    data, err = np.asarray(exp.data), np.asarray(exp.err)
    model = np.asarray(rendered["model"])
    bad = (
        np.asarray(exp.bad, dtype=bool)
        | ~np.isfinite(data)
        | ~np.isfinite(err)
        | (err <= 0)
    )
    z = np.full_like(data, np.nan, dtype=float)
    np.divide(data - model, err, out=z, where=~bad)
    valid = ~bad & np.isfinite(z)
    return dict(
        data=data,
        err=err,
        bad=bad,
        model=model,
        resid=data - model,
        z=z,
        psf_unit=np.asarray(rendered["psf_unit"]),
        flux=float(rendered["flux"]),
        background=float(rendered["background"]),
        mean_z2=float(np.mean(z[valid] ** 2)) if valid.any() else np.nan,
    )


@dataclass
class FitProblem:
    """Loaded data and immutable model inputs; no notebook globals are used."""

    config: FitConfig
    paths: dict
    exposures: dict
    model: Any
    params_template: Any
    initial_params: dict
    segment_labels: Any
    mirror_mask: Any
    ptt_masks: Any = None
    ptt_basis: Any = None
    ptt_dual: Any = None  # see ptt_dual_basis
    plane_A: Any = None
    plane_pinv: Any = None
    plane_pinv_grid: Any = None  # plane_pinv on the pupil grid, zero off-mask
    segment_box_shape: tuple | None = None  # common per-mirror bounding box

    @property
    def observation_time(self):
        """UTC datetime of the earliest exposure (None if unknown)."""
        mjds = [e.mjd for e in self.exposures.values() if e.mjd is not None]
        if not mjds:
            return None
        from astropy.time import Time

        return Time(min(mjds), format="mjd", scale="utc").to_datetime()

    @property
    def physical_ids(self):
        return tuple(PHYSICAL_TO_LABELS) if self.config.fit_mode == "ptt_pixel" else ()

    @property
    def n_ptt(self):
        return 3 * len(self.physical_ids)

    @property
    def unique_labels(self):
        return tuple(int(x) for x in np.unique(self.segment_labels) if x > 0)

    def filter_context(self):
        return _local_filter(self.paths["filter"], self.config.filter_name)

    def ptt_project(self, pixel_nm):
        """Split a pixel map into (PTT coefficients, orthogonal remainder)."""
        coefficients = jnp.tensordot(self.ptt_dual, pixel_nm, axes=2)
        return coefficients, pixel_nm - self.ptt_map(coefficients)

    def ptt_map(self, coefficients):
        if self.ptt_basis is None:
            return jnp.zeros_like(self.mirror_mask, dtype=jnp.float64)
        # Preserve the source notebook's per-segment sum order.
        opd = jnp.zeros_like(self.mirror_mask, dtype=jnp.float64)
        for i, mask in enumerate(self.ptt_masks):
            opd = (
                opd
                + (
                    coefficients[3 * i]
                    + coefficients[3 * i + 1] * self.ptt_basis[3 * i + 1]
                    + coefficients[3 * i + 2] * self.ptt_basis[3 * i + 2]
                )
                * mask
            )
        return opd

    def with_ptt(self, base, coefficients, pixel_nm=None):
        if pixel_nm is None:
            pixel_nm = jnp.zeros_like(self.mirror_mask, dtype=jnp.float64)
        return {
            **base,
            "bad_plane_nm": coefficients,
            "full_pixel_opd_nm": pixel_nm,
            "aberrations_shared": (self.ptt_map(coefficients) + pixel_nm) * 1e-9,
        }

    def exposure_models(self, params):
        """Return {pupil: model with that exposure's parameters injected}."""
        full = build_full_params(params, self.params_template, self.exposures)
        opd, _ = center_opd_on_pupil(params["aberrations_shared"], self.mirror_mask)
        full = eqx.tree_at(lambda p: p.params["aberrations_shared"], full, opd)
        return {
            pup: cam.inject_into_model(
                self.model, cam.inject_views_for_pupil(full, pup, exp)
            )
            for pup, exp in self.exposures.items()
        }

    def segment_fields(self, params, index):
        """Cache the image field of every mirror but physical mirror `index`.

        Returns {pupil: SegmentField} for render(..., segments=...), valid
        while only that mirror's PTT coefficients change. Monochromatic only.
        """
        mask = np.asarray(self.ptt_masks[index])
        models = self.exposure_models(params)
        fields = {}
        for pup, exp in self.exposures.items():
            model = models[pup]
            source = exp.fit.make_source(model, exp)
            if np.size(source.wavelengths) != 1:
                raise NotImplementedError("Segment propagation needs n_wavels=1")
            optics = exp.fit.update_optics(model, exp)
            wavelength = source.wavelengths[0]
            u_in = optics.pupil_field(wavelength, source.position)
            x_in = optics.pupil_coords()
            post = np.flip(mask, 0) if optics.orientation == "pupil_flip" else mask
            rows, cols = self.segment_box_shape
            n = post.shape[0]
            r, c = np.flatnonzero(post.any(1)), np.flatnonzero(post.any(0))
            start = jnp.asarray([min(r[0], n - rows), min(c[0], n - cols)])
            outside = np.where(post, 0.0, np.abs(np.asarray(u_in)))
            ref = int(np.argmax(outside))
            field = SegmentField(
                u_rest=jnp.zeros(()),
                mask=jnp.asarray(post, dtype=jnp.float64),
                start=start,
                ref=jnp.asarray(ref),
                u_ref=u_in.ravel()[ref],
                shape=self.segment_box_shape,
            )
            u_full = optics.propagate_field(u_in, x_in, wavelength)
            u_seg = field.segment_image_field(optics, u_in, x_in, wavelength)
            fields[pup] = eqx.tree_at(lambda f: f.u_rest, field, u_full - u_seg)
        return fields

    def render(self, params, segments=None):
        """Render both images with the exact notebook flux/background solve.

        Internal JIT callers must be inside filter_context(). Public result
        methods manage this automatically. segments (from segment_fields)
        propagates only one mirror on top of a cached field of the others.
        """
        models = self.exposure_models(params)
        rendered = {}
        for pup, exp in self.exposures.items():
            model = models[pup]
            if segments is not None:
                model = eqx.tree_at(
                    lambda m: m.optics.segment,
                    model,
                    segments[pup],
                    is_leaf=lambda x: x is None,
                )
            psf = dlu.resize(exp.fit(model, exp), self.config.fit_npix)
            img_fit = jnp.where(exp.bad, psf, exp.data)
            err_fit = jnp.where(exp.bad, 1e20, exp.err)
            unit = psf / (jnp.sum(jnp.where(exp.bad, 0.0, psf)) + EPS)
            flux, background = cam.solve_flux_bg_weighted_jax_nansafe(
                img_fit, err_fit, exp.bad, unit
            )
            rendered[pup] = dict(
                psf_unit=unit,
                model=flux * unit + background,
                flux=flux,
                background=background,
                img_fit=img_fit,
                err_fit=err_fit,
            )
        return rendered

    def data_loss(self, params, segments=None):
        rendered = self.render(params, segments)
        total = 0.0
        for pup, exp in self.exposures.items():
            r = rendered[pup]
            total = total + jnp.nansum(
                -jnp.where(
                    exp.bad,
                    0.0,
                    jsp.stats.norm.logpdf(r["model"], r["img_fit"], r["err_fit"]),
                )
            )
        return total

    def regularisation(self, params):
        c = self.config
        if c.fit_mode == "pixel":
            opd, _ = center_opd_on_pupil(params["aberrations_shared"], self.mirror_mask)
            centred = opd * self.mirror_mask
            l1 = jnp.sum(jnp.abs(centred))
            qv = segmentwise_qv(self.segment_labels, centred * 1e9, self.unique_labels)
            return dict(opd_l1=c.lambda_l1 * l1, pixel_qv=c.lambda_qv * qv)
        pixel = params["full_pixel_opd_nm"] * self.mirror_mask
        qv = qv_masked_nm(pixel, self.mirror_mask)
        l2 = jnp.mean(pixel**2)  # Mean over the full square, as in the notebook.
        # plane_pinv @ pixel[mask] without a gather: its gradient would be a
        # scatter, which XLA:CPU runs as a per-pixel loop.
        ax, ay, _ = jnp.tensordot(self.plane_pinv_grid, pixel, axes=2)
        # Retain original uncentred coordinate convention for this penalty.
        slope = ax * self.plane_A[:, 0] + ay * self.plane_A[:, 1]
        # Fixed masks, combined in NumPy: inside jit XLA would constant-fold
        # an 18x512x512 reduction at every compile.
        bad = np.asarray(self.ptt_masks).any(axis=0)
        mix = good_bad_boundary_mix_nm(pixel, np.asarray(self.mirror_mask) & ~bad, bad)
        return dict(
            pixel_qv=c.lambda_qv * qv,
            pixel_l2=c.lambda_pixel_l2 * l2,
            global_plane=c.lambda_global_plane * jnp.mean(slope**2),
            boundary_mix=c.lambda_boundary_mix * mix,
        )

    def loss(self, params, regularise=True):
        total = self.data_loss(params)
        return (
            total + sum(self.regularisation(params).values()) if regularise else total
        )

    def display_mask(self):
        from scipy.ndimage import binary_erosion

        display = (
            np.asarray(self.model.optics.layers["pupil"].transmission)
            > self.config.display_transmission_threshold
        )
        if self.config.display_segment_outlines:
            labels = np.asarray(self.segment_labels)
            # Only use the physical mapping on its native-label geometry.
            groups = (
                PHYSICAL_TO_LABELS.values()
                if self.config.fit_mode == "ptt_pixel"
                else ((x,) for x in self.unique_labels)
            )
            for group in groups:
                mask = np.isin(labels, group)
                display &= ~(mask & ~binary_erosion(mask))
        return display

    @cached_property
    def position_scales(self):
        c = self.config
        scales = {
            "positions_wlp8_xy": c.position_scale,
            "positions_wlm8_xy": c.position_scale,
        }
        if c.bfgs_defocus:
            scales.update(
                defocus_wlp8_val=c.defocus_scale, defocus_wlm8_val=c.defocus_scale
            )
        return scales

    @cached_property
    def position_value_and_grad(self):
        """Jitted (x, params) -> loss and gradient for _run_positions.

        x holds the position (and defocus) entries of params divided by
        position_scales; params is an argument so it is not baked in.
        """
        scales = self.position_scales
        sizes = [int(np.size(self.initial_params[k])) for k in scales]

        def loss(x, params):
            parts = jnp.split(x, np.cumsum(sizes)[:-1])
            update = {k: part * scales[k] for k, part in zip(scales, parts)}
            return self.data_loss({**params, **update})

        return jax.jit(jax.value_and_grad(loss))


def load_data(
    wlp8_path,
    wlm8_path,
    *,
    pupil_path=None,
    filter_path=None,
    fit_mode=None,
    config=None,
):
    """Load one WLP8/WLM8 pair and initialise a ZERO OPD in both modes.

    By default, the bundled JWST pupil FITS and F212N throughput table are used.
    Custom pupil_path and filter_path values may still be supplied explicitly.
    """

    from importlib.resources import files

    if pupil_path is None:
        pupil_path = files("camino").joinpath("data", "jwst_pupil_flight_npix1024.fits")

    if filter_path is None:
        filter_path = files("camino").joinpath("data", "F212N.dat")

    c = _resolve_config(fit_mode, config)
    paths = {
        k: str(Path(v).expanduser().resolve())
        for k, v in dict(
            wlp8=wlp8_path, wlm8=wlm8_path, pupil=pupil_path, filter=filter_path
        ).items()
    }
    missing = [p for p in paths.values() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))
    jax.config.update("jax_enable_x64", True)
    with _local_filter(paths["filter"], c.filter_name):
        # FITS arrays are often big-endian; JAX requires native-endian dtypes.
        primary = jnp.asarray(
            np.asarray(fits.getdata(paths["pupil"]), dtype=np.float64)
        )
        if primary.ndim != 2 or primary.shape[0] != primary.shape[1]:
            raise ValueError("Pupil FITS must be a square 2-D array")
        if primary.shape[0] % c.pupil_downsample_factor:
            raise ValueError(
                "Pupil size must divide exactly by pupil_downsample_factor"
            )
        transmission = dlu.downsample(
            (primary >= 256.0).astype(jnp.int32), c.pupil_downsample_factor
        )
        optics = NIRCamFresnelOptics(
            transmission,
            defocus=3000.0,
            wf_npixels=transmission.shape[0],
            psf_npixels=c.psf_npixels,
            oversample=c.oversample,
            pixel_scale=c.pixel_scale,
            pixel_pitch=c.pixel_pitch,
            diameter=c.diameter,
            opd_phase_sign=c.opd_phase_sign,
            orientation=c.orientation,
        )
        detector = NRCDetectorLong(npixels_in=c.fit_npix, oversample=c.oversample)
        fit_poly = cam.SinglePointFilterFit(nwavels=c.n_wavels)
        exposures = {
            pup: cam.exposure_from_defocus_file(
                paths[pup.lower()], fit_poly, crop=c.fit_npix
            )
            for pup in ("WLP8", "WLM8")
        }
        for pup, exp in exposures.items():
            expected = (c.fit_npix, c.fit_npix)
            if any(tuple(x.shape) != expected for x in (exp.data, exp.err, exp.bad)):
                raise ValueError(f"{pup}: exposure crop does not match {expected}")
            if exp.filter != c.filter_name:
                raise ValueError(
                    f"{pup}: filter {exp.filter!r} differs from {c.filter_name!r}"
                )
        params, _ = init_params(exposures, optics, c.defocus_nm)
        template = cam.ModelParams(params)
        model = NIRCamModel(
            list(exposures.values()),
            params,
            optics,
            detector,
            {c.filter_name: cam.get_filter_test(paths["filter"])},
        )
        mask = model.optics.layers["pupil"].transmission > 0
        if not np.any(np.asarray(mask)):
            raise ValueError("Empty illuminated pupil")
        if c.fit_mode == "ptt_pixel":
            labels = build_segment_labels(paths["pupil"], c.pupil_downsample_factor)
        else:
            labels = measure.label(
                np.asarray(transmission).astype(bool), connectivity=2
            )
        if labels.shape != mask.shape:
            raise ValueError("Segment mask and optical pupil have different shapes")
        initial = dict(
            positions_wlp8_xy=jnp.zeros(2, dtype=jnp.float64),
            positions_wlm8_xy=jnp.zeros(2, dtype=jnp.float64),
            defocus_wlp8_val=jnp.asarray([c.defocus_nm[0]], dtype=jnp.float64),
            defocus_wlm8_val=jnp.asarray([c.defocus_nm[1]], dtype=jnp.float64),
            aberrations_shared=jnp.zeros_like(mask, dtype=jnp.float64),
            pupil_delta=jnp.zeros_like(mask, dtype=jnp.float64),
        )
        problem = FitProblem(
            c, paths, exposures, model, template, initial, jnp.asarray(labels), mask
        )
        if c.fit_mode == "ptt_pixel":
            problem.ptt_masks, problem.ptt_basis = build_physical_ptt_basis(labels)
            problem.ptt_dual = ptt_dual_basis(problem.ptt_basis, problem.ptt_masks)
            masks = np.asarray(problem.ptt_masks)
            problem.segment_box_shape = (
                int(max(np.ptp(np.flatnonzero(m.any(1))) + 1 for m in masks)),
                int(max(np.ptp(np.flatnonzero(m.any(0))) + 1 for m in masks)),
            )
            yy, xx = jnp.indices(labels.shape)
            problem.plane_A = jnp.stack(
                [xx[mask], yy[mask], jnp.ones(int(mask.sum()))], axis=1
            ).astype(jnp.float64)
            problem.plane_pinv = jnp.linalg.pinv(problem.plane_A)
            problem.plane_pinv_grid = (
                jnp.zeros((3, *mask.shape)).at[:, mask].set(problem.plane_pinv)
            )
        return problem


class _ScipyObjective:
    """Cache the last evaluation so accepted-iterate callbacks are accurate."""

    def __init__(self, value_and_grad, history):
        self.value_and_grad = value_and_grad
        self.history = history
        self.x = None
        self.value = None
        self.grad = None
        self.last_finite = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or not np.array_equal(x, self.x):
            value, grad = self.value_and_grad(jnp.asarray(x, dtype=jnp.float64))
            self.x, self.value, self.grad = (
                x.copy(),
                float(value),
                np.asarray(grad, dtype=np.float64),
            )
            if not np.isfinite(self.value) or not np.all(np.isfinite(self.grad)):
                if self.last_finite is None:
                    raise FloatingPointError(
                        f"Non-finite loss/gradient in {self.history.name}"
                    )
                # inf/NaN make L-BFGS-B stop; a finite penalty makes it back off.
                self.value = self.last_finite + 1e3 * (abs(self.last_finite) + 1)
                self.grad = np.zeros_like(x)
            else:
                self.last_finite = self.value
                self.history.eval_losses.append(self.value)
        return self.value, self.grad.copy()


def _run_optimizer(
    value_and_grad,
    x0,
    *,
    name,
    method,
    options,
    config,
    histories,
    bounds=None,
    record=None,
):
    history = StageHistory(name, method)
    histories.append(history)
    fun = _ScipyObjective(value_and_grad, history)
    start = time.perf_counter()
    history.initial_loss, _ = fun(x0)
    with tqdm(
        total=options["maxiter"], desc=name, unit="iter", disable=not config.progress
    ) as bar:

        def callback(x):
            value, grad = fun(x)
            history.losses.append(value)
            history.grad_norms.append(float(np.linalg.norm(grad)))
            history.max_abs_grads.append(float(np.max(np.abs(grad))))
            if record is not None:
                record(history, np.asarray(x))
            bar.set_postfix(
                loss=f"{value:.6e}", grad=f"{np.linalg.norm(grad):.3e}", refresh=False
            )
            bar.update(1)

        result = spo.minimize(
            fun,
            np.asarray(x0, dtype=np.float64),
            jac=True,
            method=method,
            bounds=bounds,
            callback=callback,
            options=options,
        )
        # maxiter is a budget; show a finished bar when the run stops early.
        bar.total = bar.n
        bar.refresh()
    history.result = result
    history.elapsed_seconds = time.perf_counter() - start
    print(f"{name}: {result.message}; iterations={result.nit}, loss={result.fun:.8e}")
    if config.show_plots:
        history.plot()
    return result


def _record_small(history, x):
    history.parameter_history.append(x.copy())


def _checkpoint(problem, params, histories, output_dir, name):
    """Save ordinary arrays after each completed stage (no Python pickles)."""
    if output_dir is None:
        return
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in params.items()}
    payload.update(
        physical_ids=np.asarray(problem.physical_ids, dtype=int),
        fit_mode=np.asarray(problem.config.fit_mode),
        config_json=np.asarray(json.dumps(asdict(problem.config))),
        paths_json=np.asarray(json.dumps(problem.paths)),
        mirror_mask=np.asarray(problem.mirror_mask),
    )
    np.savez_compressed(root / f"{name}_checkpoint.npz", **payload)
    _save_histories(histories, root / "stage_histories.npz")


def _save_histories(histories, path):
    arrays = {}
    summaries = []
    for i, h in enumerate(histories):
        prefix = f"stage_{i:02d}"
        for field_name in (
            "losses",
            "grad_norms",
            "max_abs_grads",
            "parameter_history",
            "pixel_rms_nm",
            "eval_losses",
        ):
            arrays[f"{prefix}_{field_name}"] = np.asarray(getattr(h, field_name))
        summary = dict(
            name=h.name,
            method=h.method,
            initial_loss=h.initial_loss,
            elapsed_seconds=h.elapsed_seconds,
        )
        if h.result is not None:
            summary.update(
                success=bool(h.result.success),
                message=str(h.result.message),
                nit=int(h.result.nit),
                nfev=int(h.result.nfev),
                fun=float(h.result.fun),
            )
        summaries.append(summary)
    arrays["summary_json"] = np.asarray(json.dumps(summaries))
    np.savez_compressed(path, **arrays)


def _run_pixel_initial_stage(problem, histories):
    c = problem.config
    params = dict(problem.initial_params)
    if c.stage1_sgd_steps == 0:
        return params
    labels = dict(
        positions_wlp8_xy="pos_wlp8",
        positions_wlm8_xy="pos_wlm8",
        defocus_wlp8_val="def_wlp8",
        defocus_wlm8_val="def_wlm8",
        aberrations_shared="freeze",
        pupil_delta="freeze",
    )
    names = ("def_wlp8", "pos_wlp8", "def_wlm8", "pos_wlm8")
    transforms = {
        name: make_sgd_with_schedule(lr, step, c.sgd_momentum)
        for name, lr, step in zip(names, c.sgd_learning_rates, c.sgd_start_steps)
    }
    transforms["freeze"] = optax.set_to_zero()
    optimizer = optax.multi_transform(transforms, labels)
    state = optimizer.init(params)
    # OPD is zero and frozen, so the source's stage-1 L1/shear terms are zero.
    evaluate = jax.value_and_grad(problem.data_loss)

    @jax.jit
    def step(params, state):
        value, grads = evaluate(params)
        updates, state = optimizer.update(grads, state, params=params)
        return (
            value,
            grad_global_norm(grads),
            optax.apply_updates(params, updates),
            state,
        )

    h = StageHistory("Stage 1 — SGD defocus and positions", "SGD")
    histories.append(h)
    started = time.perf_counter()
    with tqdm(total=c.stage1_sgd_steps, desc=h.name, disable=not c.progress) as bar:
        for _ in range(c.stage1_sgd_steps):
            value, grad_norm, new_params, state = step(params, state)
            h.losses.append(float(value))
            h.grad_norms.append(float(grad_norm))
            # These histories correspond to the pre-update loss, as in the source.
            h.parameter_history.append(
                np.concatenate(
                    [
                        np.asarray(params[k]).ravel()
                        for k in (
                            "positions_wlp8_xy",
                            "positions_wlm8_xy",
                            "defocus_wlp8_val",
                            "defocus_wlm8_val",
                        )
                    ]
                )
            )
            params = new_params
            bar.set_postfix(loss=f"{float(value):.6e}", refresh=False)
            bar.update(1)
    h.initial_loss = h.losses[0]
    h.elapsed_seconds = time.perf_counter() - started
    if c.show_plots:
        h.plot()
    return params


def _run_positions(problem, params, histories, maxiter, name):
    """BFGS fit of the image positions, and the defocus if config.bfgs_defocus."""
    if maxiter == 0:
        return params
    c = problem.config
    scales = problem.position_scales
    sizes = [int(np.size(params[k])) for k in scales]
    x0 = np.concatenate([np.ravel(params[k]) / v for k, v in scales.items()])

    def unpack(x):
        parts = jnp.split(x, np.cumsum(sizes)[:-1])
        return {
            **params,
            **{k: part * scales[k] for k, part in zip(scales, parts)},
        }

    def evaluate(x):
        return problem.position_value_and_grad(x, params)

    result = _run_optimizer(
        evaluate,
        x0,
        name=name,
        method="BFGS",
        options=dict(maxiter=maxiter, gtol=c.ptt_gtol),
        config=c,
        histories=histories,
        record=_record_small,
    )
    return unpack(jnp.asarray(result.x))


def _run_ptt_initial_stages(
    problem, histories, output_dir, from_params=None, mirrors=None, extra_seeds=None
):
    """Positions, per-mirror PTT seeds and fits, joint PTT, position refit.

    From zero by default. With from_params (a previous fit's params) the seed
    grid runs only for `mirrors` (physical ids), each also seeded from its
    current coefficients and any extra_seeds[id]; the pixel residual in
    from_params is kept throughout.
    """
    c = problem.config
    if from_params is None:
        from_params = problem.with_ptt(
            problem.initial_params, jnp.zeros(problem.n_ptt, dtype=jnp.float64)
        )
    extra_seeds = {} if extra_seeds is None else extra_seeds
    params = _run_positions(
        problem, from_params, histories, c.position_maxiter, "Stage 0 — BFGS positions"
    )
    _checkpoint(problem, params, histories, output_dir, "stage0")
    base = dict(params)
    residual = base["full_pixel_opd_nm"]
    coefficients = np.array(base["bad_plane_nm"], dtype=np.float64)
    warm = mirrors is not None

    def loss_ptt(x):
        return problem.data_loss(problem.with_ptt(base, x, residual))

    # One compiled gradient serves the joint fit (and the per-mirror fits
    # without segment propagation): in reverse mode all 54 partials cost the
    # same as the 3 a mirror needs.
    ptt_value_and_grad = jax.jit(jax.value_and_grad(loss_ptt))
    segmented = c.ptt_segment_propagation and c.n_wavels == 1

    def mirror_loss(q, fixed, start, fields):
        """Loss with mirror coefficients q at `start`; the rest from fixed."""
        x = jax.lax.dynamic_update_slice(fixed, q, (start,))
        if fields is None:
            return loss_ptt(x)
        return problem.data_loss(problem.with_ptt(base, x, residual), segments=fields)

    # start and fields are traced, so one compilation serves every mirror.
    seed_losses = jax.jit(jax.vmap(mirror_loss, in_axes=(0, None, None, None)))
    mirror_value_and_grad = jax.jit(jax.value_and_grad(mirror_loss))
    if not c.skip_grid_init:
        grid = np.asarray(
            list(product(*(np.unique([lo, 0.0, hi]) for lo, hi in c.ptt_bounds_nm)))
        )
        for i, physical_id in enumerate(problem.physical_ids):
            if warm and physical_id not in mirrors:
                continue
            start = 3 * i
            seeds = grid
            if warm:
                extra = [coefficients[start : start + 3]]
                extra += [
                    np.asarray(h, float) for h in extra_seeds.get(physical_id, [])
                ]
                seeds = np.concatenate(
                    [grid, np.clip(extra, *np.transpose(c.ptt_bounds_nm))]
                )
            batch = min(c.ptt_seed_batch, len(seeds))
            # Pad to whole batches so every call has the same shape.
            padded = np.concatenate(
                [seeds, np.repeat(seeds[-1:], -len(seeds) % batch, 0)]
            )
            fixed = jnp.asarray(coefficients)
            fields = (
                problem.segment_fields(problem.with_ptt(base, fixed, residual), i)
                if segmented
                else None
            )
            values = []
            with tqdm(
                total=len(seeds),
                desc=f"PTT seeds — physical {physical_id}",
                disable=not c.progress,
            ) as bar:
                for k in range(0, len(padded), batch):
                    chunk = jnp.asarray(padded[k : k + batch])
                    values.extend(np.asarray(seed_losses(chunk, fixed, start, fields)))
                    bar.update(min(batch, len(seeds) - k))
            values = np.asarray(values[: len(seeds)])
            if not np.any(np.isfinite(values)):
                raise FloatingPointError(
                    f"No finite PTT seed for physical segment {physical_id}"
                )
            # First minimum, as the original sequential strict-< search.
            best_seed = seeds[
                int(np.argmin(np.where(np.isfinite(values), values, np.inf)))
            ]

            def local_value_and_grad(q, start=start, fixed=fixed, fields=fields):
                return mirror_value_and_grad(jnp.asarray(q), fixed, start, fields)

            if c.local_ptt_maxiter:
                local = _run_optimizer(
                    local_value_and_grad,
                    best_seed,
                    name=f"Stage 1a — local PTT physical {physical_id}",
                    method="L-BFGS-B",
                    options=dict(
                        maxiter=c.local_ptt_maxiter, gtol=c.ptt_gtol, maxcor=10
                    ),
                    bounds=c.ptt_bounds_nm,
                    config=c,
                    histories=histories,
                    record=_record_small,
                )
                coefficients[start : start + 3] = local.x
            else:
                coefficients[start : start + 3] = best_seed
            _checkpoint(
                problem,
                problem.with_ptt(base, jnp.asarray(coefficients), residual),
                histories,
                output_dir,
                f"stage1a_physical_{physical_id:02d}",
            )
    if c.joint_ptt_maxiter:
        joint = _run_optimizer(
            ptt_value_and_grad,
            coefficients,
            name="Stage 1 — joint PTT BFGS",
            method="BFGS",
            options=dict(maxiter=c.joint_ptt_maxiter, gtol=c.ptt_gtol),
            config=c,
            histories=histories,
            record=_record_small,
        )
        coefficients = joint.x
    params = problem.with_ptt(base, jnp.asarray(coefficients), residual)
    _checkpoint(problem, params, histories, output_dir, "stage1")
    params = _run_positions(
        problem,
        params,
        histories,
        c.position_refit_maxiter,
        "Stage 1b — BFGS position refit",
    )
    _checkpoint(problem, params, histories, output_dir, "stage1b")
    return params


def make_mask_embed(mask):
    """Return embed(x): scatter a vector into the True pixels of a 2-D mask.

    Equivalent to zeros(mask.shape).at[mask].set(x), but written as a gather
    with a gather as its transpose: on CPU, XLA lowers scatter (and the JVP of
    a non-unique scatter) to a serial loop over every element.
    """
    mask = np.asarray(mask, dtype=bool)
    flat = np.flatnonzero(mask)
    # Index into [x, 0]: masked pixels read their entry, the rest read the 0.
    source = np.full(mask.size, flat.size)
    source[flat] = np.arange(flat.size)

    @jax.custom_vjp
    def embed(x):
        return jnp.append(x, jnp.zeros((), x.dtype))[source].reshape(mask.shape)

    def embed_bwd(_, g):
        return (g.ravel()[flat],)

    embed.defvjp(lambda x: (embed(x), None), embed_bwd)
    return embed


def _make_final_objective(problem, base_params):
    """Return the stage-2 vector, unpacker and compiled objective."""
    mask = problem.mirror_mask
    n_pixels = int(np.asarray(mask).sum())
    embed = make_mask_embed(mask)
    base = dict(base_params)
    if problem.config.fit_mode == "pixel":
        x0 = np.asarray(base["aberrations_shared"])[np.asarray(mask)] * 1e9

        def unpack(x):
            pixel = embed(jnp.asarray(x, dtype=jnp.float64))
            return {**base, "aberrations_shared": pixel * 1e-9}

    else:
        # A warm start carries its pixel residual; after the PTT stages it is zero.
        pixel = base.get("full_pixel_opd_nm")
        pixel = (
            np.zeros(n_pixels) if pixel is None else np.asarray(pixel)[np.asarray(mask)]
        )
        coefficients = np.asarray(base["bad_plane_nm"])
        orthogonal = problem.config.ptt_orthogonal_residual
        if orthogonal:
            # Move any plane already in the residual (e.g. from a warm start)
            # into the coefficients; the total OPD is unchanged.
            planes, rest = problem.ptt_project(embed(jnp.asarray(pixel)))
            coefficients = coefficients + np.asarray(planes)
            pixel = np.asarray(rest)[np.asarray(mask)]
        scale = problem.config.ptt_coefficient_scale
        x0 = np.concatenate([coefficients / scale, pixel])

        def unpack(x):
            pixel = embed(jnp.asarray(x[problem.n_ptt :], dtype=jnp.float64))
            if orthogonal:
                pixel = problem.ptt_project(pixel)[1]
            return problem.with_ptt(base, x[: problem.n_ptt] * scale, pixel)

    evaluate = jax.jit(jax.value_and_grad(lambda x: problem.loss(unpack(x))))
    return x0, unpack, evaluate


@dataclass
class FitResult:
    """Current solution, per-stage diagnostics and methods for notebook use.

    params['aberrations_shared'] is the raw total OPD in METRES; opd_nm is
    the same map in nm. centred_opd_nm removes global pupil piston for the
    renderer convention. Explicit PTT is in nm. No saved OPD is flipped to
    match the other mode's convention automatically.
    """

    problem: FitProblem = field(repr=False)
    params: dict
    scipy_result: Any
    histories: list
    output_dir: Path | None
    runtime_seconds: float
    _unpack: Callable = field(repr=False)
    _value_and_grad: Callable = field(repr=False)
    products: dict = field(default_factory=dict)
    objective_terms: dict = field(default_factory=dict)
    initialisation: str = "zero"

    @property
    def config(self):
        return self.problem.config

    @property
    def opd_nm(self):
        return np.asarray(self.params["aberrations_shared"]) * 1e9

    @property
    def centred_opd_nm(self):
        opd, _ = center_opd_on_pupil(jnp.asarray(self.opd_nm), self.problem.mirror_mask)
        return np.asarray(opd)

    @property
    def ptt_nm(self):
        return np.asarray(self.params.get("bad_plane_nm", np.empty(0))).reshape(-1, 3)

    @property
    def ptt_opd_nm(self):
        return np.asarray(self.problem.ptt_map(self.ptt_nm.ravel()))

    @property
    def pixel_opd_nm(self):
        return np.asarray(self.params.get("full_pixel_opd_nm", self.opd_nm))

    def refresh(self):
        """Always regenerate products from the latest optimiser vector."""
        self.params = self._unpack(jnp.asarray(self.scipy_result.x, dtype=jnp.float64))
        with self.problem.filter_context():
            rendered = self.problem.render(self.params)
            self.products = {
                pup: _image_products(exp, rendered[pup])
                for pup, exp in self.problem.exposures.items()
            }
            self.objective_terms = {
                k: float(v) for k, v in self.problem.regularisation(self.params).items()
            }
            self.objective_terms["data"] = float(self.problem.data_loss(self.params))
            self.objective_terms["total"] = sum(self.objective_terms.values())
        return self

    def continue_fit(self, maxiter=2000):
        return continue_fit(self, maxiter=maxiter)

    def plot_loss(self, last=None):
        import matplotlib.pyplot as plt

        stages = [h for h in self.histories if h.name.startswith("Stage 2")]
        values = np.asarray(
            [stages[0].initial_loss] + [v for h in stages for v in h.losses]
        )
        x = np.arange(values.size)
        if last is not None:
            if not isinstance(last, int) or last <= 0:
                raise ValueError("last must be a positive integer")
            x, values = x[-last:], values[-last:]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, values)
        ax.set(xlabel="L-BFGS-B iteration", ylabel="Loss", title="Stage 2 loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
        return fig

    def plot_opd(self, limit_nm=None):
        import matplotlib.pyplot as plt

        mask = self.problem.display_mask()
        maps = [("Total OPD", self.opd_nm)]
        if self.config.fit_mode == "ptt_pixel":
            maps += [
                ("Physical-segment PTT", self.ptt_opd_nm),
                ("Pixel residual", self.pixel_opd_nm),
            ]
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad("black")
        if limit_nm is None:
            limit_nm = (
                max(float(np.percentile(np.abs(self.opd_nm[mask]), 99.5)), 1e-9)
                if np.any(mask)
                else 1.0
            )
        fig, axes = plt.subplots(
            1,
            len(maps),
            figsize=(5 * len(maps), 4),
            squeeze=False,
            layout="constrained",
        )
        for ax, (title, array) in zip(axes[0], maps):
            im = ax.imshow(
                np.where(mask, array, np.nan),
                origin="lower",
                cmap=cmap,
                vmin=-limit_nm,
                vmax=limit_nm,
            )
            ax.set_title(title)
            ax.set_axis_off()
        fig.colorbar(im, ax=list(axes[0]), label="OPD (nm)")
        plt.show()
        return fig

    def compare_with_mast(self, mast_wss_path, **kwargs):
        return compare_with_mast(self, mast_wss_path, **kwargs)

    def save(self, output_dir=None):
        """Save the latest solution, stage histories, FITS products and config.

        Repeated saves replace these output files, including after continuation.
        restart_state.npz records the numerical state (not L-BFGS memory).
        """
        root = (
            Path(output_dir).expanduser() if output_dir is not None else self.output_dir
        )
        if root is None:
            raise ValueError("Supply output_dir to save the result")
        root.mkdir(parents=True, exist_ok=True)
        self.output_dir = root
        np.savez_compressed(
            root / "final_params.npz",
            **{k: np.asarray(v) for k, v in self.params.items()},
        )
        _save_histories(self.histories, root / "stage_histories.npz")
        np.savez_compressed(
            root / "restart_state.npz",
            x=np.asarray(self.scipy_result.x),
            fun=np.asarray(self.scipy_result.fun),
            jac=np.asarray(self.scipy_result.jac),
            **{k: np.asarray(v) for k, v in self.params.items()},
            mirror_mask=np.asarray(self.problem.mirror_mask),
            physical_ids=np.asarray(self.problem.physical_ids, dtype=int),
            segment_labels=np.asarray(self.problem.segment_labels),
            config_json=np.asarray(json.dumps(asdict(self.config))),
            paths_json=np.asarray(json.dumps(self.problem.paths)),
        )
        np.savez_compressed(
            root / "metrics.npz",
            **self.objective_terms,
            final_grad_norm=np.linalg.norm(self.scipy_result.jac),
            final_max_abs_grad=np.max(np.abs(self.scipy_result.jac)),
        )
        header = fits.Header()
        header["BUNIT"] = "nm"
        header["FITMODE"] = self.config.fit_mode
        for name, array in (
            ("final_opd_nm", self.opd_nm),
            ("final_opd_centred_nm", self.centred_opd_nm),
            ("ptt_opd_nm", self.ptt_opd_nm),
            ("pixel_opd_nm", self.pixel_opd_nm),
        ):
            fits.writeto(root / f"{name}.fits", array, header=header, overwrite=True)
        for pup, products in self.products.items():
            for key in ("data", "model", "resid", "z", "psf_unit", "err", "bad"):
                name = {"resid": "residual", "z": "zscore"}.get(key, key)
                array = (
                    products[key].astype(np.uint8) if key == "bad" else products[key]
                )
                fits.writeto(root / f"{name}_{pup.lower()}.fits", array, overwrite=True)
        config = dict(
            config=asdict(self.config),
            inputs=self.problem.paths,
            physical_ids=self.problem.physical_ids,
            runtime_seconds=self.runtime_seconds,
            initialisation=self.initialisation,
            backend=jax.default_backend(),
            objective_terms=self.objective_terms,
            optimizer=dict(
                success=bool(self.scipy_result.success),
                message=str(self.scipy_result.message),
                nit=int(self.scipy_result.nit),
                nfev=int(self.scipy_result.nfev),
            ),
            total_stage2_iterations=sum(
                h.result.nit
                for h in self.histories
                if h.name.startswith("Stage 2") and h.result is not None
            ),
        )
        (root / "run_config.json").write_text(json.dumps(config, indent=2))
        return root


def _record_final(problem):
    def record(h, x):
        h.pixel_rms_nm.append(float(np.std(x[problem.n_ptt :])))
        if problem.n_ptt:
            scale = problem.config.ptt_coefficient_scale
            h.parameter_history.append(x[: problem.n_ptt] * scale)

    return record


_WARM_START_KEYS = (
    "aberrations_shared",
    "positions_wlp8_xy",
    "positions_wlm8_xy",
    "defocus_wlp8_val",
    "defocus_wlm8_val",
    "pupil_delta",
)


def _initial_params_from(initial, problem):
    """Return (params, label) to warm-start problem from a previous fit.

    initial is a FitResult, a params dict, or a final_params.npz /
    restart_state.npz path written by FitResult.save. Keys it lacks keep
    problem's zero-start values. The detector window may differ from the
    previous fit: the OPD lives in the pupil plane and positions are in
    arcsec, so only the fit mode and pupil sampling must match.

    A pixel fit may also seed a ptt_pixel fit: its OPD is split into the 54
    PTT coefficients and the orthogonal remainder. When the previous fit's
    orientation is known and differs, y positions change sign (the image
    is otherwise identical under the two conventions).
    """
    c = problem.config
    orientation = None
    if isinstance(initial, FitResult):
        source, mode, label = initial.params, initial.config.fit_mode, "previous fit"
        orientation = initial.config.orientation
    elif isinstance(initial, dict):
        source, mode, label = initial, None, "parameter dict"
    else:
        path = Path(initial).expanduser()
        with np.load(path) as saved:
            source = {k: saved[k] for k in saved.files}
        saved_config = (
            json.loads(str(source["config_json"])) if "config_json" in source else {}
        )
        mode, orientation = saved_config.get("fit_mode"), saved_config.get(
            "orientation"
        )
        label = path.name
    if mode is None:
        # final_params.npz and plain dicts carry no config; PTT fits are the
        # ones with explicit plane coefficients.
        mode = "ptt_pixel" if "bad_plane_nm" in source else "pixel"
    from_pixel = mode == "pixel" and c.fit_mode == "ptt_pixel"
    if mode != c.fit_mode and not from_pixel:
        raise ValueError(f"Cannot warm-start a {c.fit_mode} fit from a {mode} fit")
    required = (
        ("bad_plane_nm",)
        if c.fit_mode == "ptt_pixel" and not from_pixel
        else ("aberrations_shared",)
    )
    missing = [k for k in required if k not in source]
    if missing:
        raise ValueError(f"Warm start is missing {missing}")

    params = dict(problem.initial_params)
    for key in _WARM_START_KEYS:
        if key not in source:
            continue
        value = jnp.asarray(np.asarray(source[key]), dtype=jnp.float64)
        if value.shape != jnp.shape(params[key]):
            raise ValueError(
                f"Warm-start {key} has shape {value.shape}; this problem needs "
                f"{jnp.shape(params[key])} (check pupil_downsample_factor)"
            )
        params[key] = value
    if orientation is not None and orientation != c.orientation:
        for key in ("positions_wlp8_xy", "positions_wlm8_xy"):
            params[key] = params[key] * jnp.asarray([1.0, -1.0])
    if from_pixel:
        coefficients, residual = problem.ptt_project(params["aberrations_shared"] * 1e9)
        return problem.with_ptt(params, coefficients, residual), f"{label} (pixel fit)"
    if c.fit_mode == "ptt_pixel":
        coefficients = jnp.asarray(np.asarray(source["bad_plane_nm"]), jnp.float64)
        if coefficients.shape != (problem.n_ptt,):
            raise ValueError(f"bad_plane_nm must have {problem.n_ptt} entries")
        pixel = source.get("full_pixel_opd_nm")
        if pixel is not None:
            pixel = jnp.asarray(np.asarray(pixel), jnp.float64)
            if pixel.shape != problem.mirror_mask.shape:
                raise ValueError("full_pixel_opd_nm does not match the pupil")
        params = problem.with_ptt(params, coefficients, pixel)
    return params, label


def diagnose_mirror_changes(problem, result, reference, threshold_nm=None):
    """Find the mirrors that moved between two fits (e.g. a segment tilt event).

    problem is a ptt_pixel FitProblem, used for its mirror segmentation;
    result and reference are fits of the same pupil, typically pixel-mode
    fits of an epoch and of the one before it. Each mirror's OPD change is
    projected onto its piston/tip/tilt, and the median change over mirrors
    is removed: a common tip/tilt shifts every segment's sub-image together,
    like an image translation, so fits trade it against the positions.
    Mirrors whose remaining tip/tilt change, hypot(tip, tilt) in nm at the
    mirror edge, exceeds threshold_nm (default 3x the median over mirrors)
    are flagged. Piston is reported but not used: it is weakly constrained
    by defocused images.

    Returns dict(table=[row per mirror], flagged=(ids...), seeds={id:
    [coefficients]}, threshold_nm), where seeds are result's own PTT
    coefficients for the flagged mirrors, for fit_data(..., extra_seeds=...).
    """
    if problem.config.fit_mode != "ptt_pixel":
        raise ValueError("diagnose_mirror_changes needs a ptt_pixel problem")
    mask = np.asarray(problem.mirror_mask)

    def ptt(opd):
        return np.asarray(
            problem.ptt_project(jnp.asarray(np.where(mask, opd, 0.0)))[0]
        ).reshape(-1, 3)

    result_ptt = ptt(np.asarray(result.opd_nm))
    change = result_ptt - ptt(np.asarray(reference.opd_nm))
    change -= np.median(change, axis=0)
    tip_tilt = np.hypot(change[:, 1], change[:, 2])
    if threshold_nm is None:
        threshold_nm = 3 * float(np.median(tip_tilt))
    table = [
        dict(
            mirror=pid,
            tip_tilt_change_nm=float(tt),
            piston_nm=float(c[0]),
            tip_nm=float(c[1]),
            tilt_nm=float(c[2]),
            flagged=bool(tt > threshold_nm),
        )
        for pid, tt, c in zip(problem.physical_ids, tip_tilt, change)
    ]
    flagged = tuple(row["mirror"] for row in table if row["flagged"])
    seeds = {
        pid: [result_ptt[i]]
        for i, pid in enumerate(problem.physical_ids)
        if pid in flagged
    }
    return dict(
        table=table, flagged=flagged, seeds=seeds, threshold_nm=float(threshold_nm)
    )


def fit_data(
    wlp8_path=None,
    wlm8_path=None,
    *,
    pupil_path=None,
    filter_path=None,
    fit_mode=None,
    config=None,
    data=None,
    output_dir=None,
    initial=None,
    refit_positions=True,
    reseed=None,
    extra_seeds=None,
):
    """Fit a pair from zero using 'pixel' (default) or 'ptt_pixel'.

    Supply the four paths OR a FitProblem returned by load_data. A preset
    can be customised with FitConfig.for_mode(...). Every BFGS invocation
    displays a loss curve unless config.show_plots=False. output_dir enables
    per-stage checkpoints and final saving; otherwise nothing is written.
    Final-stage maxiter is a budget, not an instruction to stop only upon
    convergence. Inspect result.scipy_result.message or continue the fit.

    initial warm-starts from a previous fit instead of zero: a FitResult,
    a params dict, or a saved final_params.npz / restart_state.npz. The
    initial stages are skipped; with refit_positions (default) a short BFGS
    fit of the four position parameters runs with the OPD held fixed,
    because pointing changes between epochs. Then only the final L-BFGS-B
    stage runs.

    For a ptt_pixel warm start, reseed (physical mirror ids, or "all") instead
    runs the full PTT stage sequence from initial, with the seed grid only for
    those mirrors, each also seeded from its current coefficients; extra_seeds
    maps ids to additional (piston, tip, tilt) seeds. With segment
    propagation, reseeding a mirror costs about a second.
    """
    if data is not None:
        if any(p is not None for p in (wlp8_path, wlm8_path, pupil_path, filter_path)):
            raise ValueError("Supply data or input paths, not both")
        if config is not None and config != data.config:
            raise ValueError("Load data again to change its configuration")
        _resolve_config(fit_mode, data.config)
        problem = data
    else:
        if any(p is None for p in (wlp8_path, wlm8_path, pupil_path, filter_path)):
            raise ValueError(
                "Supply WLP8, WLM8, pupil and filter paths, or data=load_data(...)"
            )
        problem = load_data(
            wlp8_path,
            wlm8_path,
            pupil_path=pupil_path,
            filter_path=filter_path,
            fit_mode=fit_mode,
            config=config,
        )
    c = problem.config
    root = Path(output_dir).expanduser() if output_dir is not None else None
    histories = []
    started = time.perf_counter()
    if initial is not None:
        params, label = _initial_params_from(initial, problem)
        start = f"warm start from {label}"
    else:
        start = "zero initial OPD"
    print(f"CAMINO {c.fit_mode}: {c.fit_npix}x{c.fit_npix} cutouts; {start}")
    with problem.filter_context():
        if initial is not None and reseed is not None:
            if c.fit_mode != "ptt_pixel":
                raise ValueError("reseed applies to ptt_pixel fits")
            params = _run_ptt_initial_stages(
                problem,
                histories,
                root,
                from_params=params,
                mirrors=set(problem.physical_ids if reseed == "all" else reseed),
                extra_seeds=extra_seeds,
            )
        elif initial is not None:
            if refit_positions:
                params = _run_positions(
                    problem,
                    params,
                    histories,
                    c.position_refit_maxiter,
                    "Warm start — BFGS position refit",
                )
            _checkpoint(problem, params, histories, root, "warm_start")
        elif c.fit_mode == "pixel":
            params = _run_pixel_initial_stage(problem, histories)
            _checkpoint(problem, params, histories, root, "stage1")
        else:
            params = _run_ptt_initial_stages(problem, histories, root)
        x0, unpack, evaluate = _make_final_objective(problem, params)
        optimizer = _run_optimizer(
            evaluate,
            x0,
            name="Stage 2 — L-BFGS-B",
            method="L-BFGS-B",
            options=c.stage2_options(),
            config=c,
            histories=histories,
            record=_record_final(problem),
        )
    result = FitResult(
        problem,
        unpack(jnp.asarray(optimizer.x)),
        optimizer,
        histories,
        root,
        time.perf_counter() - started,
        unpack,
        evaluate,
        initialisation="zero" if initial is None else "warm",
    )
    result.refresh()
    if root is not None:
        result.save()
    return result


def continue_fit(result, maxiter=2000):
    """Continue the final stage in memory; update this result and its files.

    Free parameters and objective are unchanged. The L-BFGS history is reset,
    exactly as with the notebook continuation cells. Returns the same result.
    """
    if not isinstance(maxiter, int) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer")
    started = time.perf_counter()
    with result.problem.filter_context():
        optimizer = _run_optimizer(
            result._value_and_grad,
            np.asarray(result.scipy_result.x).copy(),
            name="Stage 2 — L-BFGS-B continuation",
            method="L-BFGS-B",
            options=result.config.stage2_options(maxiter),
            config=result.config,
            histories=result.histories,
            record=_record_final(result.problem),
        )
    result.scipy_result = optimizer
    result.runtime_seconds += time.perf_counter() - started
    result.refresh()
    if result.output_dir is not None:
        result.save()
    return result


def get_mast_wss_path(date, choice="closest", verbose=True):
    """Resolve the official WSS product using the stpsf (formerly WebbPSF) API.

    Requires stpsf's reference data. Network access may be used by stpsf. A local WSS path can instead be passed directly to
    compare_with_mast, with no lookup required. verbose prints stpsf's OPD
    query summary; download messages (which show local paths) are hidden.
    The lookup is cached (see camino.data_utils.cached_lookup) and the file
    kept in stpsf's data directory, so repeat calls need no network.
    """
    import stpsf

    from .data_utils import cached_lookup, hide_download_messages

    def query():
        with hide_download_messages():
            filename = stpsf.mast_wss.get_opd_at_time(
                date, choice=choice, verbose=verbose
            )
        return Path(filename).name

    name = cached_lookup("wss_opd", f"{date}|{choice}", query)
    path = Path(stpsf.utils.get_stpsf_data_path()) / "MAST_JWST_WSS_OPDs" / name
    if not path.is_file():
        with hide_download_messages():
            stpsf.mast_wss.mast_retrieve_opd(name)
    return path


def compare_with_mast(
    result,
    mast_wss_path,
    *,
    calc_hdu=None,
    phase_scale_to_nm=1000.0,
    bright_fraction=0.15,
    cdf_xmax=25.0,
    hist_bins=120,
    output_dir=None,
    output_basename="model_comparison",
    show=True,
):
    """Compare final model/OPD with the official WSS calculated images.

    Preserves the PTT notebook's multi-panel figure for either fit mode.
    Default calculated-image HDUs: WLM8=6, WLP8=11. RESULT_PHASE is assumed
    to be micrometres (scale=1000 to nm), as in the source. Override these
    arguments for other products. No OPD registration or orientation change
    is performed; maps are displayed in their native conventions.

    Residual statistic is mean(z**2), NOT a degrees-of-freedom-corrected
    reduced chi-square. Returns the figure and both sets of image products.
    Saves PNG/PDF to output_dir, or the fit's output_dir when set.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not 0 < bright_fraction <= 1 or hist_bins < 2 or cdf_xmax <= 0:
        raise ValueError("Invalid comparison plot settings")
    if not np.isfinite(phase_scale_to_nm) or phase_scale_to_nm <= 0:
        raise ValueError("phase_scale_to_nm must be finite and positive")
    mast_wss_path = Path(mast_wss_path).expanduser()
    if not mast_wss_path.is_file():
        raise FileNotFoundError(mast_wss_path)
    MAST_CALC_HDU = {"WLM8": 6, "WLP8": 11} if calc_hdu is None else dict(calc_hdu)
    if set(MAST_CALC_HDU) != {"WLP8", "WLM8"}:
        raise ValueError("calc_hdu must specify WLP8 and WLM8")
    COMPARISON_PSF_CUTOUT_SIZE = result.config.fit_npix
    COMPARISON_BRIGHT_FRACTION = bright_fraction
    COMPARISON_CDF_XMAX = cdf_xmax
    COMPARISON_HIST_BINS = hist_bins
    COMPARISON_OUTPUT_BASENAME = output_basename
    comparison_output_dir = (
        Path(output_dir).expanduser() if output_dir is not None else result.output_dir
    )
    exposures = result.problem.exposures
    camino_model_images = {pup: result.products[pup]["model"] for pup in exposures}
    camino_opd_plot = np.where(result.problem.display_mask(), result.opd_nm, np.nan)
    with fits.open(mast_wss_path) as hdul:
        mast_opd_nm = hdul["RESULT_PHASE"].data.astype(float) * phase_scale_to_nm
        mast_pupil_mask = hdul["PUPIL_MASK"].data.astype(bool)
        mast_psf = {}
        for pup, hdu_index in MAST_CALC_HDU.items():
            mast_psf[pup], _ = cam.cutout_around_defocused_psf(
                img=hdul[hdu_index].data.astype(float), size=COMPARISON_PSF_CUTOUT_SIZE
            )
    mast_opd_nm = np.where(mast_pupil_mask, mast_opd_nm, np.nan)
    opd_combined = np.concatenate(
        [
            camino_opd_plot[np.isfinite(camino_opd_plot)].ravel(),
            mast_opd_nm[np.isfinite(mast_opd_nm)].ravel(),
        ]
    )
    comparison_limP = (
        max(float(np.nanpercentile(np.abs(opd_combined), 99.5)), 1e-9)
        if opd_combined.size
        else 1.0
    )

    def comparison_make_camino_products(exp, model):
        data = np.asarray(exp.data, dtype=float)
        err = np.asarray(exp.err, dtype=float)
        bad = np.asarray(exp.bad, dtype=bool)
        bad = bad | ~np.isfinite(data) | ~np.isfinite(err) | (err <= 0)
        model = np.asarray(model, dtype=float)
        if model.shape != data.shape:
            raise ValueError(
                f"CAMINO model/data shape mismatch: {model.shape} vs {data.shape}"
            )
        residual = data - model
        zscore = np.full_like(residual, np.nan, dtype=float)
        good = ~bad
        zscore[good] = residual[good] / err[good]
        good = good & np.isfinite(zscore)
        npix = int(np.sum(good))
        mean_z2 = float(np.sum(zscore[good] ** 2) / max(npix, 1))
        return {
            "data": data,
            "err": err,
            "bad": bad,
            "model": model,
            "resid": residual,
            "z": zscore,
            "rchisq": mean_z2,
            "npix": npix,
        }

    def comparison_make_mast_products(exp, mast_psf_img):
        data = np.asarray(exp.data, dtype=float)
        err = np.asarray(exp.err, dtype=float)
        bad = np.asarray(exp.bad, dtype=bool)
        bad = bad | ~np.isfinite(data) | ~np.isfinite(err) | (err <= 0)
        psf = np.asarray(mast_psf_img, dtype=float)
        if psf.shape != data.shape:
            raise ValueError(
                f"MAST PSF/data shape mismatch: {psf.shape} vs {data.shape}"
            )
        psf_unit = psf / (np.nansum(psf) + 1e-12)
        f_star, b_star = cam.solve_flux_bg_weighted_jax_nansafe(
            jnp.asarray(data, dtype=jnp.float64),
            jnp.asarray(err, dtype=jnp.float64),
            jnp.asarray(bad, dtype=bool),
            jnp.asarray(psf_unit, dtype=jnp.float64),
        )
        model = float(f_star) * psf_unit + float(b_star)
        residual = data - model
        zscore = np.full_like(residual, np.nan, dtype=float)
        good = ~bad
        zscore[good] = residual[good] / err[good]
        good = good & np.isfinite(zscore)
        npix = int(np.sum(good))
        mean_z2 = float(np.sum(zscore[good] ** 2) / max(npix, 1))
        return {
            "model": model,
            "resid": residual,
            "z": zscore,
            "rchisq": mean_z2,
            "npix": npix,
            "flux": float(f_star),
            "background": float(b_star),
        }

    comparison_out = {
        pup: comparison_make_camino_products(exposures[pup], camino_model_images[pup])
        for pup in ("WLP8", "WLM8")
    }
    comparison_out_mast = {
        pup: comparison_make_mast_products(exposures[pup], mast_psf[pup])
        for pup in ("WLP8", "WLM8")
    }
    for pup in ("WLP8", "WLM8"):
        intensity_values = np.concatenate(
            [
                img[np.isfinite(img)].ravel()
                for img in (
                    comparison_out[pup]["data"],
                    comparison_out[pup]["model"],
                    comparison_out_mast[pup]["model"],
                )
            ]
        )
        comparison_out[pup]["vminI"] = np.nanpercentile(intensity_values, 0.5)
        comparison_out[pup]["vmaxI"] = np.nanpercentile(intensity_values, 99.5)
        z_comb = np.concatenate(
            [
                comparison_out[pup]["z"][np.isfinite(comparison_out[pup]["z"])].ravel(),
                comparison_out_mast[pup]["z"][
                    np.isfinite(comparison_out_mast[pup]["z"])
                ].ravel(),
            ]
        )
        comparison_out[pup]["limZ"] = (
            np.nanpercentile(np.abs(z_comb), 99.5) if z_comb.size else 1.0
        )
        comparison_out[pup]["limZ"] = max(float(comparison_out[pup]["limZ"]), 1e-9)
        comparison_out[pup]["limH"] = comparison_out[pup]["limZ"]

    def comparison_collect_abs_z(bright_frac=None):
        all_model = []
        all_mast = []
        for pup in ("WLP8", "WLM8"):
            data = comparison_out[pup]["data"]
            z_model = comparison_out[pup]["z"]
            z_mast = comparison_out_mast[pup]["z"]
            err = comparison_out[pup]["err"]
            good = (
                np.isfinite(z_model)
                & np.isfinite(z_mast)
                & np.isfinite(err)
                & (err > 0)
                & np.isfinite(data)
            )
            if bright_frac is not None and np.any(good):
                threshold = np.percentile(data[good], 100 * (1 - bright_frac))
                good = good & (data >= threshold)
            all_model.append(np.abs(z_model[good]).ravel())
            all_mast.append(np.abs(z_mast[good]).ravel())
        return (np.concatenate(all_model), np.concatenate(all_mast))

    def make_camino_mast_comparison_figure():
        cmap_opd = plt.cm.RdBu_r.copy()
        cmap_opd.set_bad(color="black")
        fig = plt.figure(figsize=(13.8, 11.0), constrained_layout=False)
        gs = GridSpec(
            4,
            5,
            figure=fig,
            width_ratios=[1.0, 1.0, 1.0, 0.1, 1.3],
            wspace=0.35,
            hspace=0.35,
        )
        right_gs = gs[:, 4].subgridspec(
            5, 1, height_ratios=[1.2, 0.05, 1.2, 0.05, 0.9], hspace=0.08
        )

        def image_panel(
            ax,
            image,
            title,
            cmap="inferno",
            vmin=None,
            vmax=None,
            show_cbar_labels=True,
        ):
            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if not show_cbar_labels:
                cb.ax.set_yticks([])
                cb.ax.tick_params(
                    left=False, right=False, labelleft=False, labelright=False
                )
            ax.set_title(title, fontsize=11)
            ax.minorticks_on()
            tick_color = "white" if cmap == "inferno" else "black"
            ax.tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1.2,
                colors=tick_color,
                labelcolor=tick_color,
            )
            ax.tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                colors=tick_color,
            )
            if cmap == "RdBu_r":
                ax.tick_params(labelbottom=False, labelleft=False)
            return im

        def histogram_panel(
            ax, z_model, z_mast, title, xlim, rchisq_model, rchisq_mast
        ):
            z1 = z_model[np.isfinite(z_model)].ravel()
            z2 = z_mast[np.isfinite(z_mast)].ravel()
            xmin = -xlim
            xmax = xlim
            edges = np.linspace(xmin, xmax, COMPARISON_HIST_BINS + 1)
            h_model, _, _ = ax.hist(
                z1, bins=edges, density=True, alpha=0.6, label="Model"
            )
            h_mast, _, _ = ax.hist(
                z2, bins=edges, density=True, alpha=0.4, label="MAST"
            )
            ax.set_xlim(xmin, xmax)
            ymax = 1.1 * max(
                np.nanmax(h_model) if h_model.size else 0,
                np.nanmax(h_mast) if h_mast.size else 0,
            )
            ax.set_ylim(0.0005, ymax)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("z-score", labelpad=-2)
            ax.set_ylabel("Density", labelpad=2)
            ax.minorticks_on()
            ax.tick_params(
                which="major", direction="in", top=True, right=True, length=6, width=1.2
            )
            ax.tick_params(
                which="minor", direction="in", top=True, right=True, length=3, width=0.8
            )
            ax.grid(which="major", alpha=0.4, linewidth=0.8)
            ax.grid(which="minor", alpha=0.2, linewidth=0.5)
            ax.axvline(0, color="k", ls=":", lw=1)
            ax.text(
                0.94,
                0.97,
                rf"$\chi^2_\nu$ = {rchisq_model:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="tab:blue",
            )
            ax.text(
                0.94,
                0.86,
                rf"$\chi^2_\nu$ = {rchisq_mast:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="tab:orange",
            )
            legend = ax.legend(
                frameon=False,
                fontsize=9,
                loc="upper left",
                handlelength=1.0,
                handletextpad=0.5,
            )
            for txt in legend.get_texts():
                if txt.get_text().startswith("Model"):
                    txt.set_color("tab:blue")
                elif txt.get_text().startswith("MAST"):
                    txt.set_color("tab:orange")
            ax.set_box_aspect(1)

        def opd_panel(ax, image, title, label, label_color):
            im = ax.imshow(
                image,
                origin="lower",
                cmap=cmap_opd,
                vmin=-comparison_limP,
                vmax=comparison_limP,
            )
            ax.set_title(title, fontsize=11)
            ax.minorticks_on()
            ax.tick_params(
                which="major",
                direction="in",
                top=True,
                right=True,
                length=6,
                width=1.2,
                colors="white",
                labelbottom=False,
                labelleft=False,
            )
            ax.tick_params(
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3,
                width=0.8,
                colors="white",
            )
            ax.text(
                0.04,
                0.98,
                f"{image.shape[0]}×{image.shape[1]}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=9,
                weight="bold",
            )
            ax.text(
                0.04,
                0.035,
                label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                color=label_color,
                fontsize=9,
                weight="bold",
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.04)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.set_title("nm", fontsize=8)
            cb.ax.tick_params(labelsize=8)

        row0 = {"WLP8": 0, "WLM8": 2}
        z_axes = []
        for pup in ("WLP8", "WLM8"):
            row = row0[pup]
            prod = comparison_out[pup]
            mast = comparison_out_mast[pup]
            image_panel(
                fig.add_subplot(gs[row, 0]),
                prod["data"],
                f"{pup}: Data",
                "inferno",
                prod["vminI"],
                prod["vmaxI"],
                show_cbar_labels=True,
            )
            image_panel(
                fig.add_subplot(gs[row, 1]),
                prod["model"],
                f"{pup}: Model",
                "inferno",
                prod["vminI"],
                prod["vmaxI"],
                show_cbar_labels=True,
            )
            ax_z_model = fig.add_subplot(gs[row, 2])
            z_axes.append(ax_z_model)
            image_panel(
                ax_z_model,
                prod["z"],
                f"{pup}: Model z-score",
                "RdBu_r",
                -prod["limZ"],
                prod["limZ"],
            )
            histogram_panel(
                fig.add_subplot(gs[row + 1, 0]),
                prod["z"],
                mast["z"],
                f"{pup}: Normalised residuals",
                prod["limH"],
                prod["rchisq"],
                mast["rchisq"],
            )
            image_panel(
                fig.add_subplot(gs[row + 1, 1]),
                mast["model"],
                f"{pup}: MAST",
                "inferno",
                prod["vminI"],
                prod["vmaxI"],
                show_cbar_labels=True,
            )
            ax_z_mast = fig.add_subplot(gs[row + 1, 2])
            z_axes.append(ax_z_mast)
            image_panel(
                ax_z_mast,
                mast["z"],
                f"{pup}: MAST z-score",
                "RdBu_r",
                -prod["limZ"],
                prod["limZ"],
            )
        ax_camino_opd = fig.add_subplot(right_gs[0, 0])
        opd_panel(ax_camino_opd, camino_opd_plot, "Model OPD", "Model", "#4fc3ff")
        ax_mast_opd = fig.add_subplot(right_gs[2, 0])
        opd_panel(ax_mast_opd, mast_opd_nm, "MAST OPD", "MAST", "#ffb347")
        ax_cdf = fig.add_subplot(right_gs[4, 0])
        ax_cdf.set_box_aspect(1)
        absz_model, absz_mast = comparison_collect_abs_z()
        absz_model_bright, absz_mast_bright = comparison_collect_abs_z(
            bright_frac=COMPARISON_BRIGHT_FRACTION
        )

        def plot_cdf(values, label, **kwargs):
            x = np.sort(values)
            y = np.arange(1, x.size + 1) / x.size
            ax_cdf.plot(x, y, label=label, **kwargs)

        plot_cdf(absz_model, "Model, all pixels", lw=2, color="#1f77b4", alpha=0.4)
        plot_cdf(absz_mast, "MAST, all pixels", lw=2, color="#ff7f0e", alpha=0.4)
        plot_cdf(
            absz_model_bright,
            f"Model, brightest {100 * COMPARISON_BRIGHT_FRACTION:.0f}%",
            lw=2,
            ls="--",
            color="#1f77b4",
            alpha=0.9,
        )
        plot_cdf(
            absz_mast_bright,
            f"MAST, brightest {100 * COMPARISON_BRIGHT_FRACTION:.0f}%",
            lw=2,
            ls="--",
            color="#ff7f0e",
            alpha=0.9,
        )
        ax_cdf.set_title("Cumulative $|z|$", fontsize=11)
        ax_cdf.set_xlabel("$|z|$", fontsize=11)
        ax_cdf.set_ylabel("Cumulative fraction", fontsize=11)
        ax_cdf.set_xlim(0, COMPARISON_CDF_XMAX)
        ax_cdf.set_ylim(0, 1)
        ax_cdf.minorticks_on()
        ax_cdf.tick_params(
            which="major", direction="in", top=True, right=True, length=6, width=1.2
        )
        ax_cdf.tick_params(
            which="minor", direction="in", top=True, right=True, length=3, width=0.8
        )
        ax_cdf.grid(which="major", alpha=0.4, linewidth=0.8)
        ax_cdf.grid(which="minor", alpha=0.2, linewidth=0.5)
        ax_cdf.legend(frameon=False, fontsize=8, loc="lower right")
        fig.canvas.draw()
        opd_left = ax_camino_opd.get_position().x0
        x_sep = opd_left - 0.05
        y_bottom = min((ax.get_position().y0 for ax in z_axes))
        y_top = max((ax.get_position().y1 for ax in z_axes))
        fig.add_artist(
            Line2D(
                [x_sep, x_sep],
                [y_bottom, y_top],
                transform=fig.transFigure,
                color="k",
                lw=1.2,
                alpha=1,
                zorder=100,
            )
        )
        if comparison_output_dir is not None:
            comparison_output_dir.mkdir(parents=True, exist_ok=True)
            output_base = comparison_output_dir / COMPARISON_OUTPUT_BASENAME
            fig.savefig(str(output_base) + ".pdf", bbox_inches="tight", dpi=300)
            fig.savefig(str(output_base) + ".png", bbox_inches="tight", dpi=300)
        return fig

    figure = make_camino_mast_comparison_figure()
    for pup in exposures:
        print(
            f"{pup}: CAMINO mean(z²)={comparison_out[pup]['rchisq']:.3f}; "
            f"MAST mean(z²)={comparison_out_mast[pup]['rchisq']:.3f}"
        )
    if show:
        plt.show()
    return dict(figure=figure, camino=comparison_out, mast=comparison_out_mast)

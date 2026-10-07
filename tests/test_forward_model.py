"""End-to-end forward model: optics, detector and exposure on a tiny grid."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import camino as cam  # noqa: E402
from camino.fitting import (  # noqa: E402
    NIRCamFresnelOptics,
    NIRCamModel,
    NRCDetectorLong,
    init_params,
)

NPIX = 16
PSF_NPIX = 8


def _model_and_exposure():
    yy, xx = np.indices((NPIX, NPIX))
    pupil = jnp.asarray(((xx - 7.5) ** 2 + (yy - 7.5) ** 2 < 7**2).astype(float))
    optics = NIRCamFresnelOptics(
        pupil, defocus=4.4e7, psf_npixels=PSF_NPIX, oversample=4, wf_npixels=NPIX
    )
    detector = NRCDetectorLong(npixels_in=PSF_NPIX, oversample=4)
    fit = cam.SinglePointFilterFit(nwavels=1)
    data = jnp.ones((PSF_NPIX, PSF_NPIX))
    exp = cam.NIRCamExposure(
        "tiny_cal", "tiny", "F212N", data, 60000.0, data, fit, jnp.zeros_like(data)
    )
    params, _ = init_params({"WLP8": exp}, optics, [0.0, 0.0])
    filter_table = jnp.column_stack([jnp.linspace(2.1e11, 2.14e11, 5), jnp.ones(5)])
    model = NIRCamModel([exp], params, optics, detector, {"F212N": filter_table})
    return model, exp


def test_forward_model_is_finite_with_detector_shape():
    model, exp = _model_and_exposure()

    out = exp.fit(model, exp)

    assert out.shape == (PSF_NPIX, PSF_NPIX)
    assert jnp.isfinite(out).all()
    assert out.sum() > 0

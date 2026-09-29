"""Segment-only propagation and the direct-PSF propagate override."""

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from camino.fitting import NIRCamFresnelOptics, SegmentField  # noqa: E402

N = 32
WAVELENGTH = 2.12e-6
OFFSET = jnp.asarray([3e-7, -2e-7])


def _pupil():
    yy, xx = np.indices((N, N))
    pupil = ((xx - 15.5) ** 2 + (yy - 15.5) ** 2 < 15**2).astype(float)
    segment = pupil * (xx > 18) * (yy > 10) * (yy < 24)
    return jnp.asarray(pupil), np.asarray(segment, dtype=bool)


def _optics(pupil, opd=None, orientation="pupil_flip"):
    return NIRCamFresnelOptics(
        pupil,
        defocus=4.4e7,
        opd=opd,
        psf_npixels=8,
        oversample=2,
        wf_npixels=N,
        orientation=orientation,
    )


def _field(optics, segment, box_shape=(16, 16)):
    """SegmentField for `segment`, built as FitProblem.segment_fields does."""
    post = np.flip(segment, 0) if optics.orientation == "pupil_flip" else segment
    u_in = optics.pupil_field(WAVELENGTH, OFFSET)
    x_in = optics.pupil_coords()
    r, c = np.flatnonzero(post.any(1)), np.flatnonzero(post.any(0))
    start = [min(r[0], N - box_shape[0]), min(c[0], N - box_shape[1])]
    ref = int(np.argmax(np.where(post, 0.0, np.abs(np.asarray(u_in)))))
    field = SegmentField(
        u_rest=jnp.zeros(()),
        mask=jnp.asarray(post, dtype=float),
        start=jnp.asarray(start),
        ref=jnp.asarray(ref),
        u_ref=u_in.ravel()[ref],
        shape=box_shape,
    )
    full = optics.propagate_field(u_in, x_in, WAVELENGTH)
    seg = field.segment_image_field(optics, u_in, x_in, WAVELENGTH)
    return eqx.tree_at(lambda f: f.u_rest, field, full - seg)


def test_cropped_transform_equals_full_transform_of_masked_field():
    pupil, segment = _pupil()
    optics = _optics(pupil)
    field = _field(optics, segment)
    u_in = optics.pupil_field(WAVELENGTH, OFFSET)
    x_in = optics.pupil_coords()

    cropped = field.segment_image_field(optics, u_in, x_in, WAVELENGTH)
    full = optics.propagate_field(u_in * field.mask, x_in, WAVELENGTH)

    np.testing.assert_allclose(cropped, full, rtol=0, atol=1e-12 * np.abs(full).max())


@pytest.mark.parametrize("orientation", ["pupil_flip", "output_flip"])
def test_cached_rest_reproduces_psf_after_segment_and_piston_change(orientation):
    pupil, segment = _pupil()
    rng = np.random.default_rng(0)
    opd0 = jnp.asarray(rng.normal(0, 5e-8, (N, N))) * pupil
    field = _field(_optics(pupil, opd0, orientation), segment)

    # Move only the segment, plus a global piston (as piston removal does).
    tilt = 3e-7 * np.indices((N, N))[1] / N
    opd1 = opd0 + jnp.asarray(segment * (2e-7 + tilt)) + 4e-8
    optics = _optics(pupil, opd1, orientation)
    expected = optics.propagate(WAVELENGTH, OFFSET)
    actual = eqx.tree_at(
        lambda o: o.segment, optics, field, is_leaf=lambda x: x is None
    ).propagate(WAVELENGTH, OFFSET)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)


def test_direct_psf_matches_dlux_wavefront_path():
    pupil, _ = _pupil()
    optics = _optics(
        pupil, jnp.asarray(np.random.default_rng(1).normal(0, 5e-8, (N, N)))
    )
    wavelengths = jnp.asarray([2.10e-6, 2.14e-6])
    weights = jnp.asarray([0.3, 0.7])

    direct = optics.propagate(wavelengths, OFFSET, weights, return_psf=True)
    wavefronts = optics.propagate(wavelengths, OFFSET, weights, return_wf=True)

    np.testing.assert_allclose(direct.data, wavefronts.psf.sum(0), rtol=1e-12)
    np.testing.assert_allclose(direct.pixel_scale, wavefronts.pixel_scale.mean())


def test_propagate_rejects_both_return_flags():
    pupil, _ = _pupil()

    with pytest.raises(ValueError, match="cannot both be True"):
        _optics(pupil).propagate(WAVELENGTH, OFFSET, return_wf=True, return_psf=True)

import jax
import jax.numpy as jnp
import pytest

import dLux as dl

import camino


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_pixel_anisotropy_keeps_jax_array(dtype):
    psf = dl.PSF(jnp.ones((8, 8), dtype=dtype), pixel_scale=1.0)
    out = camino.PixelAnisotropy(order=1).apply(psf)
    assert isinstance(out.data, jax.Array)
    assert out.data.shape == psf.data.shape
    assert out.data.dtype == jnp.float64


def test_jacfwd_returns_jax_array():
    model = camino.ModelParams({"x": jnp.array([1.0, 2.0], dtype=jnp.float64)})
    jac = model.jacfwd(lambda m: jnp.sum(m.x**2))
    assert isinstance(jac, jax.Array)
    assert jac.dtype == jnp.float64
    assert jac.shape == (2,)


def test_public_api_excludes_legacy_transfer_helpers():
    assert hasattr(camino, "__all__")
    assert "transfer_fn" in camino.__all__
    assert "transfer_fn_old" not in camino.__all__
    assert "transfer_fn_patched" not in camino.__all__


def test_default_float64_contract_for_core_helpers():
    img = jnp.arange(9.0, dtype=jnp.float64).reshape(3, 3)

    out_shear = camino.apply_shear(img, 0.1)
    out_pupil = camino.apply_pupil_shear(img, 0.1)
    out_radial = camino.radial_zoom(img, 1.25)
    out_blur = camino.gaussian_blur_fft(img, 1.0)

    for out in (out_shear, out_pupil, out_radial, out_blur):
        assert isinstance(out, jax.Array)
        assert out.shape == img.shape
        assert out.dtype == jnp.float64
        assert jnp.isfinite(out).all()


def test_map_coordinates_2d_is_identity_on_grid_points():
    img = jnp.arange(9.0, dtype=jnp.float64).reshape(3, 3)
    coords = jnp.indices((3, 3))

    out = camino.map_coordinates_2d(img, coords, order=1, mode="constant", cval=0.0)

    assert isinstance(out, jax.Array)
    assert out.shape == img.shape
    assert out.dtype == jnp.float64
    assert jnp.allclose(out, img)


def test_transfer_and_rotate_keep_expected_shapes():
    coords = jnp.zeros((2, 2, 2), dtype=jnp.float64)
    transfer = camino.transfer_fn(coords, 10, 1.0, 1.0, 1.0)
    rotated = camino.Rotate(rotation_deg=90.0).apply(
        jnp.arange(9.0, dtype=jnp.float64).reshape(3, 3)
    )

    assert transfer.shape == (2, 2)
    assert transfer.dtype == jnp.complex128
    assert jnp.isfinite(transfer.real).all()
    assert jnp.isfinite(transfer.imag).all()
    assert rotated.shape == (3, 3)
    assert rotated.dtype == jnp.float64
    assert jnp.isfinite(rotated).all()


def test_set_array_promotes_leaf_values_to_float64():
    tree = {"a": jnp.array([1.0], dtype=jnp.float32), "b": 2.0, "c": [3.0]}
    out = camino.set_array(tree)

    assert isinstance(out["a"], jax.Array)
    assert out["a"].dtype == jnp.float64
    assert out["b"].dtype == jnp.float64


def test_jwst_primary_normalises_with_jax_norm():
    wavefront = type("Wavefront", (), {})()
    wavefront.amplitude = jnp.array([3.0, 4.0], dtype=jnp.float64)
    wavefront.phase = jnp.zeros_like(wavefront.amplitude)
    wavefront.wavenumber = jnp.array(1.0, dtype=jnp.float64)
    wavefront.set = lambda keys, values: {"amplitude": values[0], "phase": values[1]}

    optic = camino.JWSTPrimary(
        transmission=jnp.ones_like(wavefront.amplitude),
        opd=jnp.zeros_like(wavefront.amplitude),
    )
    out = optic.apply(wavefront)

    assert isinstance(out["amplitude"], jax.Array)
    assert out["amplitude"].dtype == jnp.float64
    assert jnp.isclose(jnp.linalg.norm(out["amplitude"]), 1.0)


def test_map_coordinates_2d_cubic_reproduces_quadratic_in_interior():
    rows, cols = jnp.indices((8, 8), dtype=jnp.float64)
    img = rows**2 + 0.5 * cols
    coords = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(2.0, 5.0, 7), jnp.linspace(2.0, 5.0, 7), indexing="ij"
        )
    )

    out = camino.map_coordinates_2d(img, coords, order=3, mode="constant", cval=0.0)

    expected = coords[0] ** 2 + 0.5 * coords[1]
    assert out.dtype == jnp.float64
    assert jnp.allclose(out, expected)

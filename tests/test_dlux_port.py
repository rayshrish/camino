"""Layers behave correctly against the phasor-based dLux (>=0.15.1) API."""

import jax

jax.config.update("jax_enable_x64", True)

import dLux as dl  # noqa: E402
import dLux.utils as dlu  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import camino  # noqa: E402

WAVELENGTH = 2e-6
N = 8


def test_jwst_primary_has_unit_power_and_expected_phasor():
    wf = dl.Wavefront(WAVELENGTH, N, diameter=1.0)
    transmission = jnp.zeros((N, N)).at[2:6, 2:6].set(1.0)
    opd = 1e-7 * jnp.arange(N * N, dtype=jnp.float64).reshape(N, N)

    out = camino.JWSTPrimary(transmission, opd=opd)(wf)

    amplitude = transmission / jnp.linalg.norm(transmission)
    expected = amplitude * jnp.exp(1j * 2 * jnp.pi / WAVELENGTH * opd)
    assert jnp.isclose(out.power, 1.0)
    assert jnp.allclose(out.phasor, expected)


def test_apply_sensitivities_instantiates_and_runs():
    FF = jnp.full((N, N), 2.0)
    SRF = jnp.full((2, 2), 3.0)
    psf = dl.PSF(jnp.ones((2 * N, 2 * N)), pixel_scale=1.0)

    out = camino.ApplySensitivities(FF, SRF)(psf)

    assert out.data.shape == (2 * N, 2 * N)
    assert jnp.allclose(out.data, 6.0)


def test_apply_pupil_curvature_matches_analytic_phase():
    wf = dl.Wavefront(WAVELENGTH, N, diameter=1.0)
    R = 50.0

    out = camino.apply_pupil_curvature(wf, R)

    r2 = (dlu.pixel_coords(N, 1.0) ** 2).sum(0)
    expected = wf.phasor * jnp.exp(1j * jnp.pi * r2 / (WAVELENGTH * R))
    assert jnp.allclose(out.phasor, expected)


def test_plane_to_plane_zero_distance_is_identity():
    wf = dl.Wavefront(WAVELENGTH, N, diameter=1.0).add_phase(
        jnp.linspace(0.0, 1.0, N * N).reshape(N, N)
    )

    out = camino.plane_to_plane(wf, 0.0)

    assert jnp.allclose(out.phasor, wf.phasor)

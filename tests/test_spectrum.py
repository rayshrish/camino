"""Spectrum weights shared by the forward model and the diagnostics."""

import types

import dLux.utils as dlu
import jax.numpy as jnp
import numpy as np
import pytest

import camino


class _Fit(camino.SinglePointFilterFit):
    def map_param(self, exposure, param):
        return param

    def get_key(self, exposure, param):
        return "obs"


class _Exposure:
    filter = "F212N"

    def __init__(self, nwavels):
        self.fit = _Fit(nwavels)


class _Model:
    def __init__(self, coeffs, pixel_scale=0.031):
        self.params = {"spectrum": {"obs": jnp.asarray(coeffs)}}
        self.optics = types.SimpleNamespace(psf_pixel_scale=pixel_scale)

    def get(self, key):
        return {
            "fluxes": jnp.asarray(1.0),
            "positions": jnp.asarray([1.0, -2.0]),
            "spectrum": self.params["spectrum"]["obs"],
        }[key]


@pytest.mark.parametrize("coeffs", [[0.0], [0.0, 2.0], [0.1, -1.5, 8.0]])
def test_weights_used_by_fit_match_make_source(coeffs):
    nw = 6
    exposure = _Exposure(nw)
    model = _Model(coeffs)
    params = types.SimpleNamespace(params=model.params)

    wv, weights = camino.weights_used_by_fit(params, exposure, nw=nw)
    source = exposure.fit.make_source(model, exposure)

    np.testing.assert_allclose(np.asarray(source.spectrum.wavelengths), wv)
    np.testing.assert_allclose(np.asarray(source.spectrum.weights), weights)
    assert weights.sum() == pytest.approx(1.0)


def test_flat_spectrum_reproduces_the_filter_weights():
    wv, filt = camino.calc_throughput("F212N", nwavels=5)

    np.testing.assert_allclose(
        np.asarray(camino.spectrum_weights(wv, filt, jnp.zeros(1))), np.asarray(filt)
    )


def test_single_wavelength_has_unit_weight():
    wv, filt = camino.calc_throughput("F212N", nwavels=1)

    assert float(camino.spectrum_weights(wv, filt, jnp.array([0.3, 5.0]))[0]) == 1.0


@pytest.mark.parametrize("pixel_scale", [0.031, 0.063])
def test_make_source_position_uses_the_optics_pixel_scale(pixel_scale):
    exposure = _Exposure(2)
    model = _Model([0.0], pixel_scale)

    source = exposure.fit.make_source(model, exposure)

    np.testing.assert_allclose(
        np.asarray(source.position),
        np.array([1.0, -2.0]) * dlu.arcsec2rad(pixel_scale),
    )

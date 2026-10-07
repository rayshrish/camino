"""Detector layer defaults and the convergence/spectrum diagnostics."""

import sys
import types

import jax.numpy as jnp
import numpy as np
import pytest

import dLux as dl

import camino


class _StubPyplot(types.ModuleType):
    """Records pyplot calls so the plotting path runs without matplotlib installed."""

    def __init__(self):
        super().__init__("matplotlib.pyplot")
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append(name)

        return _record


class _SpectrumFit:
    def map_param(self, exposure, param):
        return f"{param}.{exposure.filter}"


class _SpectrumExposure:
    filter = "F212N"
    fit = _SpectrumFit()


class _CoeffModel:
    def __init__(self, coeffs):
        self._coeffs = jnp.asarray(coeffs)

    def get(self, key):
        return self._coeffs


def _fake_throughput(filt, nwavels=1, filters_dir=None):
    return jnp.linspace(2.10e-6, 2.14e-6, nwavels), jnp.ones(nwavels) / nwavels


def test_pixel_anisotropy_default_order_applies():
    psf = dl.PSF(jnp.ones((8, 8), dtype=jnp.float64), pixel_scale=1.0)

    out = camino.PixelAnisotropy().apply(psf)

    assert out.data.shape == (8, 8)
    assert bool(jnp.isfinite(out.data).all())


@pytest.mark.parametrize("order", [2, 4, -1])
def test_pixel_anisotropy_rejects_unsupported_order(order):
    with pytest.raises(ValueError, match="order 0"):
        camino.PixelAnisotropy(order=order)


def test_pixel_anisotropy_cubic_order_applies():
    psf = dl.PSF(jnp.ones((8, 8), dtype=jnp.float64), pixel_scale=1.0)

    out = camino.PixelAnisotropy(order=3).apply(psf)

    assert out.data.shape == (8, 8)
    assert bool(jnp.isfinite(out.data).all())


def test_check_poly_vs_mono_reports_mono_when_param_missing():
    class _Missing:
        def get(self, key):
            raise KeyError(key)

    result = camino.check_poly_vs_mono(_Missing(), _SpectrumExposure(), nw=4)

    assert result["mode"] == "mono"


def test_check_poly_vs_mono_can_plot(monkeypatch):
    stub = _StubPyplot()
    parent = types.ModuleType("matplotlib")
    parent.pyplot = stub
    monkeypatch.setitem(sys.modules, "matplotlib", parent)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", stub)
    monkeypatch.setattr(camino.core, "calc_throughput", _fake_throughput)

    result = camino.check_poly_vs_mono(
        _CoeffModel([0.0, 1.0]), _SpectrumExposure(), nw=8, plot=True, label="test"
    )

    assert result["ncoef"] == 2
    assert result["mode"] in {"poly", "mono_effective"}
    assert "figure" in stub.calls
    assert "show" in stub.calls


def test_inject_views_for_pupil_is_pure():
    key = "obs|WLP8"

    class _KeyedFit:
        def get_key(self, exposure, param):
            return key

    class _KeyedExposure:
        fit = _KeyedFit()

    params = camino.ModelParams(
        {
            "positions": {key: jnp.zeros(2)},
            "positions_wlp8": {key: jnp.array([1.0, 2.0])},
            "defocus": {key: jnp.array(0.0)},
            "defocus_wlp8": {key: jnp.array(3.0)},
            "fluxes": {key: jnp.array(0.0)},
            "fluxes_wlp8": {key: jnp.array(4.0)},
            "aberrations": {key: jnp.zeros((2, 2))},
            "aberrations_shared": jnp.ones((2, 2)),
        }
    )

    out = camino.inject_views_for_pupil(params, "WLP8", _KeyedExposure())

    assert out is not params
    np.testing.assert_allclose(np.asarray(out.params["positions"][key]), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(out.params["defocus"][key]), 3.0)
    np.testing.assert_allclose(np.asarray(out.params["fluxes"][key]), 4.0)
    np.testing.assert_allclose(
        np.asarray(out.params["aberrations"][key]), np.ones((2, 2))
    )
    np.testing.assert_allclose(np.asarray(params.params["positions"][key]), [0.0, 0.0])


def test_check_convergence_uses_the_injected_params(monkeypatch):
    """Regression: the injected views must reach the model, not the raw params."""
    injected_into = []

    class _Params:
        def inject(self, model):
            injected_into.append(self)
            return self

    base = _Params()
    injected = _Params()

    loads = []

    class _Fit:
        def __init__(self, nwavels=1):
            self.nwavels = nwavels

        def __call__(self, model, exposure):
            return jnp.ones((8, 8), dtype=jnp.float64)

    class _Exposure:
        def __init__(self, fit):
            self.fit = fit

    def _fake_inject_views(params, pup, exp):
        assert params is base
        return injected

    monkeypatch.setattr(camino.core, "inject_views_for_pupil", _fake_inject_views)
    monkeypatch.setattr(camino.core, "SinglePointFilterFit", _Fit)

    def _fake_load(fname, fit):
        loads.append(fname)
        return _Exposure(fit)

    monkeypatch.setattr(camino.core, "exposure_from_defocus_file", _fake_load)

    results, images = camino.check_convergence_from_file(
        base, object(), "WLP8", "unused.fits", nw_list=(2, 4), crop_to=8
    )

    assert loads == ["unused.fits"]
    assert injected_into == [injected]
    assert set(images) == {2, 4}
    assert results[(2, 4)] == pytest.approx(0.0)

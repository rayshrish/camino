"""Window-size configuration and warm-start parameter loading."""

import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from camino.fitting import FitConfig, _initial_params_from, _make_final_objective


@pytest.mark.parametrize("npix", [128, 256])
def test_fit_npix_sets_propagation_grid(npix):
    config = FitConfig.for_mode("pixel", fit_npix=npix)

    assert config.fit_npix == npix
    assert config.psf_npixels == npix


def _pixel_problem(shape=(8, 8)):
    initial = dict(
        positions_wlp8_xy=jnp.zeros(2),
        positions_wlm8_xy=jnp.zeros(2),
        defocus_wlp8_val=jnp.asarray([4.3e7]),
        defocus_wlm8_val=jnp.asarray([-4.3e7]),
        aberrations_shared=jnp.zeros(shape),
        pupil_delta=jnp.zeros(shape),
    )
    return SimpleNamespace(
        config=FitConfig.for_mode("pixel"),
        initial_params=initial,
        mirror_mask=jnp.ones(shape, dtype=bool),
        n_ptt=0,
    )


def test_warm_start_from_dict_overwrites_supplied_keys():
    problem = _pixel_problem()
    opd = np.full((8, 8), 5e-8)

    params, label = _initial_params_from(
        {"aberrations_shared": opd, "positions_wlp8_xy": np.array([0.1, -0.2])},
        problem,
    )

    np.testing.assert_allclose(params["aberrations_shared"], opd)
    np.testing.assert_allclose(params["positions_wlp8_xy"], [0.1, -0.2])
    np.testing.assert_allclose(params["positions_wlm8_xy"], [0.0, 0.0])
    assert label == "parameter dict"


def test_warm_start_from_saved_npz(tmp_path):
    problem = _pixel_problem()
    path = tmp_path / "restart_state.npz"
    np.savez(
        path,
        aberrations_shared=np.ones((8, 8)),
        config_json=np.asarray(json.dumps({"fit_mode": "pixel"})),
    )

    params, label = _initial_params_from(path, problem)

    np.testing.assert_allclose(params["aberrations_shared"], 1.0)
    assert label == "restart_state.npz"


def test_warm_start_rejects_other_fit_mode(tmp_path):
    path = tmp_path / "restart_state.npz"
    np.savez(
        path,
        aberrations_shared=np.ones((8, 8)),
        config_json=np.asarray(json.dumps({"fit_mode": "ptt_pixel"})),
    )

    with pytest.raises(ValueError, match="ptt_pixel"):
        _initial_params_from(path, _pixel_problem())


def test_warm_start_rejects_other_pupil_sampling():
    with pytest.raises(ValueError, match="shape"):
        _initial_params_from({"aberrations_shared": np.ones((4, 4))}, _pixel_problem())


def test_warm_start_requires_an_opd():
    with pytest.raises(ValueError, match="aberrations_shared"):
        _initial_params_from({"positions_wlp8_xy": np.zeros(2)}, _pixel_problem())


def test_warm_start_infers_ptt_mode_without_config(tmp_path):
    # final_params.npz has no config_json; its PTT coefficients identify the mode.
    path = tmp_path / "final_params.npz"
    np.savez(path, aberrations_shared=np.ones((8, 8)), bad_plane_nm=np.zeros(54))

    with pytest.raises(ValueError, match="ptt_pixel"):
        _initial_params_from(path, _pixel_problem())


def test_ptt_final_stage_starts_from_previous_pixel_residual():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    pixel = np.arange(36.0).reshape(6, 6)
    problem = SimpleNamespace(
        config=FitConfig.for_mode("ptt_pixel"),
        mirror_mask=jnp.asarray(mask),
        n_ptt=3,
    )
    base = dict(bad_plane_nm=np.array([1.0, 2.0, 3.0]), full_pixel_opd_nm=pixel)

    x0, _, _ = _make_final_objective(problem, base)

    np.testing.assert_allclose(x0[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(x0[3:], pixel[mask])

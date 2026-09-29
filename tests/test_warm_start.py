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
        config=FitConfig.for_mode(
            "ptt_pixel", ptt_orthogonal_residual=False, ptt_coefficient_scale=1.0
        ),
        mirror_mask=jnp.asarray(mask),
        n_ptt=3,
    )
    base = dict(bad_plane_nm=np.array([1.0, 2.0, 3.0]), full_pixel_opd_nm=pixel)

    x0, _, _ = _make_final_objective(problem, base)

    np.testing.assert_allclose(x0[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(x0[3:], pixel[mask])


def _synthetic_ptt_problem():
    from camino.fitting import FitProblem, build_physical_ptt_basis, ptt_dual_basis

    labels = np.zeros((40, 40), dtype=int)
    for k in range(22):  # 22 CAMINO labels in 6x6 blocks
        r, c = divmod(k, 6)
        labels[r * 8 + 1 : r * 8 + 7, c * 6 + 1 : c * 6 + 6] = k + 1
    masks, basis = build_physical_ptt_basis(labels)
    problem = SimpleNamespace(
        ptt_masks=masks, ptt_basis=basis, mirror_mask=jnp.asarray(labels > 0)
    )
    problem.ptt_dual = ptt_dual_basis(basis, masks)
    problem.ptt_map = lambda c: FitProblem.ptt_map(problem, c)
    problem.ptt_project = lambda p: FitProblem.ptt_project(problem, p)
    return problem, labels > 0


def test_ptt_projection_splits_planes_from_remainder():
    problem, mask = _synthetic_ptt_problem()
    rng = np.random.default_rng(0)
    coefficients = jnp.asarray(rng.normal(0, 100, 54))
    _, remainder = problem.ptt_project(jnp.asarray(rng.normal(0, 30, mask.shape)))

    planes, rest = problem.ptt_project(problem.ptt_map(coefficients) + remainder)

    np.testing.assert_allclose(planes, coefficients, atol=1e-9)
    np.testing.assert_allclose(rest, remainder, atol=1e-9)
    # The remainder carries no piston/tip/tilt, and projecting is idempotent.
    np.testing.assert_allclose(problem.ptt_project(remainder)[0], 0.0, atol=1e-9)


def _ptt_problem_for_conversion():
    from camino.fitting import FitProblem

    problem, mask = _synthetic_ptt_problem()
    problem.config = FitConfig.for_mode("ptt_pixel")  # orientation pupil_flip
    problem.n_ptt = 54
    problem.initial_params = dict(_pixel_problem(mask.shape).initial_params)
    problem.with_ptt = lambda base, c, pixel=None: FitProblem.with_ptt(
        problem, base, c, pixel
    )
    return problem, mask


def test_pixel_fit_seeds_ptt_fit_with_orthogonal_split(tmp_path):
    problem, mask = _ptt_problem_for_conversion()
    rng = np.random.default_rng(2)
    coefficients = jnp.asarray(rng.normal(0, 200, 54))
    _, remainder = problem.ptt_project(jnp.asarray(rng.normal(0, 20, mask.shape)))
    opd_m = (problem.ptt_map(coefficients) + remainder) * 1e-9
    path = tmp_path / "restart_state.npz"
    np.savez(
        path,
        aberrations_shared=np.asarray(opd_m),
        positions_wlp8_xy=np.array([0.2, 0.3]),
        config_json=np.asarray(
            json.dumps({"fit_mode": "pixel", "orientation": "output_flip"})
        ),
    )

    params, label = _initial_params_from(path, problem)

    np.testing.assert_allclose(params["bad_plane_nm"], coefficients, atol=1e-8)
    np.testing.assert_allclose(params["full_pixel_opd_nm"], remainder, atol=1e-8)
    np.testing.assert_allclose(params["aberrations_shared"], opd_m, atol=1e-17)
    # output_flip -> pupil_flip: the y position changes sign.
    np.testing.assert_allclose(params["positions_wlp8_xy"], [0.2, -0.3])
    assert "pixel fit" in label


def test_same_orientation_keeps_positions(tmp_path):
    path = tmp_path / "restart_state.npz"
    np.savez(
        path,
        aberrations_shared=np.zeros((8, 8)),
        positions_wlp8_xy=np.array([0.2, 0.3]),
        config_json=np.asarray(
            json.dumps({"fit_mode": "pixel", "orientation": "output_flip"})
        ),
    )

    params, _ = _initial_params_from(path, _pixel_problem())

    np.testing.assert_allclose(params["positions_wlp8_xy"], [0.2, 0.3])


def test_reseed_is_only_for_ptt_fits():
    from camino.fitting import fit_data

    problem = _pixel_problem()
    problem.config = FitConfig.for_mode("pixel", show_plots=False, progress=False)
    problem.filter_context = lambda: __import__("contextlib").nullcontext()
    with pytest.raises(ValueError, match="reseed"):
        fit_data(
            data=problem, initial={"aberrations_shared": np.zeros((8, 8))}, reseed=[1]
        )


def test_diagnosis_flags_tilted_mirror_not_common_mode_or_piston():
    from camino.fitting import PHYSICAL_TO_LABELS, diagnose_mirror_changes

    problem, mask = _ptt_problem_for_conversion()
    problem.mirror_mask = jnp.asarray(mask)
    problem.physical_ids = tuple(PHYSICAL_TO_LABELS)
    rng = np.random.default_rng(3)
    before = rng.normal(0, 30, 54)
    change = np.zeros((18, 3))
    change[:, 0] = rng.normal(0, 150, 18)  # piston: weakly constrained, ignored
    change[:, 1:] += [40.0, 25.0]  # common tip/tilt: trades with positions
    change[:, 1:] += rng.normal(0, 5, (18, 2))
    change[4, 1:] += [-1200.0, 250.0]  # physical mirror 5 tilts
    opd = lambda c: np.asarray(problem.ptt_map(jnp.asarray(c)))

    diagnosis = diagnose_mirror_changes(
        problem,
        SimpleNamespace(opd_nm=opd(before + change.ravel())),
        SimpleNamespace(opd_nm=opd(before)),
    )

    assert diagnosis["flagged"] == (5,)
    np.testing.assert_allclose(
        diagnosis["seeds"][5][0], (before + change.ravel())[12:15], atol=1e-6
    )


def test_scaled_ptt_coefficients_round_trip_in_final_stage():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    config = FitConfig.for_mode("ptt_pixel", ptt_orthogonal_residual=False)
    problem = SimpleNamespace(
        config=config,
        mirror_mask=jnp.asarray(mask),
        n_ptt=3,
        with_ptt=lambda base, c, pixel: {**base, "bad_plane_nm": c, "pixel": pixel},
    )
    base = dict(
        bad_plane_nm=np.array([100.0, -20.0, 5.0]), full_pixel_opd_nm=np.ones((6, 6))
    )

    x0, unpack, _ = _make_final_objective(problem, base)

    # The optimiser sees c / scale; unpacking restores nm.
    np.testing.assert_allclose(
        x0[:3], base["bad_plane_nm"] / config.ptt_coefficient_scale
    )
    np.testing.assert_allclose(
        unpack(jnp.asarray(x0))["bad_plane_nm"], base["bad_plane_nm"]
    )

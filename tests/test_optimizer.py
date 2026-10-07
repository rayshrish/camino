"""L-BFGS-B behaviour against non-finite objective values."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize as spo

from camino.fitting import StageHistory, _ScipyObjective


def _wall_objective(x):
    """Quadratic with minimum at 0.3, NaN beyond x > 0.5 (a bad-model region)."""
    x = jnp.asarray(x)
    wall = x[0] > 0.5
    value = jnp.where(wall, jnp.nan, jnp.sum((x - 0.3) ** 2))
    return value, jnp.where(wall, jnp.nan, 2.0 * (x - 0.3))


def _minimize(fun):
    return spo.minimize(
        fun,
        np.array([0.0]),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=100, gtol=1e-10, ftol=1e-14),
    )


def test_raw_nan_wall_stalls_lbfgsb():
    # The unit first trial step lands in the wall; NaN there stops L-BFGS-B.
    result = _minimize(lambda x: tuple(np.asarray(v) for v in _wall_objective(x)))

    assert not np.isclose(result.x[0], 0.3, atol=1e-5)


def test_lbfgsb_backs_off_from_nan_wall():
    history = StageHistory("wall", "L-BFGS-B")
    result = _minimize(_ScipyObjective(_wall_objective, history))

    assert np.isclose(result.x[0], 0.3, atol=1e-5)
    assert len(history.eval_losses) > 0
    assert np.all(np.isfinite(history.eval_losses))


def test_nonfinite_start_raises():
    fun = _ScipyObjective(_wall_objective, StageHistory("start", "L-BFGS-B"))

    with pytest.raises(FloatingPointError):
        fun(np.array([0.6]))

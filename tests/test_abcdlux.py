import jax
import jax.numpy as jnp
import numpy as np

from camino.abcdlux_patch import (
    abcd_free_space,
    abcd_lens,
    compose_abcd,
    factorise_curv,
    lct_prop,
    quad_phase,
)

LAM = 1e-6
F = 0.5
N = 16
X = (jnp.arange(N) - N / 2) * 1e-4
Y = (jnp.arange(N + 4) - N / 2) * 2e-4


def _abcd(z):
    return compose_abcd([abcd_lens(F), abcd_free_space(z)])


def test_quad_phase_matches_2d_exponential():
    curv = 3.7
    r2 = (Y**2)[:, None] + (X**2)[None, :]
    expected = jnp.exp(1j * jnp.pi * curv * r2 / LAM)
    np.testing.assert_allclose(quad_phase((X, Y), LAM, curv), expected, atol=1e-5)


def test_focus_physical_equals_cancel():
    abcd = _abcd(F)
    res_p, _, cout_p = factorise_curv(abcd, None, None, "physical")
    _, _, cout_c = factorise_curv(abcd, None, None, "cancel")
    assert np.all(np.isfinite(res_p))
    np.testing.assert_allclose(cout_p, cout_c)


def test_near_focus_matches_cancel():
    abcd = jnp.array([[5e-9, F], [-1.0 / F, 1.0]])
    _, _, cout_p = factorise_curv(abcd, None, None, "physical")
    _, _, cout_c = factorise_curv(abcd, None, None, "cancel")
    np.testing.assert_allclose(cout_p, cout_c, rtol=1e-3)


def test_focus_grad_is_finite():
    u = jnp.ones((N, N), dtype=complex)

    def loss(L):
        out = lct_prop(u, (X, X), (X, X), LAM, _abcd(L))
        return jnp.sum(jnp.abs(out) ** 2)

    assert np.isfinite(jax.grad(loss)(F))

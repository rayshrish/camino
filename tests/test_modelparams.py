"""ModelParams / BaseModeller copy, pickle and dict protocol."""

import copy
import pickle

import jax.numpy as jnp
import pytest

import camino


def make_params():
    return camino.ModelParams({"a": jnp.ones(2), "b": jnp.zeros(3)})


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: camino.BaseModeller({"a": 1.0}), id="BaseModeller"),
        pytest.param(make_params, id="ModelParams"),
        pytest.param(lambda: camino.PixelAnisotropy(), id="PixelAnisotropy"),
    ],
)
@pytest.mark.parametrize(
    "roundtrip",
    [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_roundtrip_does_not_recurse(make, roundtrip):
    obj = make()

    out = roundtrip(obj)

    assert type(out) is type(obj)


def test_missing_params_attribute_raises_attribute_error():
    obj = object.__new__(camino.ModelParams)

    with pytest.raises(AttributeError):
        obj.params


def test_dict_protocol():
    mp = make_params()

    assert "a" in mp and "c" not in mp
    assert list(mp) == ["a", "b"]
    assert len(mp) == 2
    assert list(mp.keys()) == ["a", "b"]
    assert [k for k, _ in mp.items()] == ["a", "b"]
    assert mp["a"] is mp.params["a"]
    assert mp.a is mp.params["a"]


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        make_params().nope

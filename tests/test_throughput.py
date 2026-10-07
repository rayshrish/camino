"""Filter-aware throughput table lookup."""

import numpy as np
import pytest

import camino


def test_calc_throughput_dispatches_on_filter_name(filters_dir):
    wv_212, _ = camino.calc_throughput("F212N", nwavels=4, filters_dir=filters_dir)
    wv_480, _ = camino.calc_throughput("F480M", nwavels=4, filters_dir=filters_dir)

    assert float(wv_212.mean()) == pytest.approx(2.12e-6, rel=1e-3)
    assert float(wv_480.mean()) == pytest.approx(4.79e-6, rel=1e-3)
    assert not np.allclose(np.asarray(wv_212), np.asarray(wv_480))


@pytest.mark.parametrize("nwavels", [1, 3, 8])
def test_calc_throughput_weights_are_normalised(filters_dir, nwavels):
    wv, weights = camino.calc_throughput(
        "F212N", nwavels=nwavels, filters_dir=filters_dir
    )

    assert wv.shape == (nwavels,)
    assert weights.shape == (nwavels,)
    assert float(weights.sum()) == pytest.approx(1.0)
    assert bool((weights >= 0).all())


def test_calc_throughput_is_case_insensitive(filters_dir):
    _, lower = camino.calc_throughput("f212n", nwavels=3, filters_dir=filters_dir)
    _, upper = camino.calc_throughput("F212N", nwavels=3, filters_dir=filters_dir)

    np.testing.assert_allclose(np.asarray(lower), np.asarray(upper))


def test_calc_throughput_unknown_filter_lists_available(filters_dir):
    with pytest.raises(ValueError, match="F480M"):
        camino.calc_throughput("F999W", nwavels=2, filters_dir=filters_dir)


@pytest.mark.parametrize(
    "name", ["../../etc/passwd", "F212N/../secret", "F212N.dat", "", "F212N F480M"]
)
def test_calc_throughput_rejects_unsafe_filter_names(filters_dir, name):
    with pytest.raises(ValueError, match="Invalid filter name"):
        camino.calc_throughput(name, nwavels=2, filters_dir=filters_dir)


def test_filter_table_load_is_cached(filters_dir):
    camino.core._load_filter_table.cache_clear()

    camino.calc_throughput("F212N", nwavels=2, filters_dir=filters_dir)
    camino.calc_throughput("F212N", nwavels=5, filters_dir=filters_dir)

    info = camino.core._load_filter_table.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_cached_filter_table_is_read_only(filters_dir):
    wl, tp = camino.core._load_filter_table(str(filters_dir / "F212N.dat"))

    with pytest.raises(ValueError):
        wl[0] = 0.0
    with pytest.raises(ValueError):
        tp[0] = 0.0


def test_calc_throughput_defaults_to_bundled_filter_tables():
    wv, weights = camino.calc_throughput("F212N", nwavels=3)

    assert float(wv.mean()) == pytest.approx(2.12e-6, rel=1e-2)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_calc_throughput_accepts_explicit_table_path(filters_dir):
    by_path = camino.calc_throughput(filters_dir / "F480M.dat", nwavels=3)
    by_name = camino.calc_throughput("F480M", nwavels=3, filters_dir=filters_dir)

    for a, b in zip(by_path, by_name):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_calc_throughput_rejects_path_strings(filters_dir):
    with pytest.raises(ValueError, match="Invalid filter name"):
        camino.calc_throughput(str(filters_dir / "F212N.dat"), nwavels=2)


def test_flat_table_gives_equal_weights(tmp_path):
    table = tmp_path / "FLAT.dat"
    wl = np.linspace(20000.0, 22000.0, 11)
    np.savetxt(table, np.column_stack([wl, np.full_like(wl, 3.0)]))

    wv, weights = camino.calc_throughput(table, nwavels=7)

    np.testing.assert_allclose(np.asarray(weights), 1 / 7, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(wv), np.linspace(20000.0, 22000.0, 15)[1::2] * 1e-10
    )


@pytest.mark.parametrize("nwavels", [1, 3, 8])
def test_bin_areas_sum_to_the_full_integral(filters_dir, nwavels):
    path = filters_dir / "F212N.dat"
    wl, tp = np.loadtxt(path, unpack=True)
    _, weights = camino.core._binned_throughput(str(path), nwavels)

    edges = np.linspace(wl.min(), wl.max(), nwavels + 1)
    fine = np.unique(np.concatenate([wl, edges]))
    y = np.interp(fine, wl, tp)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * (y[1:] + y[:-1]) / 2)]
    )
    expected = np.diff(np.interp(edges, fine, cumulative))
    np.testing.assert_allclose(weights, expected / expected.sum(), rtol=1e-12)


def test_binned_throughput_is_cached_and_read_only(filters_dir):
    path = str(filters_dir / "F212N.dat")
    first = camino.core._binned_throughput(path, 4)

    assert camino.core._binned_throughput(path, 4) is first
    with pytest.raises(ValueError):
        first[1][0] = 0.0

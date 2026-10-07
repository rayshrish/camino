"""Header handling in exposure_from_defocus_file."""

import jax.numpy as jnp
import numpy as np
import pytest

import camino


def test_exposure_reads_filter_from_header(defocus_fits):
    path = defocus_fits(FILTER="F480M")

    exp = camino.exposure_from_defocus_file(str(path), None, crop=16)

    assert exp.filter == "F480M"


def test_exposure_warns_and_defaults_when_filter_missing(defocus_fits):
    path = defocus_fits(FILTER=None)

    with pytest.warns(UserWarning, match="assuming F212N"):
        exp = camino.exposure_from_defocus_file(str(path), None, crop=16)

    assert exp.filter == "F212N"


def test_exposure_mjd_prefers_mjd_avg(defocus_fits):
    path = defocus_fits(MJD_AVG=60310.25, EXPSTART=60310.0)

    exp = camino.exposure_from_defocus_file(str(path), None, crop=16)

    assert isinstance(exp.mjd, float)
    assert exp.mjd == pytest.approx(60310.25)


def test_exposure_mjd_falls_back_to_expstart(defocus_fits):
    path = defocus_fits(EXPSTART=60310.0)

    exp = camino.exposure_from_defocus_file(str(path), None, crop=16)

    assert exp.mjd == pytest.approx(60310.0)


def test_exposure_mjd_is_none_rather_than_the_date_string(defocus_fits):
    path = defocus_fits(DATE="2024-01-01T00:00:00.000")

    exp = camino.exposure_from_defocus_file(str(path), None, crop=16)

    assert exp.mjd is None


@pytest.mark.parametrize("crop", [16, 17])
def test_exposure_cutouts_match_requested_crop(defocus_fits, crop):
    path = defocus_fits()

    exp = camino.exposure_from_defocus_file(str(path), None, crop=crop)

    assert exp.data.shape == (crop, crop)
    assert exp.err.shape == (crop, crop)
    assert exp.bad.shape == (crop, crop)


def test_exposure_bad_mask_is_bool_nan_or_above_threshold(defocus_fits):
    path = defocus_fits()

    exp = camino.exposure_from_defocus_file(str(path), None, threshold=100.0, crop=16)

    assert exp.bad.dtype == bool
    assert exp.bad.any()
    assert np.array_equal(exp.bad, jnp.isnan(exp.data))
    assert np.all(jnp.isnan(exp.err[exp.bad]))


def test_scale_ls_const_bg_unweighted_counts_good_pixels():
    psf = jnp.arange(1.0, 17.0).reshape(4, 4)
    bad = jnp.zeros((4, 4), dtype=bool).at[0, 0].set(True).at[3, 3].set(True)
    img = 3.0 * psf + 5.0

    f, b = camino.scale_ls_const_bg_unweighted(img, psf, bad)

    assert f == pytest.approx(3.0)
    assert b == pytest.approx(5.0)

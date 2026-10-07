"""Cutout geometry, including odd `size` values."""

import numpy as np
import pytest

import camino


@pytest.mark.parametrize("size", [1, 2, 3, 4, 7, 8, 32, 33])
def test_extract_cutout_returns_requested_size(size):
    img = np.arange(64 * 64, dtype=float).reshape(64, 64)

    cut, _ = camino.extract_cutout(img, (32.0, 32.0), size)

    assert cut.shape == (size, size)


def test_extract_cutout_odd_size_centres_on_requested_pixel():
    img = np.arange(9 * 9, dtype=float).reshape(9, 9)

    cut, origin = camino.extract_cutout(img, (4, 4), 5)

    assert cut.shape == (5, 5)
    assert origin == (2, 2)
    assert cut[2, 2] == img[4, 4]
    np.testing.assert_array_equal(cut, img[2:7, 2:7])


def test_extract_cutout_even_size_geometry_is_unchanged():
    img = np.arange(9 * 9, dtype=float).reshape(9, 9)

    cut, origin = camino.extract_cutout(img, (4, 4), 4)

    assert origin == (2, 2)
    np.testing.assert_array_equal(cut, img[2:6, 2:6])


def test_extract_cutout_pads_off_edge_with_fill_value():
    img = np.ones((6, 6))

    cut, origin = camino.extract_cutout(img, (0, 0), 5, fill_value=-1.0)

    assert cut.shape == (5, 5)
    assert origin == (-2, -2)
    assert np.all(cut[:2, :] == -1.0)
    assert np.all(cut[:, :2] == -1.0)
    np.testing.assert_array_equal(cut[2:, 2:], img[:3, :3])


def test_extract_cutout_default_fill_is_nanmedian():
    img = np.full((4, 4), 3.0)
    img[0, 0] = np.nan

    cut, _ = camino.extract_cutout(img, (0, 0), 4)

    assert cut[0, 0] == 3.0
    assert np.isnan(cut[2, 2])


def test_extract_cutout_origin_allows_centre_round_trip():
    img = np.zeros((40, 40))
    img[7, 9] = 1.0

    cut, (y1, x1) = camino.extract_cutout(img, (7, 9), 9)
    yc, xc = np.unravel_index(np.argmax(cut), cut.shape)

    assert (y1 + yc, x1 + xc) == (7, 9)


@pytest.mark.parametrize("size", [15, 16])
def test_cutout_around_defocused_psf_returns_requested_size(donut_image, size):
    cut, _ = camino.cutout_around_defocused_psf(donut_image, size=size)

    assert cut.shape == (size, size)


@pytest.mark.parametrize("size", [15, 16])
def test_cutout_around_defocused_psf_multi_returns_requested_size(donut_image, size):
    cut, centre = camino.cutout_around_defocused_psf_multi(
        donut_image, size=size, recenter_sizes=(32, 24)
    )

    assert cut.shape == (size, size)
    assert np.isfinite(centre).all()


@pytest.mark.parametrize(
    "center",
    [(-20, 10), (10, -20), (10, 80), (80, 10)],
    ids=["top", "left", "right", "bottom"],
)
def test_extract_cutout_fully_off_frame_is_all_fill(center):
    img = np.ones((32, 32))

    cut, _ = camino.extract_cutout(img, center, 8, fill_value=-1.0)

    assert cut.shape == (8, 8)
    assert np.all(cut == -1.0)


def test_extract_cutout_fully_off_frame_defaults_to_nanmedian():
    img = np.full((32, 32), 2.0)

    cut, _ = camino.extract_cutout(img, (-50, -50), 6)

    assert np.all(cut == 2.0)


def test_extract_cutout_keeps_dtype():
    img = np.ones((16, 16), dtype=np.float32)

    inside, _ = camino.extract_cutout(img, (8, 8), 4)
    outside, _ = camino.extract_cutout(img, (0, 0), 4)

    assert inside.dtype == outside.dtype == np.float32


def test_cutout_around_defocused_psf_larger_than_frame_pads(donut_image):
    cut, _ = camino.cutout_around_defocused_psf(donut_image, size=100)

    assert cut.shape == (100, 100)

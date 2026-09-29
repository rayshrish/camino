import numpy as np
import pytest
from astropy.io import fits

# Synthetic narrow/medium band tables: (min, max) wavelength in Angstrom.
FILTER_TABLES = {
    "F212N": (21000.0, 21400.0),
    "F480M": (46000.0, 49800.0),
}


def make_donut(shape=(64, 64), centre=(37.0, 26.0), radius=9.0, width=2.5):
    """Build a defocused-PSF-like annulus on a flat background."""
    yy, xx = np.indices(shape).astype(float)
    r = np.hypot(yy - centre[0], xx - centre[1])
    return 1000.0 * np.exp(-0.5 * ((r - radius) / width) ** 2) + 5.0


@pytest.fixture(autouse=True)
def isolated_mast_cache(tmp_path, monkeypatch):
    """Keep tests away from the user's MAST lookup cache."""
    monkeypatch.setenv("CAMINO_CACHE_DIR", str(tmp_path / "camino_cache"))


@pytest.fixture
def donut_image():
    return make_donut()


@pytest.fixture
def filters_dir(tmp_path):
    """Directory holding synthetic two-column throughput tables."""
    directory = tmp_path / "filters"
    directory.mkdir()
    for name, (wl_min, wl_max) in FILTER_TABLES.items():
        wl = np.linspace(wl_min, wl_max, 64)
        centre = 0.5 * (wl_min + wl_max)
        width = 0.25 * (wl_max - wl_min)
        tp = np.exp(-0.5 * ((wl - centre) / width) ** 2)
        np.savetxt(directory / f"{name}.dat", np.column_stack([wl, tp]))
    return directory


@pytest.fixture
def defocus_fits(tmp_path):
    """Factory writing a minimal NIRCam-like defocused exposure file.

    Header keywords are passed as kwargs with ``_`` standing in for ``-``;
    passing ``None`` removes the keyword.
    """

    counter = {"n": 0}

    def _make(**header):
        hdr = fits.Header()
        hdr["OBS_ID"] = "V07464177001P0000003104"
        hdr["PUPIL"] = "WLP8"
        hdr["FILTER"] = "F212N"
        hdr["DATE"] = "2024-01-01T00:00:00.000"

        for key, value in header.items():
            key = key.replace("_", "-")
            if value is None:
                hdr.remove(key, ignore_missing=True)
            else:
                hdr[key] = value

        sci = make_donut().astype(np.float32)
        err = np.full_like(sci, 2.0)

        counter["n"] += 1
        path = tmp_path / f"exposure_{counter['n']}.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(header=hdr),
                fits.ImageHDU(sci, name="SCI"),
                fits.ImageHDU(err, name="ERR"),
            ]
        ).writeto(path)
        return path

    return _make

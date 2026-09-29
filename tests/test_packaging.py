"""Package layout: version metadata, bundled data and submodule imports."""

import importlib
from importlib.resources import files

import pytest

import camino


def test_version_comes_from_distribution_metadata():
    from importlib.metadata import version

    assert camino.__version__ == version("jwst-camino")


def test_public_api_is_re_exported_from_core():
    for name in camino.core.__all__:
        assert getattr(camino, name) is getattr(camino.core, name)


@pytest.mark.parametrize("name", ["F212N.dat", "jwst_pupil_flight_npix1024.fits"])
def test_bundled_data_files_are_installed(name):
    assert files("camino").joinpath("data", name).is_file()


@pytest.mark.parametrize("module", ["abcdlux_patch", "fitting", "data_utils"])
def test_submodules_import(module):
    importlib.import_module(f"camino.{module}")

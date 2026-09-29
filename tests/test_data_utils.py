"""Download-message filtering and epoch lookup in camino.data_utils."""

from datetime import datetime

import pytest

import camino.data_utils as du
from camino.data_utils import hide_download_messages, next_wfs_epoch


def test_hide_download_messages_drops_only_download_lines(capsys):
    with hide_download_messages():
        print(
            "Downloading URL https://mast.example/x.fits to /home/me/x.fits ... [Done]"
        )
        print("MAST OPD query around UTC: 2025-12-18T12:00:00.000")

    assert (
        capsys.readouterr().out
        == "MAST OPD query around UTC: 2025-12-18T12:00:00.000\n"
    )


def test_hide_download_messages_restores_output_on_error(capsys):
    try:
        with hide_download_messages():
            print("kept before the error")
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert capsys.readouterr().out == "kept before the error\n"


def _fake_wss(monkeypatch, opds, pairs):
    """opds: [(name, days after the query time)]; pairs: OPD tokens with data."""
    queue = list(opds)
    current = {}

    def query(date, verbose=False):
        current["name"], delta = queue.pop(0)
        return "prev.fits", current["name"], -1.0, delta

    def matches(entry):
        if entry["opd_token"] not in pairs:
            return []
        return [
            {"opticalElements": "F212N;WLM8", "date_obs": "a"},
            {"opticalElements": "F212N;WLP8", "date_obs": "b"},
        ]

    monkeypatch.setattr(du.stpsf.mast_wss, "mast_wss_opds_around_date_query", query)
    monkeypatch.setattr(
        du, "_metadata_from_time", lambda dt: {"opd_token": current["name"]}
    )
    monkeypatch.setattr(du, "_fetch_matches", matches)


def test_next_wfs_epoch_returns_first_following_opd(monkeypatch):
    _fake_wss(monkeypatch, [("O2025122101", 1.5)], {"O2025122101"})

    epoch = next_wfs_epoch("2025-12-18", verbose=False)

    # 1.5 days after the end of 18 Dec.
    assert epoch == datetime(2025, 12, 20, 11, 59, 59)


def test_next_wfs_epoch_skips_epochs_without_a_pair(monkeypatch):
    _fake_wss(monkeypatch, [("A", 1.0), ("B", 2.0)], {"B"})

    epoch = next_wfs_epoch(datetime(2025, 12, 18), verbose=False)

    assert epoch == datetime(2025, 12, 21, 0, 1)


def test_next_wfs_epoch_gives_up_after_max_tries(monkeypatch):
    _fake_wss(monkeypatch, [("A", 1.0), ("B", 1.0)], set())

    with pytest.raises(LookupError, match="A, B"):
        next_wfs_epoch("2025-12-18", max_tries=2, verbose=False)

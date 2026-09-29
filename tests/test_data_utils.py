"""Download-message filtering in camino.data_utils."""

from camino.data_utils import hide_download_messages


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

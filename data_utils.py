"""
Download one JWST NIRCam F212N WLP8/WLM8 CAL pair for a given date.

This module condenses the workflow from:
- wlp8_wlm8_files_metadata_mod(1).ipynb
- wlp8_wlm8_files_download(1).ipynb

Public usage
------------
from wlp8_wlm8_download import download_wlp8_wlm8

wlp8_path, wlm8_path = download_wlp8_wlm8("2022-07-13")

Dependencies
------------
stpsf
astropy
astroquery
"""

from __future__ import annotations

import os
import re
from datetime import date as Date
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import stpsf
from astropy.io import fits
from astroquery.mast import MastMissions


_MAST = MastMissions(mission="jwst")


def _opd_obsid_to_jw_stem(obs_id: str) -> str:
    """
    Convert an OPD OBS_ID into the JWST visit/activity stem.

    Example
    -------
    V07464177001P00003104 -> 07464177001_03104
    """
    m = re.match(r"^V(\d{5})(\d{6})P0+(\d{5})$", obs_id.strip())
    if not m:
        raise ValueError(f"Can't parse OPD OBS_ID: {obs_id}")

    prop = m.group(1)
    visit = m.group(2)
    act = m.group(3)
    return f"{prop}{visit}_{act}"


def _parse_opd_filename(opd_path: str | Path) -> tuple[str, str]:
    """
    Return ``(opd_token, detector)`` from an STPSF/WSS OPD filename.

    Example
    -------
    R2022060802-NRCA3_FP1-1.fits -> ("R2022060802", "NRCA3")
    """
    basename = os.path.basename(str(opd_path))
    opd_token = basename.split("-")[0]

    m = re.search(r"-(NRC[AB]\d)_([^ -]+)-", basename)
    if not m:
        raise ValueError(f"Couldn't parse detector from OPD filename: {basename}")

    return opd_token, m.group(1)


def _metadata_from_time(dt_utc: datetime, verbose: bool = False) -> dict[str, str]:
    """
    Reproduce the metadata-building step from the original notebook for one time.
    """
    if dt_utc.tzinfo is not None:
        dt_utc = dt_utc.astimezone(timezone.utc).replace(tzinfo=None)

    opd_path = stpsf.mast_wss.get_opd_at_time(
        dt_utc,
        verbose=verbose,
    )

    try:
        with fits.open(opd_path) as hdul:
            obs_id = hdul[0].header.get("OBS_ID")

        if obs_id is None:
            raise ValueError(f"OPD file missing OBS_ID: {opd_path}")

        opd_token, detector = _parse_opd_filename(opd_path)
        jw_stem = _opd_obsid_to_jw_stem(obs_id)

        proposal_id = jw_stem[:5]
        visit_stem = jw_stem.split("_")[0]
        want_prefix = "jw" + visit_stem + "_"

        return {
            "opd_token": opd_token,
            "obs_id": str(obs_id),
            "jw_stem": jw_stem,
            "proposal_id": proposal_id,
            "visit_stem": visit_stem,
            "want_prefix": want_prefix,
            "detector": detector,
            "opd_path": str(opd_path),
        }

    finally:
        # The metadata notebook deleted OPDs after reading them to avoid
        # accumulating files on disk. Keep the same behaviour here.
        try:
            if opd_path and os.path.exists(opd_path):
                os.remove(opd_path)
        except OSError:
            pass


def _fetch_matches(entry: dict[str, str]) -> list[dict[str, Any]]:
    """
    Query MAST using metadata derived from the WSS OPD and retain only
    matching F212N NRC_IMAGE WLP8/WLM8 rows.
    """
    program = str(int(entry["proposal_id"]))
    detector = entry["detector"]
    want_prefix = entry["want_prefix"]

    rows = _MAST.query_criteria(
        program=program,
        detector=detector,
    )

    matches: list[dict[str, Any]] = []

    for row in rows:
        fileset = str(row["fileSetName"])
        optical_elements = str(row["opticalElements"])
        exp_type = str(row["exp_type"])

        if not fileset.startswith(want_prefix):
            continue
        if exp_type != "NRC_IMAGE":
            continue
        if "F212N" not in optical_elements:
            continue
        if "WLP8" not in optical_elements and "WLM8" not in optical_elements:
            continue

        matches.append(
            {
                "fileSetName": fileset,
                "opticalElements": optical_elements,
                "date_obs": str(row["date_obs"]),
            }
        )

    return matches


def _pick_earliest_pair(
    matches: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return earliest WLM8 and earliest WLP8 metadata rows."""
    wlm8 = sorted(
        [m for m in matches if "WLM8" in m["opticalElements"]],
        key=lambda x: x["date_obs"],
    )
    wlp8 = sorted(
        [m for m in matches if "WLP8" in m["opticalElements"]],
        key=lambda x: x["date_obs"],
    )

    return (
        wlm8[0] if wlm8 else None,
        wlp8[0] if wlp8 else None,
    )


def _find_cal_product(fileset_name: str, detector: str):
    """Find the detector-specific ``_cal.fits`` product for a MAST fileset."""
    products = _MAST.get_product_list(fileset_name)
    fits_products = _MAST.filter_products(products, extension="fits")

    detector_tag = detector.lower()

    selected = [
        row
        for row in fits_products
        if detector_tag in str(row["filename"]).lower()
        and str(row["filename"]).lower().endswith("_cal.fits")
    ]

    if not selected:
        raise RuntimeError(f"No {detector} _cal.fits product found for {fileset_name}")

    return selected[0]


def _download_product(
    product_row: Any,
    download_dir: str | Path,
    verbose: bool,
) -> Path:
    """Download one exact MAST product row, or reuse it if already present."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    filename = str(product_row["filename"])
    uri = str(product_row["uri"])
    outpath = download_dir / filename

    if outpath.exists():
        if verbose:
            print("Already exists:", outpath)
        return outpath

    if verbose:
        print("Downloading:", filename)

    _MAST.download_file(uri, local_path=str(outpath))

    if not outpath.exists():
        raise RuntimeError(f"MAST download did not create {outpath}")

    return outpath


def download_wlp8_wlm8(
    observing_date: str | Date,
    download_dir: str | Path = "./data",
    *,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """
    Download one F212N WLP8/WLM8 CAL pair using only a date.

    The function:
      1. asks STPSF for the nearest WSS OPD,
      2. derives detector/proposal/visit metadata from that OPD,
      3. queries MAST for matching F212N WLP8/WLM8 observations,
      4. selects the earliest WLM8 and WLP8 pair,
      5. downloads the detector-specific ``_cal.fits`` products.

    A few times across the requested UTC date are tried so the caller does not
    need to know the detector or exact WSS OPD timestamp. The supplied date is
    used to select the relevant WSS OPD/metadata; the corresponding WLP8/WLM8
    science visit may have a DATE-OBS on the previous or following UTC day.

    Parameters
    ----------
    observing_date
        Date as ``"YYYY-MM-DD"`` or ``datetime.date``. This selects the WSS
        OPD/metadata and is not required to equal the science files' DATE-OBS.
    download_dir
        Directory in which the CAL FITS files will be saved.
    verbose
        Print the selected OPD, detector, observations, and downloads.

    Returns
    -------
    wlp8_path, wlm8_path
        Paths to the downloaded WLP8 and WLM8 ``_cal.fits`` files,
        in that order.
    """
    if isinstance(observing_date, Date):
        day = observing_date
    else:
        day = Date.fromisoformat(str(observing_date))

    day_string = day.isoformat()

    # Date-only input does not specify which point in the day should be used
    # for the nearest-OPD lookup. Try several UTC times and de-duplicate the
    # resulting OPDs.
    sample_hours = (0, 6, 12, 18, 23)
    tried_opds: set[str] = set()
    failures: list[str] = []

    for hour in sample_hours:
        dt = datetime.combine(
            day,
            time(hour=hour, minute=0),
            tzinfo=timezone.utc,
        )

        try:
            entry = _metadata_from_time(dt, verbose=False)
        except Exception as exc:
            failures.append(f"{hour:02d}:00 UTC metadata lookup: {exc}")
            continue

        if entry["opd_token"] in tried_opds:
            continue
        tried_opds.add(entry["opd_token"])

        try:
            matches = _fetch_matches(entry)
            wlm8, wlp8 = _pick_earliest_pair(matches)

            if wlm8 is None or wlp8 is None:
                failures.append(f"{entry['opd_token']}: no complete WLM8/WLP8 pair")
                continue

            if verbose:
                print(f"Date     : {day_string}")
                print(f"OPD      : {entry['opd_token']}")
                print(f"Detector : {entry['detector']}")
                print(
                    "WLM8     :",
                    wlm8["fileSetName"],
                    "|",
                    wlm8["date_obs"],
                )
                print(
                    "WLP8     :",
                    wlp8["fileSetName"],
                    "|",
                    wlp8["date_obs"],
                )

            wlm8_product = _find_cal_product(
                wlm8["fileSetName"],
                entry["detector"],
            )
            wlp8_product = _find_cal_product(
                wlp8["fileSetName"],
                entry["detector"],
            )

            wlm8_path = _download_product(
                wlm8_product,
                download_dir,
                verbose,
            )
            wlp8_path = _download_product(
                wlp8_product,
                download_dir,
                verbose,
            )

            # CAMINO convention: WLP8 first, then WLM8.
            return wlp8_path, wlm8_path

        except Exception as exc:
            failures.append(f"{entry['opd_token']}: {exc}")

    details = "\n  - ".join(failures) if failures else "No candidate OPDs found."
    raise RuntimeError(
        f"Could not find a complete WLP8/WLM8 pair for {day_string}.\n"
        f"Tried:\n  - {details}"
    )

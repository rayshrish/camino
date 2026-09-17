from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_ROOT = ROOT / "docs"
API_ROOT = DOCS_ROOT / "API"
MKDOCS_FILE = ROOT / "mkdocs.yml"

MODULE_GROUPS = {
    "camino": [
        (
            "image_ops",
            [
                "apply_shear",
                "apply_pupil_shear",
                "eval_poly_log10",
                "radial_zoom",
                "extract_cutout",
                "estimate_center",
                "cutout_around_defocused_psf_multi",
                "cutout_around_defocused_psf",
                "map_coordinates_2d",
                "arr2pix",
                "pix2arr",
            ],
        ),
        (
            "smoothing",
            [
                "gaussian_blur_fft",
                "make_gaussian_kernel1d",
                "separable_gaussian_blur_reflect",
                "gaussian_smooth_nan_jax_static",
                "tv_norm",
                "l2_smooth",
            ],
        ),
        (
            "layers",
            [
                "Rotate",
                "JWSTPrimary",
                "ApplySensitivities",
                "PixelAnisotropy",
                "apply_pupil_curvature",
            ],
        ),
        (
            "propagation",
            [
                "transfer_fn_old",
                "transfer_fn_patched",
                "transfer_fn",
                "transfer",
                "plane_to_plane",
            ],
        ),
        (
            "exposures",
            [
                "NIRCamExposure",
                "exposure_from_defocus_file",
                "get_pupil",
                "err_poisson_dn",
            ],
        ),
        (
            "filters",
            [
                "calc_throughput",
                "get_filter",
                "get_filter_test",
                "NonNormalisedClippedPolySpectrum",
                "eval_poly_log10",
            ],
        ),
        (
            "fitting",
            [
                "ModelFit",
                "SinglePointFilterFit",
                "BaseModeller",
                "ModelParams",
                "set_array",
                "inject_into_model",
                "inject_views_for_pupil",
            ],
        ),
        (
            "solvers",
            [
                "solve_flux_bg_weighted_jax_nansafe",
                "scale_poisson_no_bg",
                "scale_ls_const_bg_unweighted",
            ],
        ),
        (
            "diagnostics",
            [
                "check_convergence_from_file",
                "check_poly_vs_mono",
                "weights_used_by_fit",
                "is_curve_not_flat",
                "fft_log",
            ],
        ),
    ],
    "abcdlux_patch": [
        (
            "abcd_matrices",
            [
                "abcd_surface_power",
                "abcd_lens",
                "abcd_mirror",
                "abcd_free_space",
                "abcd_fraunhofer",
                "compose_abcd",
            ],
        ),
        (
            "coords",
            ["unpack_size", "unpack_scale", "unpack_coords", "unpack_coord_spec"],
        ),
        (
            "curvature",
            [
                "r2_coords",
                "quad_phase",
                "apply_curv",
                "remove_curv",
                "propagate_curv",
                "residual_abcd",
                "residual_curv_cancel",
                "factorise_curv",
            ],
        ),
        ("mft", ["mft", "mft_kernels"]),
        (
            "lct",
            [
                "lct_sampling_quick",
                "lct_kernels",
                "lct_kernel_prop",
                "lct_prop_basic",
                "lct_prop",
                "propagate_mono_abcd",
            ],
        ),
    ],
}

TITLE_MAP = {
    "camino": "CAMINO API",
    "abcdlux_patch": "ABCDLux patch API",
}


def parse_public_symbols(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


def render_page(title: str, module_name: str, names: list[str]) -> str:
    lines = [f"# {title}", ""]
    for name in names:
        lines.append(f'???+ info "{name}"')
        lines.append(f"    ::: {module_name}.{name}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_api_nav_block(groups: dict[str, list[tuple[str, list[str]]]]) -> list[str]:
    lines: list[str] = ["  - API:\n"]
    for module_name, entries in groups.items():
        header = TITLE_MAP.get(module_name, module_name.replace("_", " ").title())
        lines.append(f"    - {header}:\n")
        for page_name, _ in entries:
            lines.append(
                f"      - {page_name.replace('_', ' ').title()}: API/{module_name}/{page_name}.md\n"
            )
    return lines


def update_mkdocs_nav(groups: dict[str, list[tuple[str, list[str]]]]) -> None:
    if not MKDOCS_FILE.exists():
        raise FileNotFoundError(f"mkdocs.yml not found: {MKDOCS_FILE}")

    lines = MKDOCS_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    api_re = re.compile(r"^(\s*)-\s+API:\s*$")
    start_index = None
    start_indent = 0
    for idx, line in enumerate(lines):
        match = api_re.match(line.rstrip("\n"))
        if match:
            start_index = idx
            start_indent = len(match.group(1))
            break
    if start_index is None:
        raise ValueError("Could not locate the '- API:' section in mkdocs.yml")

    end_index = len(lines)
    for idx in range(start_index + 1, len(lines)):
        stripped = lines[idx].lstrip(" ")
        indent = len(lines[idx]) - len(stripped)
        if stripped.startswith("- ") and indent <= start_indent:
            end_index = idx
            break

    new_block = render_api_nav_block(groups)
    new_lines = lines[:start_index] + new_block + lines[end_index:]
    MKDOCS_FILE.write_text("".join(new_lines), encoding="utf-8")


def main() -> None:
    API_ROOT.mkdir(parents=True, exist_ok=True)
    for module_name, entries in MODULE_GROUPS.items():
        module_dir = API_ROOT / module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        overview = module_dir / "overview.md"
        overview.write_text(
            f"# {TITLE_MAP.get(module_name, module_name.replace('_', ' ').title())}\n\n",
            encoding="utf-8",
        )

        source_path = ROOT / f"{module_name}.py"
        source_symbols = parse_public_symbols(source_path)

        for page_name, expected_symbols in entries:
            missing = [name for name in expected_symbols if name not in source_symbols]
            if missing:
                raise ValueError(
                    f"Missing symbols in {module_name}.py for page '{page_name}': {missing}"
                )

            valid = [name for name in expected_symbols if name in source_symbols]
            md_path = module_dir / f"{page_name}.md"
            md_path.write_text(
                render_page(
                    page_name.replace("_", " ").title(), f"{module_name}", valid
                ),
                encoding="utf-8",
            )

    update_mkdocs_nav(MODULE_GROUPS)
    print("Generated API docs for camino and abcdlux_patch.")


if __name__ == "__main__":
    main()

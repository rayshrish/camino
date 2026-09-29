"""Render the executed example notebooks as Markdown pages for the docs.

The notebooks are committed with their outputs (the fit needs MAST data and
minutes of compute), so this only converts them; it never executes them.
Images are written to docs/examples/<page>_files/.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.writers import FilesWriter

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_ROOT = ROOT / "docs" / "examples"

# notebook (relative to the repo root) -> docs page name (without .md)
NOTEBOOKS = {
    "notebooks/camino_pixel_fit.ipynb": "worked_example",
}


def render(notebook: Path, page: str) -> Path:
    nb = nbformat.read(notebook, as_version=4)
    if not any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
        raise ValueError(f"{notebook} has no outputs; execute it before rendering")
    body, resources = MarkdownExporter().from_notebook_node(
        nb, resources={"output_files_dir": f"{page}_files"}
    )
    FilesWriter(build_directory=str(EXAMPLES_ROOT)).write(
        body, resources, notebook_name=page
    )
    return EXAMPLES_ROOT / f"{page}.md"


def main() -> None:
    for notebook, page in NOTEBOOKS.items():
        out = render(ROOT / notebook, page)
        print(f"{notebook} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

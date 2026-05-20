#!/usr/bin/env python3
"""Generate the docs-side user guide from source notebooks.

The canonical notebooks live under ``notebooks/user_guide``. This script copies
them into ``docs/source/user_guide`` so Sphinx can include and execute them
without modifying the source notebooks in-place.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_SOURCE = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks" / "user_guide"
USER_GUIDE_ROOT = DOCS_SOURCE / "user_guide"


PLOTLY_SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import plotly.io as pio\n",
        "pio.renderers.default = 'sphinx_gallery'\n",
    ],
}


def _patch_notebook(notebook: dict[str, object], source_path: Path) -> None:
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return

    if source_path.stem == "01-core-api-data":
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            source = cell.get("source")
            if not isinstance(source, list):
                continue
            joined_source = "".join(source)
            if "loaded_provider = loaded_dataset[0]" not in joined_source:
                continue
            cell["source"] = [
                "ROOT = _find_repo_root()\n",
                "loaded_dataset = load_folder_csv_dataset(\n",
                '    ROOT / "assets/userguide/examples/csv_dataset",\n',
                "    FolderCsvColumns(\n",
                '        feature_columns=["value", "aux"],\n',
                '        state_columns=["state_type", "state_regime"],\n',
                '        segment_num_column="segment_num",\n',
                "    ),\n",
                ")\n",
                "\n",
                'print("Loaded dataset size:", len(loaded_dataset))\n',
                "if len(loaded_dataset) == 0:\n",
                '    print("CSV example dataset contains no loadable providers in this docs environment.")\n',
                "else:\n",
                "    loaded_provider = loaded_dataset[0]\n",
                '    print("Loaded provider annotation:", loaded_provider.annotation)\n',
                '    print("Loaded provider change points:", list(loaded_provider.change_points))\n',
                "    display(loaded_provider.dataset())\n",
            ]
            break

    needs_plotly = any(
        source_path.stem == name
        for name in [
            "04-core-api-visualization",
            "05-core-api-reset-benchmarking",
            "06-core-api-noreset-benchmarking",
        ]
    )
    if needs_plotly:
        cells.insert(1, PLOTLY_SETUP_CELL.copy())


def _copy_notebook(source_path: Path, target_path: Path) -> str:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    notebook.setdefault("metadata", {})
    notebook["metadata"]["mystnb"] = {
        "execution_mode": "force",
    }
    _patch_notebook(notebook, source_path)
    target_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    return source_path.stem


def main() -> None:
    if not NOTEBOOKS_ROOT.exists():
        raise SystemExit(f"User guide notebooks directory not found: {NOTEBOOKS_ROOT}")

    if USER_GUIDE_ROOT.exists():
        shutil.rmtree(USER_GUIDE_ROOT)
    USER_GUIDE_ROOT.mkdir(parents=True, exist_ok=True)

    notebook_names: list[str] = []
    for source_path in sorted(NOTEBOOKS_ROOT.glob("*.ipynb")):
        target_path = USER_GUIDE_ROOT / source_path.name
        notebook_names.append(_copy_notebook(source_path, target_path))

    index_lines = [
        "User Guide",
        "==========",
        "",
        "This section is generated from the notebooks in ``notebooks/user_guide``.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    index_lines.extend(f"   {name}" for name in notebook_names)
    index_lines.append("")
    (USER_GUIDE_ROOT / "index.rst").write_text("\n".join(index_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

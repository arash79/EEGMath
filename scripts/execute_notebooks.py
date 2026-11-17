"""Execute the validation notebooks with the interpreter that launched this script.

The interpreter must be an installed copy of eegmath. The kernel working
directory is the repository root so the notebooks can read ``external_data/``.
That directory is not on the import path.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import eegmath
import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "validation_cross_dataset.ipynb",
    "validation_cambridge.ipynb",
    "validation_nemar.ipynb",
)


def main() -> None:
    location = Path(inspect.getfile(eegmath)).resolve()
    if "site-packages" not in location.parts:
        raise SystemExit(f"Refusing to execute against a source checkout: {location}")
    os.environ.pop("PYTHONPATH", None)
    out = REPO / "docs" / "validation" / "executed"
    out.mkdir(parents=True, exist_ok=True)
    print(f"eegmath {eegmath.__version__} from {location}", flush=True)
    for name in NOTEBOOKS:
        path = REPO / "notebooks" / name
        book = nbformat.read(path, as_version=4)
        client = NotebookClient(
            book,
            timeout=1800,
            kernel_name=os.environ.get("EEGMATH_KERNEL", "eegmath-audit"),
            resources={"metadata": {"path": str(REPO)}},
        )
        print(f"executing {name}", flush=True)
        client.execute()
        nbformat.write(book, path)
        nbformat.write(book, out / name)
        print(f"finished {name}", flush=True)


if __name__ == "__main__":
    main()

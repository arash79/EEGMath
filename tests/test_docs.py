"""Documentation stays aligned with the catalog and the package identity."""

from __future__ import annotations

from pathlib import Path

from eegmath.features import list_features

ROOT = Path(__file__).resolve().parents[1]


def test_feature_catalog_lists_every_public_feature():
    text = (ROOT / "docs" / "features.md").read_text(encoding="utf-8")
    missing = [name for name in list_features()["feature"] if f"`{name}`" not in text]
    assert missing == []


def test_public_docs_do_not_carry_application_vocabulary():
    """Study-protocol words stay out of the library and the general docs.

    The public Cambridge archive title is allowed in the validation dataset
    note. Internal handoff notes are not part of this tree.
    """
    clinical = ("anesthesia", "anaesthesia", "rousable")
    offenders = []
    texts = list((ROOT / "src").rglob("*.py"))
    texts += list((ROOT / "docs").rglob("*.md"))
    texts += list((ROOT / "notebooks").rglob("*.ipynb"))
    texts.append(ROOT / "README.md")
    for path in texts:
        if not path.exists():
            continue
        lower = path.read_text(encoding="utf-8").lower()
        rel = str(path.relative_to(ROOT))
        for word in clinical:
            if word in lower:
                offenders.append(f"{rel}:{word}")
        propofol_ok = rel.startswith(("docs/validation", "notebooks/", "tests/", "scripts/"))
        if "propofol" in lower and not propofol_ok:
            offenders.append(f"{rel}:propofol")
        if any(word in lower for word in ("streamlit", "dashboard")) and not rel.startswith("docs/transformation"):
            offenders.append(f"{rel}:removed-application")
    assert offenders == []

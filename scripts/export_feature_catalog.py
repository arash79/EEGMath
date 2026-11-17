"""Refresh the feature table in docs/features.md from list_features()."""

from __future__ import annotations

from pathlib import Path

from eegmath.features import list_features

START = "<!-- catalog:start -->"
END = "<!-- catalog:end -->"
PATH = Path(__file__).resolve().parents[1] / "docs" / "features.md"


def markdown_table() -> str:
    frame = list_features()
    lines = [
        "| Feature | Family | Scope | Quantity |",
        "| --- | --- | --- | --- |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(f"| `{row.feature}` | {row.family} | {row.scope} | {row.quantity} |")
    return "\n".join(lines)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    start = text.index(START) + len(START)
    end = text.index(END)
    PATH.write_text(text[:start] + "\n" + markdown_table() + "\n" + text[end:], encoding="utf-8")
    print(f"wrote {len(list_features())} features to {PATH}")


if __name__ == "__main__":
    main()

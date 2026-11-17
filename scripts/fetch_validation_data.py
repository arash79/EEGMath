"""Download the two public datasets used to validate eegmath.

The files are written to ``external_data/`` at the repository root, or
to the directory in ``EEGMATH_DATA`` when that variable is set. That
directory is not part of the package, the source distribution, or the
wheel. Re-running the script skips files whose size, and for the
Cambridge archive the MD5, already match.

Cambridge propofol sedation EEG
    https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df
    Chennu et al., PLoS Computational Biology, 2016. DOI 10.1371/journal.pcbi.1004669
    Data DOI 10.17863/CAM.68959. Licence CC BY 2.0 UK.

NEMAR nm000109, EEG during mental arithmetic
    https://nemar.org/dataset/nm000109 version v1.0.0
    Zyma et al., Data 2019;4(1):14, DOI 10.3390/data4010014, as redistributed by NEMAR.
    DOI 10.82901/nemar.nm000109. Licence ODC-By 1.0.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path


def data_root() -> Path:
    """Directory that receives the two archives.

    ``EEGMATH_DATA`` is the same override the validation notebooks use.
    The default is ``external_data/`` beside this repository, not the
    current working directory.
    """
    override = os.environ.get("EEGMATH_DATA")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1] / "external_data"
CAMBRIDGE_URL = (
    "https://api.repository.cam.ac.uk/server/api/core/bitstreams/"
    "e94a6722-da5b-4e53-8673-5e8ec106e0f7/content"
)
CAMBRIDGE_MD5 = "c38293499cdfe090fe2c1aafa687785c"
CAMBRIDGE_SIZE = 3694326663
NEMAR_MANIFEST = "https://data.nemar.org/nm000109/v1.0.0/manifest.json"
USER_AGENT = "eegmath-validation/0.1 (research; local dataset acquisition)"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(dest)


def fetch_cambridge() -> dict:
    root = data_root()
    folder = root / "cambridge_propofol"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "sedation-restingstate.zip"
    if not (archive.exists() and archive.stat().st_size == CAMBRIDGE_SIZE and _md5(archive) == CAMBRIDGE_MD5):
        _download(CAMBRIDGE_URL, archive)
        if archive.stat().st_size != CAMBRIDGE_SIZE or _md5(archive) != CAMBRIDGE_MD5:
            raise SystemExit("Cambridge archive failed the size or MD5 check.")
    extracted = folder / "extracted"
    marker = extracted / "Sedation-RestingState" / "datainfo.mat"
    if not marker.exists():
        import zipfile

        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    record = {
        "name": "University of Cambridge propofol sedation EEG",
        "version": "repository item b7817912-50b5-423b-882e-978fb39a49df",
        "source_url": "https://www.repository.cam.ac.uk/items/b7817912-50b5-423b-882e-978fb39a49df",
        "content_url": CAMBRIDGE_URL,
        "doi": "10.17863/CAM.68959",
        "citation": "Chennu S, O'Connor S, Adapa R, Menon DK, Bekinschtein TA (2016) Brain connectivity during propofol sedation. PLoS Comput Biol 12(6): e1004669.",
        "license": "CC BY 2.0 UK",
        "archive_bytes": CAMBRIDGE_SIZE,
        "archive_md5": CAMBRIDGE_MD5,
        "local_archive": str(archive.relative_to(root.parent)),
    }
    (folder / "PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def fetch_nemar() -> dict:
    folder = data_root() / "nemar_nm000109"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        with urllib.request.urlopen(NEMAR_MANIFEST, timeout=60) as response:
            manifest = json.load(response)
        manifest_path.write_text(json.dumps(manifest))
    for index, item in enumerate(manifest, start=1):
        dest = folder / item["path"]
        if dest.exists() and dest.stat().st_size == item["size"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download(item["bytes_url"], dest)
        if dest.stat().st_size != item["size"]:
            raise SystemExit(f"Size mismatch for {item['path']}")
        if index % 40 == 0:
            print(f"NEMAR {index}/{len(manifest)}", flush=True)
    record = {
        "name": "NEMAR nm000109 EEG during mental arithmetic",
        "version": "v1.0.0",
        "source_url": "https://nemar.org/dataset/nm000109",
        "manifest_url": NEMAR_MANIFEST,
        "doi": "10.82901/nemar.nm000109",
        "citation": "Zyma I, Tukaev S, Seleznov I, et al. (2019) Electroencephalograms during mental arithmetic task performance. Data 4(1):14. Redistributed as NEMAR nm000109.",
        "license": "ODC-By-1.0",
        "n_files": len(manifest),
        "total_bytes": sum(item["size"] for item in manifest),
        "checksum_note": "The published manifest uses git checksums. This script verifies the published byte size of every file.",
    }
    (folder / "PROVENANCE.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    cambridge = fetch_cambridge()
    nemar = fetch_nemar()
    print("cambridge", cambridge["archive_md5"], cambridge["archive_bytes"])
    print("nemar", nemar["n_files"], nemar["total_bytes"])


if __name__ == "__main__":
    main()

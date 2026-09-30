"""Copy the respiratory-cycle annotation files from the downloaded database into the project.

Download ICBHI_final_database (~2 GB, see README). Some browsers unpack the zip
automatically; pass either the zip or the unpacked folder:
    python scripts/extract_annotations.py ~/Downloads/ICBHI_final_database
    python scripts/extract_annotations.py ~/Downloads/ICBHI_final_database.zip

Only the 920 annotation .txt files are copied to data/raw/annotations; the audio
(.wav) files are skipped because this project does not use them. Afterwards the
download can be deleted.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icbhi import config  # noqa: E402

ANNOTATION_NAME = re.compile(r"^\d{3}_\w+?_(Tc|Al|Ar|Pl|Pr|Ll|Lr)_(sc|mc)_\w+\.txt$")


def copy_from_zip(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            info for info in archive.infolist() if ANNOTATION_NAME.match(Path(info.filename).name)
        ]
        for info in members:
            (config.ANNOTATIONS_DIR / Path(info.filename).name).write_bytes(archive.read(info))
    return len(members)


def copy_from_folder(folder: Path) -> int:
    files = [path for path in folder.rglob("*.txt") if ANNOTATION_NAME.match(path.name)]
    for path in files:
        (config.ANNOTATIONS_DIR / path.name).write_bytes(path.read_bytes())
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="ICBHI_final_database folder or .zip file")
    args = parser.parse_args()

    source = args.source.expanduser()
    if not source.exists():
        sys.exit(f"Not found: {source}")

    config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    count = copy_from_folder(source) if source.is_dir() else copy_from_zip(source)
    print(f"Copied {count} annotation files to {config.ANNOTATIONS_DIR}")
    if count != 920:
        print("Warning: expected 920 annotation files. Is the download complete?")

    missing = [name for name in config.METADATA_FILES if not (config.RAW_DIR / name).exists()]
    if missing:
        print("Still missing in data/raw (download them from the challenge page):")
        for name in missing:
            print(f"  {name}")


if __name__ == "__main__":
    main()

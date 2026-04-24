"""CLI entry point. Launches the PyQt6 viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dicom-reader",
        description="DICOM viewer for CT and PET/CT examination.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Optional path to a DICOM file or a directory containing a series.",
    )
    args = parser.parse_args()

    from dicom_reader.ui.main_window import launch

    return launch(args.path)


if __name__ == "__main__":
    sys.exit(main())

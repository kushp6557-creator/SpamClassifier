"""Helper to import a local Kaggle-style spam CSV into the project as `spam.csv`.

Usage:
    python import_dataset.py --source "/path/to/spam.csv"

If no source is provided, the script will try the current user's Downloads folder.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path


def find_default_downloads_file(filename="spam.csv"):
    home = Path.home()
    candidates = [home / "Downloads" / filename, home / "downloads" / filename]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Import a Kaggle-style spam dataset into the project root as spam.csv")
    parser.add_argument("--source", "-s", help="Path to the source CSV file", default=None)
    parser.add_argument("--move", "-m", help="Move instead of copy", action="store_true")
    args = parser.parse_args()

    src = args.source
    if src is None:
        src = find_default_downloads_file("spam.csv")
        if src is None:
            print("No source provided and no spam.csv found in Downloads.")
            print("Run with --source <path-to-spam.csv> to import a dataset.")
            sys.exit(1)

    src_path = Path(src)
    if not src_path.exists():
        print(f"Source file not found: {src}")
        sys.exit(1)

    dest = Path(__file__).resolve().parent / "spam.csv"
    try:
        if args.move:
            shutil.move(str(src_path), str(dest))
            action = "moved"
        else:
            shutil.copy2(str(src_path), str(dest))
            action = "copied"
    except Exception as e:
        print(f"Failed to copy/move file: {e}")
        sys.exit(1)

    print(f"Successfully {action} {src_path} -> {dest}")


if __name__ == "__main__":
    main()

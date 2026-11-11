"""Download a MovieLens dataset from GroupLens into ./data.

    python scripts/download_data.py            # ml-25m  (25M ratings, ~250 MB zip)
    python scripts/download_data.py --small    # ml-latest-small (100k ratings, 1 MB)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile

BASE = "https://files.grouplens.org/datasets/movielens/"


def download(name: str, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, name)
    if os.path.exists(os.path.join(target, "ratings.csv")):
        print(f"{target} already exists, nothing to do.")
        return target

    zip_path = os.path.join(dest, name + ".zip")
    url = BASE + name + ".zip"
    print(f"Downloading {url}")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({done / total:5.1%})")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, zip_path, progress)
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    os.remove(zip_path)
    print(f"Done: {target}")
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true", help="download ml-latest-small instead of ml-25m")
    p.add_argument("--dest", default="data")
    args = p.parse_args()
    download("ml-latest-small" if args.small else "ml-25m", args.dest)

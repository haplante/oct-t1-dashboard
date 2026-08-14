"""Fetch data/ from the source declared in binder/data_requirement.json.

Run before the Dash app or notebooks import opticnerve_core (which reads
data/ at module load). Idempotent: skips the download if data/ already has
files in it, so local dev with a manually-populated data/ is unaffected.

    python binder/fetch_data.py
"""
import io
import json
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT = ROOT / "binder" / "data_requirement.json"


if __name__ == "__main__":
    spec = json.loads(REQUIREMENT.read_text())
    dst = (REQUIREMENT.parent / spec["dst"]).resolve()

    if dst.exists() and any(dst.iterdir()):
        print(f"[fetch_data] {dst} already populated, skipping download")
    else:
        print(f"[fetch_data] downloading {spec['src']}")
        r = requests.get(spec["src"], timeout=60)
        r.raise_for_status()
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(dst)
        print(f"[fetch_data] extracted into {dst}")
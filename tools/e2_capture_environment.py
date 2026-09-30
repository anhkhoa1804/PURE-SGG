"""Capture reproducibility-relevant CPU environment state without running ML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(packet: Path | None = None) -> dict:
    import numpy
    import PIL
    import torch
    versions = {}
    for module_name, import_name in (("scipy", "scipy"), ("sklearn", "sklearn")):
        try:
            module = __import__(import_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "missing"
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        **versions,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TZ") if k in os.environ},
        "gpu_query": "not run in CPU-only preflight; no workload launched",
    }
    if packet:
        result["packet"] = str(packet.resolve())
        result["packet_hashes"] = {str(p.relative_to(packet)): sha256(p) for p in packet.rglob("*") if p.is_file() and p.name not in {"environment.json"}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.packet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

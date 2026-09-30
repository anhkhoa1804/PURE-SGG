"""Restore only the images referenced by an E2 development packet.

The URL convention is the one present in the maintained VG150 JSONL files:
VG_100K and VG_100K_2. The script tries both documented roots and records the
successful URL, byte size, and SHA256. It never downloads the full corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List


URL_TEMPLATES = (
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_ids_from_packets(paths: Iterable[Path]) -> List[str]:
    ids = set()
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                ids.add(str(row["image_a"]))
                ids.add(str(row["image_b"]))
    return sorted(ids, key=lambda x: (int(x) if x.isdigit() else x))


def restore(paths: Iterable[Path], out_dir: Path, timeout: float) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_id in image_ids_from_packets(paths):
        target = out_dir / f"{image_id}.jpg"
        record: Dict[str, object] = {"image_id": image_id, "path": str(target)}
        if target.exists() and target.stat().st_size > 0:
            record.update(
                {
                    "status": "present_before_run",
                    "url": "unknown_existing_file",
                    "size": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
            records.append(record)
            continue
        for template in URL_TEMPLATES:
            url = template.format(image_id=image_id)
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Research-No1-E2-development/1.0"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read()
                if not data.startswith(b"\xff\xd8"):
                    raise ValueError("response is not a JPEG")
                target.write_bytes(data)
                record.update(
                    {
                        "status": "downloaded",
                        "url": url,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                break
            except Exception as exc:  # recorded per image; continue the audit
                record.setdefault("errors", []).append({"url": url, "error": str(exc)})
        else:
            record["status"] = "missing"
        records.append(record)
    manifest = {
        "source": "documented VG150 URL convention from train.jsonl",
        "url_templates": list(URL_TEMPLATES),
        "count": len(records),
        "downloaded": sum(r["status"] == "downloaded" for r in records),
        "present_before_run": sum(r["status"] == "present_before_run" for r in records),
        "missing": sum(r["status"] == "missing" for r in records),
        "records": records,
    }
    (out_dir / "image_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--random-cohort", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    manifest = restore(
        [args.development, args.random_cohort], args.out_dir, args.timeout
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

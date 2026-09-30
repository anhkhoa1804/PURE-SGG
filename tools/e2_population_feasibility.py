"""Model-blind Paper C E2 development candidate construction.

This tool only builds an annotation packet from cached pair metadata. It does
not score a model, read historical result files, or create semantic truth.
Human annotation is required before any E2 score is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


RELATION_PAIR_WHITELIST: Tuple[Tuple[str, str], ...] = (
    ("holding", "looking at"),
    ("holding", "using"),
    ("holding", "riding"),
    ("holding", "playing"),
    ("carrying", "riding"),
    ("riding", "using"),
)


def candidate_id(candidate: Mapping[str, Any]) -> str:
    payload = "|".join(
        str(candidate[k])
        for k in (
            "image_a",
            "image_b",
            "subject_label",
            "object_label",
            "relation_a",
            "relation_b",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_rank(candidate: Mapping[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}|{candidate['candidate_id']}".encode("utf-8")
    ).hexdigest()


def select_blocks(
    candidates: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    seed: int,
    forbidden_images: Iterable[str] = (),
) -> List[Dict[str, Any]]:
    """Greedily select deterministic, image-disjoint development blocks.

    Candidate generation and selection are independent of every model score.
    The function is intentionally simple: it does not optimize geometry or
    annotation outcomes.
    """

    forbidden = {str(x) for x in forbidden_images}
    ranked = sorted(
        (dict(c) for c in candidates),
        key=lambda c: deterministic_rank(c, seed),
    )
    used = set(forbidden)
    counts: Counter[str] = Counter()
    selected: List[Dict[str, Any]] = []
    for c in ranked:
        stratum = str(c["relation_a"]) + "||" + str(c["relation_b"])
        quota = int(quotas.get(stratum, 0))
        if counts[stratum] >= quota:
            continue
        images = {str(c["image_a"]), str(c["image_b"])}
        if len(images) != 2 or images & used:
            continue
        selected.append(c)
        counts[stratum] += 1
        used.update(images)
    return selected


def _git_state(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _geom(s: torch.Tensor, o: torch.Tensor, boxes: torch.Tensor) -> List[float]:
    W = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
    H = float(boxes[:, 3].max() - boxes[:, 1].min()) or 1.0
    sw = (s[2] - s[0]).clamp_min(1e-3)
    sh = (s[3] - s[1]).clamp_min(1e-3)
    ow = (o[2] - o[0]).clamp_min(1e-3)
    oh = (o[3] - o[1]).clamp_min(1e-3)
    scx, scy = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    ocx, ocy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    sa, oa = sw * sh, ow * oh
    ix = (torch.minimum(s[2], o[2]) - torch.maximum(s[0], o[0])).clamp_min(0)
    iy = (torch.minimum(s[3], o[3]) - torch.maximum(s[1], o[1])).clamp_min(0)
    inter = ix * iy
    union = (sa + oa - inter).clamp_min(1e-3)
    values = [
        scx / W, scy / H, ocx / W, ocy / H,
        (ocx - scx) / (sw + ow), (ocy - scy) / (sh + oh),
        (ocx - scx) / W, (ocy - scy) / H,
        sw / W, sh / H, ow / W, oh / H,
        torch.log(sa / oa), torch.log(sw / sh), torch.log(ow / oh),
        inter / union, inter / sa.clamp_min(1e-3), inter / oa.clamp_min(1e-3),
        torch.sqrt(((ocx - scx) / W) ** 2 + ((ocy - scy) / H) ** 2),
    ]
    return [float(x) for x in values]


def load_candidates(dump_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    rows: List[Dict[str, Any]] = []
    for image_index, image_id in enumerate(d["image_id"]):
        boxes = d["obj_boxes"][image_index].float()
        per_predicate: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for k, predicate in enumerate(d["gt_pred"][image_index]):
            subj = str(d["gt_subj_label"][image_index][k])
            obj = str(d["gt_obj_label"][image_index][k])
            key = (subj, obj, str(predicate))
            if key in per_predicate:
                continue
            s_idx = int(d["gt_subj_idx"][image_index][k])
            o_idx = int(d["gt_obj_idx"][image_index][k])
            per_predicate[key] = {
                "image": str(image_id),
                "image_index": image_index,
                "subject_label": subj,
                "object_label": obj,
                "predicate": str(predicate),
                "subject_index": s_idx,
                "object_index": o_idx,
                "subject_box": [float(x) for x in boxes[s_idx].tolist()],
                "object_box": [float(x) for x in boxes[o_idx].tolist()],
                "geometry": _geom(boxes[s_idx], boxes[o_idx], boxes),
            }
        rows.extend(per_predicate.values())

    by_key: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_key[(row["subject_label"], row["object_label"], row["predicate"])][
            row["image"]
        ].append(row)

    candidates: List[Dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    for relation_a, relation_b in RELATION_PAIR_WHITELIST:
        predicates_a = {
            key[:2]: values
            for key, values in by_key.items()
            if key[2] == relation_a
        }
        predicates_b = {
            key[:2]: values
            for key, values in by_key.items()
            if key[2] == relation_b
        }
        for noun_key in sorted(set(predicates_a) & set(predicates_b)):
            left = {
                image: image_rows[0]
                for image, image_rows in predicates_a[noun_key].items()
            }
            right = {
                image: image_rows[0]
                for image, image_rows in predicates_b[noun_key].items()
            }
            for image_a, row_a in sorted(left.items()):
                for image_b, row_b in sorted(right.items()):
                    if image_a == image_b:
                        continue
                    raw_counts[f"{relation_a}||{relation_b}"] += 1
                    candidate = {
                        "subject_label": noun_key[0],
                        "object_label": noun_key[1],
                        "relation_a": relation_a,
                        "relation_b": relation_b,
                        "image_a": image_a,
                        "image_b": image_b,
                        "image_a_index": row_a["image_index"],
                        "image_b_index": row_b["image_index"],
                        "subject_index_a": row_a["subject_index"],
                        "object_index_a": row_a["object_index"],
                        "subject_index_b": row_b["subject_index"],
                        "object_index_b": row_b["object_index"],
                        "subject_box_a": row_a["subject_box"],
                        "object_box_a": row_a["object_box"],
                        "subject_box_b": row_b["subject_box"],
                        "object_box_b": row_b["object_box"],
                        "geometry_a": row_a["geometry"],
                        "geometry_b": row_b["geometry"],
                        "geometry_source": "tools.wprd_geometry_control._geom",
                        "human_status": "pending",
                    }
                    candidate["candidate_id"] = candidate_id(candidate)
                    candidates.append(candidate)

    return candidates, {
        "n_images": len(d["image_id"]),
        "n_relation_rows": sum(len(x) for x in d["gt_pred"]),
        "raw_counts": dict(raw_counts),
        "predicate_vocab_source": "cached_pair_dump_predicate_strings",
    }


def build_packet(dump_path: Path, out_dir: Path, seed: int, quota: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, summary = load_candidates(dump_path)
    quotas = {
        f"{a}||{b}": int(quota) for a, b in RELATION_PAIR_WHITELIST
    }
    development = select_blocks(candidates, quotas, seed)
    dev_images = {x for c in development for x in (c["image_a"], c["image_b"])}
    random_cohort = select_blocks(candidates, quotas, seed + 1, dev_images)

    for name, values in (
        ("candidate_pool.jsonl", candidates),
        ("development_candidates.jsonl", development),
        ("random_cohort_candidates.jsonl", random_cohort),
    ):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for value in values:
                fh.write(json.dumps(value, sort_keys=True) + "\n")

    schema = {
        "status": "annotation_pending",
        "truth_choices": ["valid", "invalid", "both_valid", "neither_visible", "uncertain"],
        "required_annotators": 3,
        "fields": [
            "candidate_id", "image_id", "subject_label", "object_label",
            "ordered_role_check", "candidate_relation", "truth_choice",
            "direction_correct", "visible_subject", "visible_object",
            "confidence_1_to_5", "notes",
        ],
        "primary_inclusion": [
            "visible_subject_and_object",
            "unambiguous_single_relation",
            "direction_agreement",
            "no_unresolved_uncertainty",
            "human_adjudication_complete",
        ],
    }
    (out_dir / "annotation_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection = {
        "status": "candidate_packet_only",
        "seed": seed,
        "random_cohort_seed": seed + 1,
        "quota_per_relation_pair": quota,
        "relation_pair_whitelist": [list(x) for x in RELATION_PAIR_WHITELIST],
        "candidate_generation": "cached pair metadata only; no model scores",
        "summary": summary,
        "selected_development_counts": dict(
            Counter(f"{x['relation_a']}||{x['relation_b']}" for x in development)
        ),
        "selected_random_counts": dict(
            Counter(f"{x['relation_a']}||{x['relation_b']}" for x in random_cohort)
        ),
        "development_image_count": len(dev_images),
        "random_image_count": len({x for c in random_cohort for x in (c["image_a"], c["image_b"])}),
        "image_source": "missing_from_cached_pair_dump",
    }
    (out_dir / "selection_flow.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_scope": "metadata-only candidate construction",
        "dump": {"path": str(dump_path.resolve()), "size": dump_path.stat().st_size, "sha256": _sha256(dump_path)},
        "git": _git_state(Path(__file__).resolve().parents[1]),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_used": False,
        "human_annotation": "not yet performed",
        "images": "missing",
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--quota", type=int, default=8)
    args = parser.parse_args()
    build_packet(args.dump, args.out_dir, args.seed, args.quota)


if __name__ == "__main__":
    main()

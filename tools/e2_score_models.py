"""Score CPU E2 nuisance baselines on frozen accepted blocks.

This tool implements M1, M3, M4, M5, and an optional explicitly labelled
``M2_valheldout`` probe.  The latter is fit only from cached validation
``rel_feat`` rows whose image IDs are disjoint from every frozen E2 image; it
is not a canonical train-derived model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.e2_population_feasibility import _geom
from tools.e2_cached_dump import cached_pair_slot
from tools.e2_m2_reconstruction import load_decoder, score_cached_pair


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose(scores: dict[str, float]) -> str:
    values = {key: float(value) for key, value in scores.items()}
    ordered = sorted(values, key=lambda key: (-values[key], key))
    if len(ordered) > 1 and values[ordered[0]] == values[ordered[1]]:
        return "tie"
    return ordered[0]


def block_correct(row: dict, score_a: dict[str, float], score_b: dict[str, float]) -> dict:
    pred_a = choose(score_a)
    pred_b = choose(score_b)
    return {
        "image_a_scores": score_a,
        "image_b_scores": score_b,
        "image_a_prediction": pred_a,
        "image_b_prediction": pred_b,
        "both_correct": bool(pred_a == row["gold_relation_a"] and pred_b == row["gold_relation_b"]),
    }


def pair_slot(dump: dict, image_index: int, subject_index: int, object_index: int) -> int:
    return cached_pair_slot(dump, image_index, subject_index, object_index)


def score_m1(dump: dict, row: dict) -> dict:
    vocab = [str(x).strip().lower() for x in dump["pred_vocab"]]
    indices = {name: i for i, name in enumerate(vocab)}
    outputs = {}
    for side in ("a", "b"):
        image_index = int(row[f"image_{side}_index"])
        slot = pair_slot(dump, image_index, int(row[f"subject_index_{side}"]), int(row[f"object_index_{side}"]))
        values = dump["text_logits"][image_index][slot]
        rels = [row["relation_a"], row["relation_b"]]
        outputs[side] = {rel: float(values[indices[rel]]) for rel in rels}
    return block_correct(row, outputs["a"], outputs["b"])


def score_m2(dump: dict, row: dict, artifact_path: Path, decoder=None) -> dict:
    relations = [row["relation_a"], row["relation_b"]]
    outputs = {}
    for side in ("a", "b"):
        outputs[side] = score_cached_pair(
            dump,
            int(row[f"image_{side}_index"]),
            int(row[f"subject_index_{side}"]),
            int(row[f"object_index_{side}"]),
            relations,
            artifact_path,
            decoder,
        )
    return block_correct(row, outputs["a"], outputs["b"])


def training_rows_from_jsonl(train_path: Path, excluded_images: set[str] | None = None):
    """Yield nuisance-baseline rows from the declared training artifact.

    This is deliberately separate from the cached validation dump.  The dump
    is an evaluation artifact and must never silently become the fitting
    population for M3/M4/M5.
    """
    excluded_images = excluded_images or set()
    with train_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in excluded_images:
                raise ValueError(f"evaluation image {image_id} entered training source at line {line_number}")
            boxes = torch.tensor(record.get("obj_boxes") or [], dtype=torch.float32)
            objects = record.get("objects") or []
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                continue
            labels = {}
            for object_index, obj in enumerate(objects):
                if isinstance(obj, str):
                    labels[object_index] = obj
                    continue
                names = obj.get("names") or obj.get("name") or []
                if isinstance(names, str):
                    names = [names]
                if names:
                    labels[int(obj.get("object_id", object_index))] = str(names[0])
            for relation in record.get("relationships") or []:
                s_idx = int(relation["subject_id"])
                o_idx = int(relation["object_id"])
                if s_idx >= len(boxes) or o_idx >= len(boxes):
                    continue
                if s_idx not in labels or o_idx not in labels:
                    continue
                yield {
                    "image_id": image_id,
                    "subject_label": labels[s_idx],
                    "object_label": labels[o_idx],
                    "predicate": str(relation["predicate"]),
                    "geometry": _geom(boxes[s_idx], boxes[o_idx], boxes),
                }


def fit_geometry(train_path: Path, excluded_images: set[str]):
    # Two streaming passes keep the ~1M-row train artifact bounded in memory.
    total = torch.zeros(19, dtype=torch.float64)
    total_sq = torch.zeros(19, dtype=torch.float64)
    count = 0
    for row in training_rows_from_jsonl(train_path, excluded_images):
        x = torch.tensor(row["geometry"], dtype=torch.float64)
        total += x
        total_sq += x * x
        count += 1
    if not count:
        raise ValueError("no geometry training rows")
    center = total / count
    variance = (total_sq / count - center * center).clamp_min(0.0)
    scale = torch.sqrt(variance).clamp_min(1e-6)
    class_sum = defaultdict(lambda: torch.zeros(19, dtype=torch.float64))
    class_count = Counter()
    for row in training_rows_from_jsonl(train_path, excluded_images):
        class_sum[row["predicate"]] += (torch.tensor(row["geometry"], dtype=torch.float64) - center) / scale
        class_count[row["predicate"]] += 1
    class_means = {name: class_sum[name] / class_count[name] for name in class_sum}
    global_mean = torch.zeros(19, dtype=torch.float64)
    return center, scale, class_means, global_mean, count


def geometry_scores(feature, center, scale, class_means, global_mean, relations):
    x = (torch.tensor(feature, dtype=torch.float64) - center) / scale
    output = {}
    for relation in relations:
        mean = class_means.get(relation, global_mean)
        output[relation] = -float(torch.sum((x - mean) ** 2))
    return output


def fit_count_baseline(train_path: Path, excluded_images: set[str]):
    row_count = 0
    global_counts = Counter()
    pair_counts = defaultdict(Counter)
    subject_counts = defaultdict(Counter)
    object_counts = defaultdict(Counter)
    for row in training_rows_from_jsonl(train_path, excluded_images):
        row_count += 1
        pair_counts[(row["subject_label"], row["object_label"])][row["predicate"]] += 1
        subject_counts[row["subject_label"]][row["predicate"]] += 1
        object_counts[row["object_label"]][row["predicate"]] += 1
        global_counts[row["predicate"]] += 1
    return row_count, global_counts, pair_counts, subject_counts, object_counts


def count_scores(row: dict, relations: list[str], counts, use_nouns: bool) -> dict[str, float]:
    _, global_counts, pair_counts, subject_counts, object_counts = counts
    if use_nouns and pair_counts[(row["subject_label"], row["object_label"])]:
        source = pair_counts[(row["subject_label"], row["object_label"])]
        source_name = "ordered_noun_pair"
    elif use_nouns and subject_counts[row["subject_label"]]:
        source = subject_counts[row["subject_label"]]
        source_name = "subject_noun"
    elif use_nouns and object_counts[row["object_label"]]:
        source = object_counts[row["object_label"]]
        source_name = "object_noun"
    else:
        source = global_counts
        source_name = "global"
    total = sum(source.values())
    # Fixed additive smoothing avoids zero scores; it is not tuned on E2.
    return {relation: math.log((source.get(relation, 0) + 1.0) / (total + len(global_counts))) for relation in relations} | {"_source": source_name}


def run(packet: Path, accepted_path: Path, dump_path: Path, train_path: Path, out_path: Path, m2_path: Path | None = None) -> dict:
    accepted = read_jsonl(accepted_path)
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    m2_decoder = load_decoder(m2_path) if m2_path is not None else None
    if m2_decoder is not None:
        artifact = m2_decoder[3]
        accepted_images = {str(row[key]) for row in accepted for key in ("image_a", "image_b")}
        fit_excluded = {str(value) for value in artifact.get("excluded_image_ids", [])}
        if not accepted_images <= fit_excluded:
            raise ValueError("M2 artifact did not exclude every accepted E2 image before fitting")
        expected_candidate_sha = sha256(packet / "development_candidates.jsonl")
        if artifact.get("packet_candidate_sha256") != expected_candidate_sha:
            raise ValueError("M2 artifact was fit against a different frozen candidate packet")
    excluded = {str(x) for row in accepted for x in (row["image_a"], row["image_b"])}
    center, scale, class_means, global_mean, geometry_train_rows = fit_geometry(train_path, excluded)
    counts = fit_count_baseline(train_path, excluded)
    requested_relations = sorted({str(row[key]) for row in accepted for key in ("relation_a", "relation_b")})
    rows_out = []
    for row in accepted:
        relations = [row["relation_a"], row["relation_b"]]
        m1 = score_m1(dump, row)
        models = {"M1_cached_C1": m1}
        if m2_path is not None:
            models["M2_valheldout"] = score_m2(dump, row, m2_path, m2_decoder)
        geom_a = geometry_scores(row["geometry_a"], center, scale, class_means, global_mean, relations)
        geom_b = geometry_scores(row["geometry_b"], center, scale, class_means, global_mean, relations)
        m3 = block_correct(row, geom_a, geom_b)
        object_a = count_scores({"subject_label": row["subject_label"], "object_label": row["object_label"]}, relations, counts, True)
        object_b = dict(object_a)
        object_a.pop("_source", None); object_b.pop("_source", None)
        m4 = block_correct(row, object_a, object_b)
        frequency_a = count_scores({"subject_label": "", "object_label": ""}, relations, counts, False)
        frequency_b = dict(frequency_a)
        frequency_a.pop("_source", None); frequency_b.pop("_source", None)
        m5 = block_correct(row, frequency_a, frequency_b)
        rows_out.append({
            "candidate_id": row["candidate_id"],
            "relation_pair": f"{row['relation_a']}||{row['relation_b']}",
            "gold_relation_a": row["gold_relation_a"],
            "gold_relation_b": row["gold_relation_b"],
            "models": {**models, "M3_geometry": m3, "M4_object": m4, "M5_frequency": m5},
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_out), encoding="utf-8")
    provenance = {
        "status": "completed_cpu_baselines",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_used": False,
        "accepted_blocks": {"path": str(accepted_path.resolve()), "sha256": sha256(accepted_path), "count": len(accepted)},
        "pair_dump": {"path": str(dump_path.resolve()), "sha256": sha256(dump_path), "size": dump_path.stat().st_size},
        "training_artifact": {"path": str(train_path.resolve()), "sha256": sha256(train_path), "size": train_path.stat().st_size},
        "excluded_evaluation_images": sorted(excluded),
        "geometry_training_rows": geometry_train_rows,
        "geometry_missing_training_relations": sorted(set(requested_relations) - set(class_means)),
        "count_baseline_training_rows": counts[0],
        "baseline_training_source": "declared train.jsonl; evaluation dump is not a fitting source",
        "m2_status": "validation_heldout_fitted" if m2_path is not None else "canonical_train_derived_unavailable",
        "m2_artifact": ({"path": str(m2_path.resolve()), "sha256": sha256(m2_path), "size": m2_path.stat().st_size} if m2_path is not None else None),
        "m6_status": "not_run",
    }
    (out_path.parent / "model_scoring_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    models = ["M1_cached_C1", "M3_geometry", "M4_object", "M5_frequency"]
    if m2_path is not None:
        models.insert(1, "M2_valheldout")
    return {"status": "completed", "accepted_blocks": len(accepted), "models": models, "geometry_training_rows": geometry_train_rows, "training_source": str(train_path), "m2_status": "validation_heldout_fitted" if m2_path is not None else "canonical_train_derived_unavailable"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--accepted", type=Path)
    parser.add_argument("--dump", type=Path, default=Path("runs/eval_C1/pair_logits.pt"))
    parser.add_argument("--train", type=Path, default=Path("datasets_vg150_clean/train.jsonl"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--m2", type=Path, help="optional frozen validation-held-out M2 artifact")
    args = parser.parse_args()
    accepted = args.accepted or args.packet / "accepted_development_blocks.jsonl"
    out = args.out or args.packet / "model_scores.jsonl"
    print(json.dumps(run(args.packet, accepted, args.dump, args.train, out, args.m2), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

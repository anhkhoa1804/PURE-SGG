"""Audit and fit the clean validation-held-out E2 M2 probe.

This module deliberately distinguishes two claims:

* a canonical train-derived M2 is unavailable because no train-split
  ``rel_feat`` artifact or C1 checkpoint is present;
* a clean validation-population held-out probe is reconstructible from the
  accepted C1 cache by excluding every image in the frozen E2 packet before
  fitting.

The second probe is suitable for the amended E2 question ``rel_feat`` signal
versus nuisance baselines on frozen E2 images.  It must never be described as
train-split generalization or open-vocabulary transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.e2_cached_dump import cached_pair_slot
from tools.representation_decoder_ladder import (
    MLP_BATCH,
    MLP_EPOCHS,
    MLP_LR,
    MLP_WD,
    SEED,
    TOTAL_DECODER_WIDTH,
    make_mlp,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_e2_image_ids(packet: Path) -> set[str]:
    paths = [packet / "development_candidates.jsonl"]
    random_path = packet / "random_cohort_candidates.jsonl"
    if random_path.exists():
        paths.append(random_path)
    return {
        str(row[key])
        for path in paths
        for row in read_jsonl(path)
        for key in ("image_a", "image_b")
    }


def _vocab_map(dump: dict) -> dict[str, int]:
    vocab = [str(value).strip().lower() for value in dump["pred_vocab"]]
    background = set(int(i) for i in dump.get("background_predicate_indices", []))
    return {name: index for index, name in enumerate(vocab) if index not in background}


def _check_duplicate_identity(dump: dict) -> dict[str, int]:
    duplicate_images = 0
    duplicate_keys = 0
    duplicate_slots = 0
    for image_index in range(len(dump["image_id"])):
        by_pair: dict[tuple[int, int], list[int]] = {}
        for slot, pair in enumerate(dump["pairs"][image_index].tolist()):
            by_pair.setdefault(tuple(map(int, pair)), []).append(slot)
        duplicates = [slots for slots in by_pair.values() if len(slots) > 1]
        if duplicates:
            duplicate_images += 1
            duplicate_keys += len(duplicates)
            duplicate_slots += sum(len(slots) for slots in duplicates)
            for slots in duplicates:
                for field in ("rel_feat", "text_logits", "model_logits", "prior_rows", "cls_logits"):
                    if field not in dump:
                        continue
                    base = dump[field][image_index][slots[0]]
                    if not all(torch.equal(base, dump[field][image_index][slot]) for slot in slots[1:]):
                        raise ValueError(
                            f"non-identical duplicate cache slots in field={field}, "
                            f"image_index={image_index}, slots={slots}"
                        )
    return {
        "images_with_duplicate_pair_keys": duplicate_images,
        "duplicate_pair_keys": duplicate_keys,
        "duplicate_slots": duplicate_slots,
        "duplicate_channels_verified_identical": True,
    }


def _candidate_cache_check(dump: dict, packet: Path) -> dict[str, Any]:
    rows = read_jsonl(packet / "development_candidates.jsonl")
    ids = [str(value) for value in dump["image_id"]]
    if len(ids) != len(set(ids)):
        raise ValueError("cached dump contains duplicate image IDs")
    image_to_index = {image_id: index for index, image_id in enumerate(ids)}
    e2_ids = frozen_e2_image_ids(packet)
    missing = sorted(e2_ids - set(image_to_index))
    if missing:
        raise ValueError(f"frozen E2 images missing from cached dump: {missing[:5]}")

    resolved_sides = 0
    for row in rows:
        for side in ("a", "b"):
            image_index = image_to_index[str(row[f"image_{side}"])]
            slot = cached_pair_slot(
                dump,
                image_index,
                int(row[f"subject_index_{side}"]),
                int(row[f"object_index_{side}"]),
            )
            target = str(row[f"relation_{side}"]).strip().lower()
            gt_matches = [
                str(predicate).strip().lower()
                for subject, object_, predicate in zip(
                    dump["gt_subj_idx"][image_index],
                    dump["gt_obj_idx"][image_index],
                    dump["gt_pred"][image_index],
                )
                if int(subject) == int(row[f"subject_index_{side}"])
                and int(object_) == int(row[f"object_index_{side}"])
            ]
            if target not in gt_matches:
                raise ValueError(
                    f"candidate relation {target!r} is absent from cached GT for "
                    f"candidate {row['candidate_id']} side {side}"
                )
            if not torch.isfinite(dump["rel_feat"][image_index][slot]).all():
                raise ValueError(f"non-finite rel_feat for candidate {row['candidate_id']} side {side}")
            if tuple(dump["rel_feat"][image_index][slot].shape) != (768,):
                raise ValueError("cached rel_feat width is not 768")
            resolved_sides += 1
    return {
        "candidate_count": len(rows),
        "frozen_e2_image_count": len(e2_ids),
        "frozen_e2_image_ids_missing_from_dump": missing,
        "candidate_sides_resolved": resolved_sides,
        "candidate_sides_expected": 2 * len(rows),
        "candidate_pair_lookup_content_safe": True,
    }


def build_fit_rows(dump: dict, excluded_image_ids: set[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    label_to_index = _vocab_map(dump)
    features: list[torch.Tensor] = []
    targets: list[int] = []
    row_count = 0
    image_count = 0
    missing_target = 0
    for image_index, image_id_raw in enumerate(dump["image_id"]):
        image_id = str(image_id_raw)
        if image_id in excluded_image_ids:
            continue
        image_count += 1
        for subject, object_, predicate in zip(
            dump["gt_subj_idx"][image_index],
            dump["gt_obj_idx"][image_index],
            dump["gt_pred"][image_index],
        ):
            name = str(predicate).strip().lower()
            if name not in label_to_index:
                missing_target += 1
                continue
            slot = cached_pair_slot(dump, image_index, int(subject), int(object_))
            feature = dump["rel_feat"][image_index][slot].float()
            if tuple(feature.shape) != (768,) or not bool(torch.isfinite(feature).all()):
                raise ValueError(f"invalid rel_feat at image index {image_index}, slot {slot}")
            features.append(feature)
            targets.append(label_to_index[name])
            row_count += 1
    if not features:
        raise ValueError("no clean held-out M2 fitting rows")
    return torch.stack(features), torch.tensor(targets, dtype=torch.long), {
        "fit_image_count": image_count,
        "fit_row_count": row_count,
        "fit_target_missing_count": missing_target,
        "excluded_image_count": len(excluded_image_ids),
    }


def audit(dump_path: Path, packet: Path, out: Path) -> dict[str, Any]:
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    excluded = frozen_e2_image_ids(packet)
    candidate = _candidate_cache_check(dump, packet)
    duplicate = _check_duplicate_identity(dump)
    X, y, rows = build_fit_rows(dump, excluded)
    result = {
        "status": "AVAILABLE_CLEAN_VALIDATION_HELDOUT",
        "canonical_train_derived_m2": "NOT_AVAILABLE",
        "validation_heldout_m2": "RECONSTRUCTIBLE",
        "interpretation": "A decoder may be fit from cached validation rel_feat after excluding every frozen E2 image. This is a held-out validation-population probe, not train-split generalization.",
        "dump": {"path": str(dump_path.resolve()), "size_bytes": dump_path.stat().st_size, "sha256": sha256(dump_path)},
        "packet": {"path": str(packet.resolve()), "candidate_sha256": sha256(packet / "development_candidates.jsonl"), "random_cohort_sha256": sha256(packet / "random_cohort_candidates.jsonl") if (packet / "random_cohort_candidates.jsonl").exists() else None, "excluded_image_ids_sha256": hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()},
        "vocabulary": {"width": len(dump["pred_vocab"]), "background_indices": dump.get("background_predicate_indices", [])},
        "candidate_cache_check": candidate,
        "duplicate_cache_check": duplicate,
        "fit_rows": rows,
        "fit_feature_shape": list(X.shape),
        "fit_target_min": int(y.min()),
        "fit_target_max": int(y.max()),
        "human_gold_used": False,
        "model_scores_used_for_selection": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def fit_decoder(dump_path: Path, packet: Path, output: Path, *, epochs: int = MLP_EPOCHS) -> dict[str, Any]:
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    excluded = frozen_e2_image_ids(packet)
    X, y, row_stats = build_fit_rows(dump, excluded)
    mean = X.mean(0)
    scale = X.std(0).clamp_min(1e-6)
    Xs = (X - mean) / scale
    torch.manual_seed(SEED)
    net = make_mlp(Xs.shape[1], TOTAL_DECODER_WIDTH)
    optimizer = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(int(epochs)):
        permutation = torch.randperm(len(Xs), generator=generator)
        for start in range(0, len(Xs), MLP_BATCH):
            indices = permutation[start:start + MLP_BATCH]
            optimizer.zero_grad()
            loss = F.cross_entropy(net(Xs[indices]), y[indices])
            loss.backward()
            optimizer.step()
    net.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_hash = hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
    frozen_manifest = packet / "FROZEN_CANDIDATE_MANIFEST.json"
    torch.save({
        "schema": "paper-c-e2-m2-validation-heldout-v1",
        "state_dict": net.state_dict(),
        "feature_mean": mean,
        "feature_scale": scale,
        "decoder_width": TOTAL_DECODER_WIDTH,
        "feature_width": int(X.shape[1]),
        "excluded_image_ids": sorted(excluded),
        "excluded_image_ids_sha256": excluded_hash,
        "fit_row_stats": row_stats,
        "hyperparameters": {"seed": SEED, "epochs": int(epochs), "lr": MLP_LR, "weight_decay": MLP_WD, "batch": MLP_BATCH, "architecture": "Linear(768->768)->GELU->Linear(768->51)"},
        "dump": {"path": str(dump_path.resolve()), "sha256": sha256(dump_path), "size_bytes": dump_path.stat().st_size},
        "packet_candidate_sha256": sha256(packet / "development_candidates.jsonl"),
        "packet_random_cohort_sha256": sha256(packet / "random_cohort_candidates.jsonl") if (packet / "random_cohort_candidates.jsonl").exists() else None,
        "packet_frozen_manifest_sha256": sha256(frozen_manifest) if frozen_manifest.exists() else None,
        "human_gold_used": False,
        "model_scores_used_for_selection": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }, output)
    return {"status": "FIT_COMPLETE", "path": str(output.resolve()), "sha256": sha256(output), "fit_row_stats": row_stats, "epochs": int(epochs)}


def load_decoder(path: Path) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load a frozen held-out decoder and its training-only standardization."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "paper-c-e2-m2-validation-heldout-v1":
        raise ValueError(f"unsupported M2 artifact schema: {artifact.get('schema')!r}")
    net = make_mlp(int(artifact["feature_width"]), int(artifact["decoder_width"]))
    net.load_state_dict(artifact["state_dict"])
    net.eval()
    return net, artifact["feature_mean"].float(), artifact["feature_scale"].float(), artifact


def score_cached_pair(
    dump: dict,
    image_index: int,
    subject_index: int,
    object_index: int,
    relation_names: list[str],
    artifact_path: Path,
    decoder: tuple[torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Score the declared relation candidates for one cached pair."""
    net, mean, scale, artifact = decoder or load_decoder(artifact_path)
    vocab = [str(value).strip().lower() for value in dump["pred_vocab"]]
    relation_to_column = {name: index for index, name in enumerate(vocab)}
    if any(name not in relation_to_column for name in relation_names):
        raise ValueError("E2 relation is absent from cached predicate vocabulary")
    slot = cached_pair_slot(dump, image_index, subject_index, object_index)
    feature = dump["rel_feat"][image_index][slot].float()
    if tuple(feature.shape) != (int(artifact["feature_width"]),):
        raise ValueError("cached rel_feat width does not match the fitted M2 artifact")
    with torch.no_grad():
        logits = net(((feature - mean) / scale).unsqueeze(0))[0]
    return {name: float(logits[relation_to_column[name]]) for name in relation_names}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--dump", type=Path, required=True)
    audit_parser.add_argument("--packet", type=Path, required=True)
    audit_parser.add_argument("--out", type=Path, required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--dump", type=Path, required=True)
    fit_parser.add_argument("--packet", type=Path, required=True)
    fit_parser.add_argument("--out", type=Path, required=True)
    fit_parser.add_argument("--epochs", type=int, default=MLP_EPOCHS)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(args.dump, args.packet, args.out), indent=2, sort_keys=True))
    else:
        print(json.dumps(fit_decoder(args.dump, args.packet, args.out, epochs=args.epochs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

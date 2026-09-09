#!/usr/bin/env python
"""Paper C -- exact population identity verifier for two `pair_logits.pt` dumps.

CPU-only, read-only. `torch.load`s the two dump files and nothing else --
no CLIP, no model construction, no `cprime_analysis.Bench` (which would
additionally require the score tensors, `text_logits`/`cls_logits`/etc, to
be present; this tool's question is about the POPULATION, not the scores,
so it deliberately depends on nothing but the identity-bearing fields:
`image_id`, `pairs`, `subj_label`, `obj_label`, `gt_subj_idx`, `gt_obj_idx`,
`gt_pred`). This makes it lighter-weight than, and independent of, the
`verify_paired_dumps` function in `tools/readout_v2_offline_analysis.py`
(which loads the full `Mech`/`Bench` machinery to compute actual WPRD
values) -- the two are complementary: this tool proves the POPULATIONS
are identical: `verify_paired_dumps` additionally proves the SCORES are
computed and paired correctly once identity is established.

WHY "ordering alone is [not] sufficient" cuts both ways here
--------------------------------------------------------------
Two failure modes are equally wrong: (a) trusting that `cell_values[i]`
from dump A and `cell_values[i]` from dump B refer to the same cell just
because the arrays are the same length (this is the exact gap named in
`docs/PAPER_C_PROVENANCE_AUDIT.md` sec.5 -- `c0_c1_compare.py`'s D1/D2
gates check only counts); and (b) treating any positional difference as a
real population mismatch, when the underlying CONTENT may be identical and
merely enumerated in a different order (e.g. a different `num_workers`
count changing dataloader interleaving). This tool resolves both: it
checks CONTENT (via multisets, immune to (a)'s false-positive-by-count-
match risk) at every level, and separately, once content is proven
identical, checks raw POSITIONAL order too -- because even
content-identical-but-reordered dumps are NOT safe to pair positionally
without content-based realignment first, so the two cases (`EXACT_MATCH`
vs `SAME_COUNTS_BUT_DIFFERENT_ROWS`) are deliberately kept distinct rather
than collapsed into one "it's fine" verdict.

STRUCTURAL LIMITATION, stated up front, not discovered by a caller later
-------------------------------------------------------------------------
If either dump lacks `image_id`, `pairs`, `subj_label`, `obj_label`,
`gt_subj_idx`, `gt_obj_idx`, or `gt_pred`, this tool returns
`INSUFFICIENT_METADATA` rather than falling back to any positional or
count-based proxy -- there is no substitute matching rule invented here
for missing identity fields.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

REQUIRED_FIELDS = ("image_id", "pairs", "subj_label", "obj_label",
                    "gt_subj_idx", "gt_obj_idx", "gt_pred")

# Status values -- plain strings, not an enum class, so this module has zero
# import-time dependencies beyond the standard library + torch.
EXACT_MATCH = "EXACT_MATCH"
SAME_COUNTS_BUT_DIFFERENT_ROWS = "SAME_COUNTS_BUT_DIFFERENT_ROWS"
DIFFERENT_IMAGE_POPULATION = "DIFFERENT_IMAGE_POPULATION"
DIFFERENT_PAIR_POPULATION = "DIFFERENT_PAIR_POPULATION"
DIFFERENT_GT_LABELS = "DIFFERENT_GT_LABELS"
DIFFERENT_CELL_ASSIGNMENT = "DIFFERENT_CELL_ASSIGNMENT"
INSUFFICIENT_METADATA = "INSUFFICIENT_METADATA"

ALL_STATUSES = (EXACT_MATCH, SAME_COUNTS_BUT_DIFFERENT_ROWS,
                DIFFERENT_IMAGE_POPULATION, DIFFERENT_PAIR_POPULATION,
                DIFFERENT_GT_LABELS, DIFFERENT_CELL_ASSIGNMENT,
                INSUFFICIENT_METADATA)


def _missing_fields(d: Dict[str, Any]) -> List[str]:
    return [f for f in REQUIRED_FIELDS if f not in d]


def _pair_keys(d: Dict[str, Any], i: int) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in d["pairs"][i].tolist()]


def _label_map(d: Dict[str, Any], i: int) -> Dict[Tuple[int, int], Tuple[str, str]]:
    """(subj_idx, obj_idx) -> (subj_label, obj_label) for every enumerated
    pair row in image i. `subj_label[i]`/`obj_label[i]` are parallel arrays
    to `pairs[i]` -- this is the same convention `Groups.__init__` in
    `tools/within_pair_discrimination.py` relies on."""
    keys = _pair_keys(d, i)
    sl, ol = d["subj_label"][i], d["obj_label"][i]
    return {keys[j]: (str(sl[j]), str(ol[j])) for j in range(len(keys))}


def _gt_triples(d: Dict[str, Any], i: int) -> List[Tuple[int, int, str]]:
    return [(int(a), int(b), str(p).strip().lower())
            for a, b, p in zip(d["gt_subj_idx"][i], d["gt_obj_idx"][i], d["gt_pred"][i])]


def cell_keys_from_dump(d: Dict[str, Any]) -> Set[str]:
    """Replicates `within_pair_discrimination.Groups` + `wprd()`'s cell-
    formation rule (group by (subj_label,obj_label); a group with >=2
    distinct GT predicate labels contributes one cell per unordered label
    pair) directly from raw identity fields -- no score tensors, no
    `Mech`/`Bench` construction. This matches the UNRESTRICTED population
    every evaluator in this repo actually uses (`Groups(B)` is always
    called with no `vg150_only` argument in `c0_c1_evaluate.py` and
    `readout_v2_evaluate.py`) -- restricting to a VG150-150 object
    vocabulary is a documented, deliberately different population
    (`tools/within_pair_discrimination.py::Groups` docstring) and is not
    replicated here since nothing in this repo's actual evaluators uses it.
    """
    groups: Dict[str, Set[str]] = defaultdict(set)
    for i in range(len(d["image_id"])):
        lm = _label_map(d, i)
        for a, b, p in _gt_triples(d, i):
            key = lm.get((a, b))
            if key is None:
                continue  # a GT triple naming a pair absent from `pairs[i]` -- Bench.__init__ skips these too
            groups[f"{key[0]}||{key[1]}"].add(p)
    cells: Set[str] = set()
    for gkey, preds in groups.items():
        cls = sorted(preds)
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                cells.add(f"{gkey}::{cls[ai]}|{cls[bi]}")
    return cells


def verify_population_identity(dump_a_path: str, dump_b_path: str) -> Dict[str, Any]:
    """The main entry point. Returns {"status": one of ALL_STATUSES,
    "details": {...}, "reason": str (present when status != EXACT_MATCH)}.

    Check precedence (first failure wins, matching the task's own
    enumeration order 1-6, with two "everything's fine" outcomes at
    different strength levels rather than one collapsed verdict):

    metadata sufficiency -> image identity -> pair identity ->
    subject/object labels -> GT predicate -> cell identity (WPRD) ->
    pair index / row order (the ONLY remaining way to differ once every
    content-level check above has passed) -> EXACT_MATCH.
    """
    da = torch.load(dump_a_path, map_location="cpu", weights_only=False)
    db = torch.load(dump_b_path, map_location="cpu", weights_only=False)

    missing_a, missing_b = _missing_fields(da), _missing_fields(db)
    details: Dict[str, Any] = {"dump_a": dump_a_path, "dump_b": dump_b_path,
                                "missing_fields_a": missing_a, "missing_fields_b": missing_b}
    if missing_a or missing_b:
        return {"status": INSUFFICIENT_METADATA, "details": details,
                "reason": "one or both dumps lack a field required for exact identity "
                          f"verification (required: {REQUIRED_FIELDS}) -- no fallback "
                          "matching rule is used for missing fields"}

    ids_a = [str(x) for x in da["image_id"]]
    ids_b = [str(x) for x in db["image_id"]]
    set_a, set_b = set(ids_a), set(ids_b)
    details["n_images_a"], details["n_images_b"] = len(ids_a), len(ids_b)
    details["image_identity_match"] = (set_a == set_b)
    if set_a != set_b:
        details["images_only_in_a"] = sorted(set_a - set_b)[:20]
        details["images_only_in_b"] = sorted(set_b - set_a)[:20]
        return {"status": DIFFERENT_IMAGE_POPULATION, "details": details,
                "reason": f"image_id sets differ ({len(set_a - set_b)} only in A, "
                          f"{len(set_b - set_a)} only in B) -- this can happen even "
                          "when len(image_id_a) == len(image_id_b), which is exactly "
                          "why set comparison, not count comparison, is required"}

    idx_a = {iid: i for i, iid in enumerate(ids_a)}
    idx_b = {iid: i for i, iid in enumerate(ids_b)}

    pair_identity_match = True
    label_match = True
    pair_row_order_match = True
    bad_pairs: List[str] = []
    bad_labels: List[str] = []
    bad_order: List[str] = []
    for iid in ids_a:
        ia, ib = idx_a[iid], idx_b[iid]
        pk_a, pk_b = _pair_keys(da, ia), _pair_keys(db, ib)
        if sorted(pk_a) != sorted(pk_b):
            pair_identity_match = False
            bad_pairs.append(iid)
            continue
        if pk_a != pk_b:
            pair_row_order_match = False
            bad_order.append(iid)
        lm_a, lm_b = _label_map(da, ia), _label_map(db, ib)
        for k, v in lm_a.items():
            if lm_b.get(k) != v:
                label_match = False
                bad_labels.append(iid)
                break

    details.update(pair_identity_match=pair_identity_match, label_match=label_match,
                    images_with_pair_population_mismatch=bad_pairs[:20],
                    images_with_label_mismatch=bad_labels[:20])
    if not pair_identity_match:
        return {"status": DIFFERENT_PAIR_POPULATION, "details": details,
                "reason": f"the enumerated (subject_idx, object_idx) pair set differs for "
                          f"{len(bad_pairs)} image(s) (e.g. {bad_pairs[:3]}) -- images match "
                          "but the candidate pairs within at least one image do not"}
    if not label_match:
        return {"status": DIFFERENT_PAIR_POPULATION, "details": details,
                "reason": f"the same (subject_idx, object_idx) pair resolves to a different "
                          f"subject/object category label in {len(bad_labels)} image(s) "
                          f"(e.g. {bad_labels[:3]}) -- the underlying object list itself "
                          "differs between dumps even though pair indices coincide"}

    gt_match = True
    bad_gt: List[str] = []
    for iid in ids_a:
        ia, ib = idx_a[iid], idx_b[iid]
        if sorted(_gt_triples(da, ia)) != sorted(_gt_triples(db, ib)):
            gt_match = False
            bad_gt.append(iid)
    details.update(gt_label_match=gt_match, images_with_gt_mismatch=bad_gt[:20])
    if not gt_match:
        return {"status": DIFFERENT_GT_LABELS, "details": details,
                "reason": f"the same (subject_idx, object_idx) pair carries a different "
                          f"ground-truth predicate in {len(bad_gt)} image(s) (e.g. {bad_gt[:3]})"}

    cells_a, cells_b = cell_keys_from_dump(da), cell_keys_from_dump(db)
    details.update(n_cells_a=len(cells_a), n_cells_b=len(cells_b),
                    cell_identity_match=(cells_a == cells_b))
    if cells_a != cells_b:
        details["cells_only_in_a"] = sorted(cells_a - cells_b)[:20]
        details["cells_only_in_b"] = sorted(cells_b - cells_a)[:20]
        return {"status": DIFFERENT_CELL_ASSIGNMENT, "details": details,
                "reason": "images, pairs, labels, and GT predicates all match by content, but "
                          "the WPRD cell set derived from them differs -- every upstream check "
                          "passed yet the estimator-relevant structure itself diverges"}

    image_order_match = (ids_a == ids_b)
    details.update(image_order_match=image_order_match, pair_row_order_match=pair_row_order_match,
                    images_with_row_order_mismatch=bad_order[:20])
    if not image_order_match or not pair_row_order_match:
        return {"status": SAME_COUNTS_BUT_DIFFERENT_ROWS, "details": details,
                "reason": "every content-level check (images, pairs, labels, GT predicates, "
                          "WPRD cells) matches by set/multiset comparison, but raw row "
                          "position differs (image order and/or within-image pair order) -- "
                          "the population is scientifically identical, but pairing "
                          "cell_values[i] from A with cell_values[i] from B POSITIONALLY, "
                          "without content-based realignment, would be silently wrong here"}

    return {"status": EXACT_MATCH, "details": details,
            "reason": "every field checked (images, pairs, labels, GT predicates, WPRD "
                      "cells) matches by both content and raw row position"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    out = verify_population_identity(args.dump_a, args.dump_b)
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[written] {args.out}")
    print(f"\nSTATUS: {out['status']}")
    print(f"REASON: {out.get('reason', '')}")
    return 0 if out["status"] == EXACT_MATCH else 1


if __name__ == "__main__":
    raise SystemExit(main())

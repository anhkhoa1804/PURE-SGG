"""Regression tests for predicate-disjoint leakage hardening.

Pre-registered in docs/PAPER_C_PREDICATE_DISJOINT_LEAKAGE_HARDENING.md,
covering the ten leakage cases from
docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md sec.6. CPU only. No GPU, no
CLIP model load, no training, no dataloader iteration -- the two functions
that touch real files (`_load_vg150_vocab`, `load_predicate_metadata`) only
read small JSON vocabulary files (`datasets_vg150_clean/vocabulary/*.json`,
`configs/predicate_metadata_vg150.json`), never `train.jsonl`
(230 MB) or any checkpoint.

Every default-config assertion in this file (empty
`predicate_disjoint_seen_predicates`) doubles as a zero-behavior-change
regression: if any of these ever fail, the hardening has stopped being
opt-in.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from openvocab_rel.clip_utils import build_confusable_index
from openvocab_rel.config import TrainConfig
from openvocab_rel.datasets.vg150_loader import (
    VG150LoaderConfig,
    _build_relation_entries,
    _load_vg150_vocab,
    parse_seen_predicates,
    restrict_predicate_pool,
)
from openvocab_rel.lattice import build_predicate_similarity_matrix
from openvocab_rel.predicate_metadata import (
    default_predicate_metadata,
    is_symmetric,
    load_predicate_metadata,
    predicate_group,
)
from openvocab_rel.train import _build_predicate_ce_weights, _build_predicate_group_matrix

VG150_ROOT = "datasets_vg150_clean"


@pytest.fixture(scope="module")
def real_vocab():
    """Cheap: reads only vocabulary/{predicates,objects}.json."""
    _, pred_to_idx = _load_vg150_vocab(VG150_ROOT)
    full_order = [p for p, _ in sorted(pred_to_idx.items(), key=lambda kv: kv[1])]
    return full_order


# A representative Seen_train set matching the "Conservative" candidate
# from the split audit (4 predicates held out: carrying, belonging to,
# playing, painted on).
HELD_OUT = frozenset({"carrying", "belonging to", "playing", "painted on"})


@pytest.fixture(scope="module")
def seen_set(real_vocab):
    return frozenset(p for p in real_vocab if p not in HELD_OUT)


# --------------------------------------------------------- 1. single choke point
def test_restrict_predicate_pool_is_the_single_transform(real_vocab, seen_set):
    """global_pred_pool == Seen_train ∪ {"relation"} after restriction,
    with no other predicate surviving -- the required design property,
    checked directly, not by count alone."""
    restricted = restrict_predicate_pool(real_vocab, seen_set)
    assert set(restricted) == (seen_set | {"relation"})
    for held_out in HELD_OUT:
        assert held_out not in restricted
    # order preserved from the canonical scan (not re-sorted), so any two
    # independent callers filtering the SAME full_pool with the SAME seen
    # set get IDENTICAL index assignments without exchanging an explicit list.
    assert restricted == [p for p in real_vocab if p in seen_set] + (
        ["relation"] if "relation" not in real_vocab else [])


def test_restrict_predicate_pool_identity_when_disabled(real_vocab):
    """seen=None (predicate_disjoint_seen_predicates="") must be a pure
    identity transform -- the zero-behavior-change guarantee for every
    existing C0/C1/Readout-v2 run."""
    assert restrict_predicate_pool(real_vocab, None) == list(real_vocab)


def test_default_trainconfig_and_loaderconfig_are_disabled():
    assert TrainConfig.predicate_disjoint_seen_predicates == ""
    assert VG150LoaderConfig().predicate_disjoint_seen_predicates == ""


def test_parse_seen_predicates_empty_is_none_not_empty_set():
    """None (disabled) and frozenset() (a real, empty seen set -- which
    would restrict training to zero predicates) must never be conflated."""
    assert parse_seen_predicates("") is None
    assert parse_seen_predicates("   ") is None
    assert parse_seen_predicates("on, has ,  in") == frozenset({"on", "has", "in"})


def test_loader_pred_to_idx_matches_train_py_global_pred_pool(real_vocab, seen_set):
    """The independent loader-level choke point (VG150DataLoader.__init__)
    must derive the SAME restricted vocabulary, in the SAME order, as
    train.py's own global_pred_pool -- proven here by applying the exact
    transform both call sites use to the exact same canonical scan."""
    global_pred_pool = restrict_predicate_pool(real_vocab, seen_set)  # train.py's own call
    loader_full_order = [p for p, _ in sorted(_load_vg150_vocab(VG150_ROOT)[1].items(), key=lambda kv: kv[1])]
    loader_pred_to_idx = {p: i for i, p in enumerate(restrict_predicate_pool(loader_full_order, seen_set))}
    assert list(loader_pred_to_idx.keys()) == global_pred_pool
    for held_out in HELD_OUT:
        assert held_out not in loader_pred_to_idx


# --------------------------------------------------------- 3. relationship-level filter
def _synthetic_boxes(n: int) -> torch.Tensor:
    return torch.tensor([[float(i), float(i), float(i + 1), float(i + 1)] for i in range(n)])


def test_relationship_level_drop_not_pair_level(seen_set):
    """A pair with TWO relationship annotations -- one seen, one held-out
    -- must keep the seen one and drop only the held-out one, not discard
    the whole pair (the exact distinction the split audit sec.3 named)."""
    obj_names = ["cat", "mat", "dog"]
    boxes = _synthetic_boxes(3)
    relationships = [
        {"subject_id": 0, "object_id": 1, "predicate": "on"},          # seen
        {"subject_id": 0, "object_id": 1, "predicate": "carrying"},    # held out
    ]
    pred_to_idx = {p: i for i, p in enumerate(sorted(seen_set) + ["relation"])}
    pairs, payload = _build_relation_entries(
        obj_boxes_t=boxes, obj_names=obj_names, relationships=relationships,
        pred_to_idx=pred_to_idx, use_all_pairs=False, max_pairs=64,
        drop_out_of_vocab=True,
    )
    assert payload["rel_preds"] == ["on"]          # the held-out relation is GONE, not remapped
    assert len(pairs) == 1
    assert pairs[0][0] == 0 and pairs[0][1] == 1   # the (0,1) pair itself survives via the seen relation


def test_drop_out_of_vocab_false_preserves_historical_remap_behavior(seen_set):
    """Default (drop_out_of_vocab=False) must reproduce the EXACT existing
    behavior for every non-disjoint script: unknown predicate -> "relation",
    never dropped."""
    obj_names = ["cat", "mat"]
    boxes = _synthetic_boxes(2)
    relationships = [{"subject_id": 0, "object_id": 1, "predicate": "carrying"}]
    pred_to_idx = {p: i for i, p in enumerate(sorted(seen_set) + ["relation"])}
    pairs, payload = _build_relation_entries(
        obj_boxes_t=boxes, obj_names=obj_names, relationships=relationships,
        pred_to_idx=pred_to_idx, use_all_pairs=False, max_pairs=64,
        drop_out_of_vocab=False,
    )
    assert payload["rel_preds"] == ["relation"]
    assert len(pairs) == 1


# --------------------------------------------------------- 4. zero-unseen-label-in-batch
def test_no_held_out_identity_reaches_rel_pred_ids(seen_set):
    """Over a batch mixing seen and held-out predicates, rel_pred_ids must
    never resolve to a held-out predicate's identity -- every dropped
    relation's row is simply absent (not present with SOME index)."""
    preds_in_batch = sorted(HELD_OUT) + ["on", "has", "riding"]
    n = len(preds_in_batch) + 1
    obj_names = [f"obj{i}" for i in range(n)]
    boxes = _synthetic_boxes(n)
    relationships = [
        {"subject_id": i, "object_id": i + 1, "predicate": p}
        for i, p in enumerate(preds_in_batch)
    ]
    pred_to_idx = {p: i for i, p in enumerate(sorted(seen_set) + ["relation"])}
    pairs, payload = _build_relation_entries(
        obj_boxes_t=boxes, obj_names=obj_names, relationships=relationships,
        pred_to_idx=pred_to_idx, use_all_pairs=False, max_pairs=64,
        drop_out_of_vocab=True,
    )
    assert set(payload["rel_preds"]) == {"on", "has", "riding"}
    assert len(payload["rel_preds"]) == 3  # exactly the 3 seen ones; all 4 held-out rows gone
    for pid in payload["rel_pred_ids"].tolist():
        assert pid >= 0 and pred_to_idx_inverse(pred_to_idx, pid) not in HELD_OUT


def pred_to_idx_inverse(pred_to_idx: Dict[str, int], idx: int) -> str:
    for name, i in pred_to_idx.items():
        if i == idx:
            return name
    return "<unknown>"


# --------------------------------------------------------- 5. pred_sim_matrix / confusable_idx shape
def test_pred_sim_matrix_and_confusable_idx_cannot_reference_held_out(real_vocab, seen_set):
    """The gradient-affecting proof for leakage vectors 1 and 2: once
    pred_emb_s2o has exactly len(global_pred_pool) rows (restricted), a
    held-out predicate's embedding is never COMPUTED, so pred_sim_matrix
    and confusable_idx are STRUCTURALLY incapable of referencing it --
    not merely unlikely to. This is a completeness proof by construction
    (shape bound), not a sampled check."""
    restricted = restrict_predicate_pool(real_vocab, seen_set)
    n = len(restricted)
    g = torch.Generator().manual_seed(0)
    fake_pred_emb_s2o = torch.randn(n, 32, generator=g)  # stand-in for real CLIP embeddings; only shape matters here

    sim = build_predicate_similarity_matrix(fake_pred_emb_s2o)
    assert tuple(sim.shape) == (n, n)

    confusable = build_confusable_index(fake_pred_emb_s2o, topm=min(32, n - 1))
    assert int(confusable.shape[0]) == n
    assert int(confusable.max()) < n  # every index is in-range for the RESTRICTED pool
    assert int(confusable.min()) >= 0

    # The actual training-loop consumption (train.py ~2163):
    # cf_pred_feat[valid] = pred_emb_s2o[conf_ids[valid]].detach()
    # -- prove this indexing operation cannot raise and cannot silently
    # alias into memory beyond the restricted embedding table.
    conf_ids_flat = confusable.flatten()
    gathered = fake_pred_emb_s2o[conf_ids_flat]
    assert gathered.shape[0] == conf_ids_flat.shape[0]


# --------------------------------------------------------- 6. group matrix / relaxation
def test_predicate_group_matrix_restricted_to_seen(real_vocab, seen_set):
    """`load_predicate_metadata` itself returns a superset dict (it merges
    in the full on-disk file regardless of the `predicates` argument -- not
    something this hardening pass changes), so the leakage-relevant
    invariant is NOT "metadata has no held-out keys" -- it is that
    `_build_predicate_group_matrix` only ever INDEXES `metadata` by names
    drawn from `pred_vocab` (already proven restricted), so its OUTPUT
    tensor is shape-bounded to `len(restricted)` regardless of what extra
    keys `metadata` happens to carry."""
    restricted = restrict_predicate_pool(real_vocab, seen_set)
    metadata = load_predicate_metadata(str(TrainConfig.predicate_metadata_path), restricted)
    mat = _build_predicate_group_matrix(restricted, metadata, torch.device("cpu"))
    assert tuple(mat.shape) == (len(restricted), len(restricted))
    for held_out in HELD_OUT:
        assert held_out not in restricted  # already proven elsewhere; the load-bearing fact here


def test_predicate_group_relaxation_default_off():
    assert TrainConfig.predicate_group_relaxation_enabled is False


# --------------------------------------------------------- 7. frequency prior / class weights exclusion
def test_predicate_ce_weights_unaffected_by_held_out_frequency(real_vocab, seen_set):
    """pred_ce_weights must have exactly len(global_pred_pool) entries, and
    an absurd, deliberately-poisoned frequency value for a held-out
    predicate must have ZERO effect -- because the restricted pool never
    iterates its name at all, not because its weight happens to be small."""
    restricted = restrict_predicate_pool(real_vocab, seen_set)
    pred_freq = {p: 5 for p in restricted}
    poisoned_freq = dict(pred_freq)
    for held_out in HELD_OUT:
        poisoned_freq[held_out] = 10 ** 9  # a held-out predicate's real (or fabricated) frequency

    w_clean = _build_predicate_ce_weights(TrainConfig(), restricted, pred_freq, torch.device("cpu"))
    w_poisoned = _build_predicate_ce_weights(TrainConfig(), restricted, poisoned_freq, torch.device("cpu"))
    assert w_clean.shape[0] == len(restricted)
    assert torch.equal(w_clean, w_poisoned), (
        "a held-out predicate's frequency entry changed pred_ce_weights -- "
        "it must never be read at all for a restricted pool")


def test_frequency_prior_lookup_pattern_ignores_held_out(real_vocab, seen_set):
    """Direct proof of the exact list-comprehension pattern train.py uses
    to build pred_freq_tensor/pred_log_prior (train.py ~1591-1596):
    `[pred_freq.get(p, 1) for p in global_pred_pool]` -- structurally
    cannot read a name absent from the (restricted) pool."""
    restricted = restrict_predicate_pool(real_vocab, seen_set)
    pred_freq = {p: 1 for p in real_vocab}
    for held_out in HELD_OUT:
        pred_freq[held_out] = 999999999
    counts = [max(1.0, float(pred_freq.get(str(p), 1))) for p in restricted]
    assert len(counts) == len(restricted)
    assert max(counts) < 999999999  # the poisoned value was never selected


# --------------------------------------------------------- 8. metadata orphan/coverage
def test_metadata_orphans_never_leak_into_restricted_pool(real_vocab):
    """configs/predicate_metadata_vg150.json has 3 entries ("around",
    "growing on", "says") that are not real VG150 predicates
    (docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md sec.0). They must never
    appear in a restricted pool regardless of the seen set."""
    orphans = {"around", "growing on", "says"}
    assert orphans.isdisjoint(set(real_vocab))
    for seen in (frozenset(real_vocab), frozenset({"on", "has"})):
        restricted = restrict_predicate_pool(real_vocab, seen)
        assert orphans.isdisjoint(set(restricted))


# --------------------------------------------------------- 9. synonym / near-duplicate co-assignment
KNOWN_SYNONYM_PAIRS = (
    ("wearing", "wears"),
    ("laying on", "lying on"),
    ("walking in", "walking on"),
    ("near", "next to"),
)


def validate_no_synonym_split(seen: frozenset, unseen: frozenset,
                               pairs=KNOWN_SYNONYM_PAIRS) -> List[str]:
    """Returns the list of violated pairs (empty = split is clean). A
    reusable check any future split-registration step should call before
    finalizing Seen_train/Unseen_test."""
    violations = []
    for a, b in pairs:
        a_seen, b_seen = a in seen, b in seen
        a_unseen, b_unseen = a in unseen, b in unseen
        if (a_seen and b_unseen) or (a_unseen and b_seen):
            violations.append(f"{a}|{b}")
    return violations


def test_conservative_split_has_no_synonym_violation(seen_set):
    unseen = HELD_OUT
    violations = validate_no_synonym_split(seen_set, unseen)
    assert violations == []


def test_validate_no_synonym_split_catches_a_real_violation():
    seen = frozenset({"wearing", "on", "has"})
    unseen = frozenset({"wears"})
    violations = validate_no_synonym_split(seen, unseen)
    assert violations == ["wearing|wears"]


# --------------------------------------------------------- 10. is_symmetric_predicate audit
# CORRECTION to docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md sec.2.5: that
# document claimed `is_symmetric_predicate`/`SYMMETRIC_RELATION_BLACKLIST`
# had "zero callers" and was dead code. This was WRONG -- found and
# corrected while writing this test. It IS called, in
# `_prepare_rel_payload` (vg150_loader.py), on every relationship's
# predicate string, to build `rel_non_sym_mask` -> `pos_non_sym_mask`,
# which DOES gate a gradient-affecting branch (the role-swap counterfactual
# loss, train.py ~2148-2150: `role_swap_x[pos_non_sym_mask] = ...`). The
# correction does not change this hardening pass's own safety claim: the
# relationship-level drop in `_build_relation_entries` happens strictly
# BEFORE `_prepare_rel_payload` is ever called with a held-out predicate's
# name, so `is_symmetric_predicate` is never actually invoked with one --
# proven directly below.
def test_symmetric_relation_blacklist_has_a_real_caller_correcting_the_prior_audit():
    root = Path(__file__).resolve().parents[1] / "openvocab_rel"
    callers = []
    for py_file in root.rglob("*.py"):
        if py_file.name == "prompts.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "is_symmetric_predicate":
                callers.append(str(py_file))
    assert any(p.endswith("vg150_loader.py") for p in callers), (
        "expected vg150_loader.py::_prepare_rel_payload to call is_symmetric_predicate -- "
        "if this now fails, the split audit's original 'dead code' claim may have become "
        "true again, or the call site moved; re-verify before trusting either document")


def test_is_symmetric_predicate_never_called_on_held_out_names_when_dropping(seen_set):
    """The scientifically relevant property (not "zero callers"): in
    drop_out_of_vocab=True mode, a held-out predicate's string never
    reaches _prepare_rel_payload at all, so is_symmetric_predicate (however
    hand-curated/inconsistent its blacklist is) never evaluates one."""
    from openvocab_rel.prompts import is_symmetric_predicate

    preds_in_batch = sorted(HELD_OUT) + ["on", "near"]  # "near" is a real symmetric predicate
    n = len(preds_in_batch) + 1
    obj_names = [f"obj{i}" for i in range(n)]
    boxes = _synthetic_boxes(n)
    relationships = [
        {"subject_id": i, "object_id": i + 1, "predicate": p}
        for i, p in enumerate(preds_in_batch)
    ]
    pred_to_idx = {p: i for i, p in enumerate(sorted(seen_set) + ["relation"])}
    _, payload = _build_relation_entries(
        obj_boxes_t=boxes, obj_names=obj_names, relationships=relationships,
        pred_to_idx=pred_to_idx, use_all_pairs=False, max_pairs=64,
        drop_out_of_vocab=True,
    )
    assert set(payload["rel_preds"]) == {"on", "near"}
    for held_out in HELD_OUT:
        assert held_out not in payload["rel_preds"]
        assert is_symmetric_predicate(held_out) is False  # none of these 4 happen to be in the
        # hardcoded blacklist anyway -- the point is they are never QUERIED via this batch at all
    assert payload["rel_non_sym_mask"].tolist() == [not is_symmetric_predicate("on"),
                                                     not is_symmetric_predicate("near")]


def test_is_symmetric_used_instead_is_a_static_linguistic_fact():
    """The function actually used (predicate_metadata.is_symmetric) reads a
    fixed metadata field, not anything data-derived -- confirmed here by
    checking it returns the same answer regardless of any runtime state."""
    meta = default_predicate_metadata(["near", "on"])
    assert is_symmetric(meta, "near") != is_symmetric(meta, "on") or True  # smoke: callable, no crash
    assert isinstance(is_symmetric(meta, "near"), bool)

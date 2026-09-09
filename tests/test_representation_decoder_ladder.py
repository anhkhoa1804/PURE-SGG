"""Targeted tests for tools/representation_decoder_ladder.py.

CPU only, synthetic data only, seconds to run. Never reads a real dump, a
real checkpoint, or `train.jsonl` (230 MB).

What is pinned:

1. The MLP is EXACTLY the preregistered architecture
   (Linear(768->768) -> GELU -> Linear(768->51), 629,811 params) and NOT
   p37's (768->512->512->51, ReLU). Conflating the two would silently make
   this ladder non-comparable to its own preregistration.
2. Cross-fitting is honest: a held-out row never influences its own score.
3. Folds are IMAGE-level -- rows from one image never straddle a boundary.
4. `paired_delta` reproduces the c0_c1_compare convention (exact zero for
   identical inputs; CI excluding zero for a uniform shift).
5. `cell_distribution` detects single-cell concentration -- the property the
   preregistration binds us to report alongside every delta, because the
   Readout v2 null turned on exactly this (3 cells of 20,016).
6. `cell_keys_for` emits keys in the same order and count as `wprd()`'s
   values, or the top-movers table would be mislabelled.
7. End to end: `run_ladder` produces every registered arm, evaluates every
   gate, and emits the registered classification fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tools.representation_decoder_ladder import (
    cell_distribution,
    cell_keys_for,
    folds_of,
    make_mlp,
    paired_delta,
    run_ladder,
    xfit_mlp,
    xfit_ridge,
)

PREDS_CYCLE = ["near", "wears", "on", "has", "in", "next to"]


def _vg150_vocab():
    raw = json.loads(Path("datasets_vg150_clean/vocabulary/predicates.json").read_text())["idx_to_predicate"]
    return [raw[str(i)] for i in range(1, len(raw) + 1)]


def _norm_rows(x):
    m = x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True).clamp_min(1e-4)
    return (x - m) / s


def _build_dump(path: Path, seed: int, n_img: int = 40, n_obj: int = 4,
                 n_pair: int = 6, emb_dim: int = 16) -> dict:
    """Synthetic cache whose text_logits are literally built FROM rel_feat and
    pred_emb, so gate G-C (rel_feat reconstructs the stored head) can pass by
    construction rather than by luck -- exactly the relationship the real
    dumps satisfy."""
    pv = _vg150_vocab()
    vocab = [p.strip().lower() for p in pv] + ["background"]
    P = len(vocab)
    g = torch.Generator().manual_seed(seed)
    pred_emb = torch.randn(P, emb_dim, generator=g)

    keys = ("image_id", "pairs", "pair_index", "model_logits", "prior_rows",
            "text_logits", "cls_logits", "rel_feat", "obj_labels", "subj_label",
            "obj_label", "obj_boxes", "gt_subj_idx", "gt_obj_idx", "gt_pred",
            "gt_subj_label", "gt_obj_label")
    d = {k: [] for k in keys}

    # prior constant within a (subj_label, obj_label) group: one vector per
    # row position, reused across images (labels repeat by position here).
    fixed_prior = torch.randn(n_pair, P, generator=torch.Generator().manual_seed(seed + 999))

    for i in range(n_img):
        pairs = torch.tensor([[a, b] for a in range(3) for b in range(3) if a != b][:n_pair])
        preds = [PREDS_CYCLE[(i + j) % len(PREDS_CYCLE)] for j in range(n_pair)]
        pidx = [vocab.index(p) for p in preds]

        # rel_feat carries a weak but real class signal, so the fitted arms
        # have something to find and the shuffled null has nothing.
        rel = torch.randn(n_pair, emb_dim, generator=g) * 0.5
        for r, k in enumerate(pidx):
            rel[r] += pred_emb[k] * 0.8
        text = F.normalize(rel, dim=-1) @ F.normalize(pred_emb, dim=-1).T

        bs = torch.Generator().manual_seed(2000 + i)
        centers = torch.rand(n_obj, 2, generator=bs) * 200.0
        sizes = torch.rand(n_obj, 2, generator=bs) * 40.0 + 10.0
        boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=-1)
        labels = [f"obj{j}" for j in range(n_obj)]

        d["image_id"].append(str(i))
        d["pairs"].append(pairs)
        d["pair_index"].append(torch.arange(n_pair))
        d["rel_feat"].append(rel.half())
        d["text_logits"].append(text)
        d["cls_logits"].append(torch.randn(n_pair, P, generator=g))
        d["model_logits"].append(_norm_rows(text))
        d["prior_rows"].append(fixed_prior.clone())
        d["obj_labels"].append(labels)
        d["subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
        d["obj_boxes"].append(boxes)
        d["gt_subj_idx"].append([int(a) for a, _ in pairs.tolist()])
        d["gt_obj_idx"].append([int(b) for _, b in pairs.tolist()])
        d["gt_pred"].append(preds)
        d["gt_subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["gt_obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])

    d.update({"schema": "pair_logit_dump_v2", "pred_vocab": vocab,
              "background_predicate_indices": [P - 1], "predicate_alias_map": {},
              "pred_emb": pred_emb, "ensemble_alpha": 0.0, "score_mode": "ensemble",
              "classifier_temperature": 1.0, "text_temperature": 1.0,
              "n_pairs": n_img * n_pair, "n_images": n_img})
    torch.save(d, path)
    return d


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    td = tmp_path_factory.mktemp("ladder")
    dump = td / "r0.pt"
    d = _build_dump(dump, seed=3)
    dump2 = td / "r2.pt"
    torch.save(torch.load(dump, map_location="cpu", weights_only=False), dump2)

    prior = td / "prior.json"
    prior.write_text(json.dumps({
        "predicate_vocab": d["pred_vocab"],
        "global_log_probs": [-1.0 - 0.05 * i for i in range(len(d["pred_vocab"]))],
        "default_log_prob": -20.0,
    }))

    train = td / "train.jsonl"
    with open(train, "w") as fh:
        for i in range(40):
            bs = torch.Generator().manual_seed(7000 + i)
            centers = torch.rand(4, 2, generator=bs) * 200.0
            sizes = torch.rand(4, 2, generator=bs) * 40.0 + 10.0
            boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=-1)
            rels = [{"subject_id": a, "object_id": b,
                     "predicate": PREDS_CYCLE[(i + j) % len(PREDS_CYCLE)]}
                    for j, (a, b) in enumerate([(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])]
            fh.write(json.dumps({"obj_boxes": boxes.tolist(), "relationships": rels}) + "\n")

    return {"dump": dump, "dump2": dump2, "prior": prior, "train": train, "dir": td}


# ------------------------------------------------------- 1. architecture is the registered one
def test_mlp_is_exactly_the_preregistered_architecture():
    net = make_mlp(768, 51)
    assert isinstance(net[0], torch.nn.Linear) and net[0].in_features == 768 and net[0].out_features == 768
    assert isinstance(net[1], torch.nn.GELU), "preregistration fixes GELU, not ReLU (p37 used ReLU)"
    assert isinstance(net[2], torch.nn.Linear) and net[2].in_features == 768 and net[2].out_features == 51
    assert len(net) == 3, "two Linear layers and one activation -- not p37's 3-layer MLP"
    assert sum(p.numel() for p in net.parameters()) == 629_811


# ------------------------------------------------------- 2. cross-fitting is honest
def test_xfit_ridge_is_out_of_fold():
    """A row's own score must not be produced by a model that trained on it.
    Constructed so a within-fold fit would be near-perfect and an honest
    out-of-fold fit cannot be."""
    g = torch.Generator().manual_seed(0)
    n, d_in, C = 200, 5, 4
    X = torch.randn(n, d_in, generator=g)
    y = torch.randint(0, C, (n,), generator=g)          # labels are pure noise
    fold = torch.arange(n) % 5
    out = xfit_ridge(X, y, fold, C)
    acc = float((out.argmax(-1) == y).float().mean())
    assert acc < 0.5, f"out-of-fold accuracy {acc} on random labels implies leakage"


def test_xfit_ridge_is_deterministic():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(120, 6, generator=g)
    y = torch.randint(0, 3, (120,), generator=g)
    fold = torch.arange(120) % 5
    a = xfit_ridge(X, y, fold, 3)
    b = xfit_ridge(X, y, fold, 3)
    assert torch.equal(a, b)


def test_xfit_mlp_is_seed_deterministic():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(160, 8, generator=g)
    y = torch.randint(0, 4, (160,), generator=g)
    fold = torch.arange(160) % 5
    a, pa = xfit_mlp(X, y, fold, 4, epochs=1, seed=0)
    b, pb = xfit_mlp(X, y, fold, 4, epochs=1, seed=0)
    assert torch.equal(a, b)
    assert pa == pb


# ------------------------------------------------------- 3. folds are image-level
def test_folds_are_image_level(synth):
    from tools.representation_decoder_ladder import MECH, WPD
    B = MECH.Mech(str(synth["dump"]), str(synth["prior"]), "raw50")
    fold = folds_of(B)
    img_of_gt = B.gt_img
    # every GT row of a given image must carry the same fold id
    for img in img_of_gt.unique().tolist():
        f = fold[img_of_gt == img].unique()
        assert f.numel() == 1, f"image {img} straddles folds {f.tolist()}"


# ------------------------------------------------------- 4. paired_delta convention
def test_paired_delta_zero_for_identical():
    v = [0.5, 0.6, 0.55, 0.7]
    out = paired_delta(v, v, n_boot=200)
    assert out["delta"] == 0.0
    assert out["ci_excludes_zero"] is False


def test_paired_delta_positive_ci_for_uniform_shift():
    a = [0.5] * 60
    b = [0.6] * 60
    out = paired_delta(a, b, n_boot=500)
    assert out["delta"] == pytest.approx(0.1, abs=1e-9)
    assert out["ci_excludes_zero"] is True
    assert out["ci95"][0] > 0


# ------------------------------------------------------- 5. concentration detection
def test_cell_distribution_flags_single_cell_domination():
    a = [0.5] * 1000
    b = list(a)
    b[7] = 0.9                      # one cell carries the entire net gain
    out = cell_distribution(a, b, keys=[f"k{i}" for i in range(1000)])
    assert out["n_exactly_zero"] == 999
    assert out["n_positive"] == 1
    assert out["n_negative"] == 0
    assert out["top_movers"][0]["key"] == "k7"
    assert out["top1pct_share_of_net"] == pytest.approx(1.0, abs=1e-9)


def test_cell_distribution_broad_gain_is_not_concentrated():
    a = [0.5] * 1000
    b = [0.51] * 1000               # every cell moves equally
    out = cell_distribution(a, b)
    assert out["n_positive"] == 1000
    assert out["top1pct_share_of_net"] == pytest.approx(0.01, abs=1e-6)


# ------------------------------------------------------- 6. cell keys align with wprd values
def test_cell_keys_align_with_wprd_values(synth):
    from tools.representation_decoder_ladder import MECH, WPD
    B = MECH.Mech(str(synth["dump"]), str(synth["prior"]), "raw50")
    Gs = WPD.Groups(B)
    keys = cell_keys_for(B, Gs)
    r = WPD.wprd(Gs, B.fixed_ensemble(0.0)[B.gt_row], 64)
    assert len(keys) == len(r["_vals"]) == r["n_cells"]
    assert all("::" in k and "|" in k for k in keys)


# ------------------------------------------------------- 7. end to end
def test_run_ladder_end_to_end(synth):
    out_path = synth["dir"] / "ladder.json"
    res = run_ladder(str(synth["dump"]), str(synth["dump2"]), str(synth["prior"]),
                     str(synth["train"]), str(out_path), mlp_epochs=1)

    # every registered arm present
    for arm in ("A1_frozen_baseline", "A2_linear", "A3_mlp", "A4_cosine_recomputed",
                "A5a_geometry_xfit", "A5b_geometry_trainfit", "A6_fusion",
                "N1_shuffled_label_null", "N2_prior_control"):
        assert arm in res["arms"], f"missing arm {arm}"
        assert 0.0 <= res["arms"][arm]["wprd_macro"] <= 1.0

    gate_names = {g["gate"].split()[0] for g in res["gates"]}
    for gname in ("G-A", "G-B", "G-C", "G-D", "G-E", "G-G"):
        assert gname in gate_names, f"gate {gname} not evaluated"

    by_name = {g["gate"].split()[0]: g for g in res["gates"]}
    # deterministic-by-construction gates must actually pass on this fixture
    assert by_name["G-A"]["pass"], "prior must be constant within group by construction"
    assert by_name["G-C"]["pass"], "text_logits were built from rel_feat; must reconstruct"
    assert by_name["G-D"]["pass"], "the two dumps are byte copies; rel_feat must match"
    assert by_name["G-E"]["pass"], "prior control must read exactly 0.5"
    assert by_name["G-G"]["pass"], "all arms must score the same cells"

    for k in ("P_star_arm", "P_star", "G_train_fitted_geometry", "vs_geometry",
              "outcome_A_decoder_ceiling_near_A1", "outcome_B_decoder_materially_exceeds_A1",
              "outcome_C_fusion_materially_exceeds_both", "outcome_D_geometry_strongest"):
        assert k in res["classification"], f"missing classification field {k}"

    assert res["classification"]["vs_geometry"] in (
        "BEYOND_GEOMETRY", "GEOMETRY_EQUIVALENT", "BELOW_GEOMETRY")
    assert Path(out_path).exists()

    # contrasts vs both references exist and carry a distribution summary
    assert "A2_linear - A1_frozen_baseline" in res["contrasts"]
    assert "distribution" in res["contrasts"]["A2_linear - A1_frozen_baseline"]
    assert "A2_linear - A5b_geometry_trainfit" in res["contrasts"]


def test_run_ladder_does_not_write_outside_its_out_path(synth):
    """The ladder must never touch historical artifacts."""
    out_path = synth["dir"] / "ladder2.json"
    before = {p: p.stat().st_mtime for p in [synth["dump"], synth["dump2"],
                                              synth["prior"], synth["train"]]}
    run_ladder(str(synth["dump"]), str(synth["dump2"]), str(synth["prior"]),
               str(synth["train"]), str(out_path), mlp_epochs=1)
    for p, mt in before.items():
        assert p.stat().st_mtime == mt, f"{p} was modified"

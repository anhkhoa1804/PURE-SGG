"""Safe lookup helpers for the cached pair-level C1 dump.

The VG cache can contain repeated ordered object-index pairs when the source
annotation contains more than one relation for the same pair.  A lookup is
safe only when all cached pair-level channels are identical across those
slots.  Otherwise the caller must fail rather than guess a slot.
"""

from __future__ import annotations

from typing import Iterable

import torch


DEFAULT_IDENTITY_FIELDS = (
    "rel_feat",
    "text_logits",
    "model_logits",
    "prior_rows",
    "cls_logits",
)


def _same_value(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        return bool(torch.equal(a, b))
    return a == b


def cached_pair_slot(
    dump: dict,
    image_index: int,
    subject_index: int,
    object_index: int,
    *,
    identity_fields: Iterable[str] = DEFAULT_IDENTITY_FIELDS,
) -> int:
    """Resolve one cached slot or raise on genuine ambiguity.

    A duplicated pair is accepted only when every available identity field is
    exactly equal across all duplicate slots.  Requiring at least one identity
    field prevents a synthetic or incomplete dump from being silently treated
    as unambiguous.
    """
    pairs = dump["pairs"][image_index]
    hits = [
        j for j, pair in enumerate(pairs.tolist())
        if tuple(map(int, pair)) == (int(subject_index), int(object_index))
    ]
    if not hits:
        raise ValueError(
            f"cached pair is absent for image index {image_index}: "
            f"({subject_index}, {object_index})"
        )
    if len(hits) == 1:
        return hits[0]

    fields = [name for name in identity_fields if name in dump]
    if not fields:
        raise ValueError(
            f"duplicate cached pair slots {hits} have no identity fields; "
            "refusing to guess"
        )
    for field in fields:
        values = [dump[field][image_index][slot] for slot in hits]
        first = values[0]
        if not all(_same_value(first, value) for value in values[1:]):
            raise ValueError(
                f"ambiguous duplicate cached pair slots {hits} for image "
                f"index {image_index}, field={field}"
            )
    return hits[0]


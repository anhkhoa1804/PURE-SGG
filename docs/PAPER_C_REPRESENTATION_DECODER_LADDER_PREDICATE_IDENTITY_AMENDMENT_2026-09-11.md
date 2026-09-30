# Paper C — predicate identity clarification for the cached-dump ladder

**Status: additive clarification registered before the accepted ladder run.**
The original ladder preregistration, the historical ladder result, the
historical checkpoint manifest, and the 2026-09-11 ladder amendment remain
unchanged.

## Decision

The byte-level CRLF identity recorded in
`docs/HISTORICAL_CHECKPOINT_MANIFEST.md` is not established as the exact
`predicates.json` byte artifact used to produce the cached C1/R0 and matched
R2 dumps. The current LF file is retained. The historical CRLF record remains
immutable and is not substituted into the current dataset.

For this ladder, the scientifically operative vocabulary identity is the
semantic canonical predicate object and its ordered mapping in the cached
dumps. The ladder reads cached `rel_feat`, logits, labels, and `pred_vocab`;
it does not regenerate model outputs from `predicates.json`. The C1/R0 and
matched R2 dumps have the same canonical 50 foreground predicates in the same
order, with synthetic `relation` at decoder column 50. The LF and CRLF files
parse to that same object and differ only in line endings.

Consequently, accepting the current LF artifact for the cached-dump ladder
does not change any tensor, label, fold, estimator, bootstrap, threshold, or
scientific population. It resolves a byte-provenance ambiguity that is
non-operative for the cached inputs while requiring the accepted run to
record the actual LF path, size, and SHA256. The run must not claim that the
LF bytes are the historical CRLF bytes.

## Evidence boundary

Established directly: cached predicate semantics/order, foreground/background
mapping, population identity, and byte-level tensor identity between C1/R0 and
matched R2.

Not established: the raw line-ending representation used by the original C1
execution. That uncertainty is retained in the accepted provenance rather
than silently resolved in favor of the historical hash.

## Scope

This clarification changes no decoder architecture, target mapping, geometry
features, fold construction, seed, optimizer, WPRD estimator, bootstrap,
materiality threshold, null, or output path. It applies only to the
cache-based representation ladder and does not amend the historical dataset
or checkpoint manifests.

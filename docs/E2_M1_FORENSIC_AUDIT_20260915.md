# E2 M1 cached-readout forensic audit

M1 is explicitly `M1_cached_C1`. It reads the accepted C1 pair dump, locates
the stored ordered subject/object pair for each candidate image, and uses the
exact cached predicate vocabulary index for relation A/B. The dump contains
duplicate ordered pair keys in some images because multiple annotated
relations can share an object pair. The scorer accepts such a duplicate only
when `rel_feat`, text, model, prior, and classifier channels are exactly
identical across all matching slots; a missing pair or non-identical duplicate
is a hard error. The audit verified 4,014 duplicate pair keys and zero
non-identical duplicate channels, including all 90 frozen candidate sides.
It does not load or substitute the demo checkpoint and does not perform fresh
image inference.

The cached dump SHA256 is
`2543c87512ba6b85e4bc8a629f4617ffd0fe57fccafce91325f636e0a34a4453`.
This is sufficient for a descriptive current development readout. It does not
restore the missing fresh C1 checkpoint lineage.

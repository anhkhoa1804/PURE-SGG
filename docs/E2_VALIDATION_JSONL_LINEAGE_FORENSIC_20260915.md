# E2 validation JSONL lineage forensic audit

`datasets_vg150_clean/validation.jsonl` is absent locally. Repository manifests
and historical documents record expected historical hashes, but no matching
file is present in accessible project storage. It cannot be reconstructed from
the cached pair dump with byte identity, so no substitute is labeled historical.

The records are themselves inconsistent: the post-B research note records the
short hash prefix `74d99779`, while the checkpoint manifest and project status
record full hash
`4348ddbb3ce85160d0ebc7522634c68f197b0c99906794e0e4731156740f3412`.
Because the bytes are absent, this audit cannot adjudicate which historical
representation was actually used. Both claims are preserved as provenance
conflicts rather than silently resolved.

For current E2 development, the frozen candidate packet stores the image IDs,
ordered object instances, boxes, relations, and selected image files needed for
annotation and block construction. The missing validation JSONL therefore does
not block human annotation or packet-level development checks. It remains a
publication-grade lineage gap and must be restored or explicitly archived
before claiming a fully reconstructible final dataset release.

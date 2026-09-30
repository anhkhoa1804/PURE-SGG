# Annotation independence audit

Status: **PASS for application-level isolation; human input pending**.

Each server process is configured for one annotator ID and filters both task
lists and saved answers to that ID. The browser state builder exposes only
packet-relative media URLs and the current annotator's saved answers; host
filesystem paths and other annotator files are excluded. Synthetic tests cover
cross-session answer visibility and missing/incomplete state.

This is application isolation, not an operating-system security boundary:
three annotators running under the same Unix account could read the packet
directory directly. Human sessions must therefore use the local interface and
must not share the annotation files manually before adjudication.

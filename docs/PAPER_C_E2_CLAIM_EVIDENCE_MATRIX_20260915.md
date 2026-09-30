# Paper C E2 claim-evidence matrix

| Claim | Required evidence | Current evidence | Allowed wording | Forbidden overclaim |
|---|---|---|---|---|
| closed-set relational decodability | accepted ladder | A3 exceeds A1 on cached population | supervised relation signal is decodable | semantic understanding |
| E2 held-out representation probe | cached C1 `rel_feat` with all frozen E2 images excluded before fitting | `M2_valheldout` fit artifact exists; human-gold scoring pending | a fresh closed-set decoder can be tested on the frozen E2 image population under validation-held-out lineage | train-split generalization, semantic transfer, open-vocabulary understanding |
| geometry competitiveness | A5a/A5b and E2 M3 | historical ladder; E2 M3 not scored | geometry is a competitive nuisance explanation | geometry removed |
| E2 human solvability | 3-annotator truth and block task | pending | measurement task is human-solvable if gates pass | human success proves machine grounding |
| E2 model-vs-geometry | frozen M3 and focal model on accepted blocks | not yet available | conditional performance beyond measured geometry baseline | causal semantic gain |
| E2 model-vs-object | frozen M4 and focal model | not yet available | performance beyond declared object nuisance baseline | object priors eliminated |
| task-seen discrimination | lineage and accepted E2 result | not yet available | task-seen controlled relation discrimination | unseen/open-vocabulary transfer |
| semantic interpretation | transfer/paraphrase/controlled evidence | not yet available | controlled image-conditioned relational discrimination | absolute semanticity |
| causal interpretation | intervention/randomization beyond matching | absent | predictive controlled comparison | causal effect or invariance |
| open vocabulary | task-unseen concepts and aliases | absent | none currently | open-vocabulary understanding |
| task-unseen transfer | clean lineage E3 | absent | none currently | unseen predicate generalization |

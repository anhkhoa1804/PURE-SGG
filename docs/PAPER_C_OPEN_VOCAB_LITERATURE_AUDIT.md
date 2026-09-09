# Paper C — Open-vocabulary SGG literature audit

Literature audit only. No source code was modified, no GPU job was launched,
no experiment was changed. Conducted via live web search in this session
(current date: September 2026); nothing below is recalled from training data
alone without a live source check, and every claim is tagged with how firmly
it is grounded.

**Evidence labels:** `[SOURCED]` = confirmed via a fetched abstract/page this
session; `[SOURCED-SECONDARY]` = confirmed only via a search-engine summary
of the paper (title/venue/authors verified, but architectural/numeric
details are second-hand, not read directly from the primary text — several
full-text PDF fetches failed to extract readable text, noted per-entry);
`[NOT FOUND]` = searched for, not located; `[INFERENCE]` = my own reasoning
connecting sourced facts, not itself a citation.

**Audit limitations, stated up front:** (1) Several papers' full PDFs did
not extract to readable text through the fetch tool (raw PDF object streams
came back instead of text) — those entries rely on abstract-page summaries
only, marked `[SOURCED-SECONDARY]`, and their quantitative results/exact
architectural wiring should be re-verified against the actual PDF before
being cited in any paper draft. (2) This audit is not exhaustive — it covers
every method the task named plus the most relevant items surfaced by
targeted searches, not a systematic literature review (no fixed search
protocol, no de-duplication against a citation database, no manual check of
every method's own related-work section). (3) Several sourced papers are
extremely recent (2025-2026, some past this model's knowledge cutoff) and
were found only through live search; they could not be cross-checked
against prior internal knowledge the way an older, well-known paper could.

---

## 1. Named systems (as requested by the task)

### 1.1 "From Pixels to Graphs" = "Pix2Graphs" = Pix2Grp (same work)

`[SOURCED]` — the task named these as if they might be two systems; they are
one. "Pixels to Graphs" and "Pix2Graphs" both refer to the same paper, whose
released codebase is named `Pix2Grp`.

- **Title**: *From Pixels to Graphs: Open-Vocabulary Scene Graph Generation
  with Vision-Language Models*
- **Authors**: Rongjie Li, Songyang Zhang, Dahua Lin, Kai Chen, Xuming He
- **Year / venue**: 2024, **CVPR 2024** (arXiv:2404.00906)
- **Architecture**: image-to-graph generation via a VLM (dubbed **PGSG**):
  the model generates a scene-graph **sequence** through image-to-text
  generation, then parses that sequence into a graph (entities + relation
  triplets), rather than scoring a fixed predicate set with a classifier
  head.
- **Predicate vocabulary mechanism**: generative/sequence-based, not a
  cosine-similarity readout against a frozen text bank. The predicate
  "vocabulary" is whatever the VLM's language decoder can produce as text,
  which is open-ended by construction of the generation task itself.
- **Genuinely unseen predicates during supervision?** `[SOURCED-SECONDARY,
  uncertain]` — the abstract frames the goal as handling "novel visual
  relation concepts" but the fetched content did not confirm a formal
  held-out-predicate training protocol (as opposed to leveraging the VLM's
  own pretraining corpus, which likely already contains many relation words
  in captions). Different supervision paradigm from our design: this method
  does not train a closed 50-class predicate classifier at all, so "held out
  from supervision" is not quite the right question to ask of it — it is
  closer to "the VLM was never told VG150's predicate list was fixed."
- **Arbitrary text predicates queryable at inference without retraining?**
  Yes, in the sense that the generative decoder can emit any predicate
  phrase — this is a fundamentally different mechanism from a dual-encoder
  cosine-scoring approach like ours (§3 of the design doc).
- **Evaluation / datasets**: Visual Genome-based open-vocabulary SGG
  benchmarks `[SOURCED-SECONDARY — exact split/table not extracted]`.
- **Strongest relevant result**: not extracted with confidence from this
  session's fetch (PDF fetch attempts returned incomplete text). **Needs
  direct re-verification from the PDF before citing any number.**
- **Limitation relative to our direction**: a generative sequence-to-graph
  approach is architecturally unrelated to a dual-encoder
  compatibility-score readout on a frozen relational representation; it does
  not isolate "does the visual relation representation itself carry
  transferable semantic content" the way a frozen-encoder, retrained-
  projection-only ablation would, because the VLM's language model is doing
  a large share of the generalization work.

### 1.2 OV-SGT (Open Vocabulary Semantic Graph Transformer)

`[SOURCED-SECONDARY]`
- **Title**: *OV-SGT: Open Vocabulary Semantic Graph Transformer for Scene
  Graph Generation*
- **Authors**: George Vanica, Adrian Bors
- **Year / venue**: **WACV 2026 Workshops** (SG4SI workshop), i.e. published
  after this model's training cutoff — found live via this session's search,
  not prior knowledge.
- **Architecture**: learns relationship embeddings **within CLIP's semantic
  space**; a node–edge fusion strategy preserving relationship
  directionality; graph-Laplacian-eigenvector positional encoding for
  structural context.
- **Predicate vocabulary mechanism**: `[SOURCED-SECONDARY]` explicitly
  described as learning relationship embeddings in CLIP's space "enabling
  **zero-shot generalization to unseen predicates**," trained with a
  **multi-component loss combining contrastive, semantic, triplet, and
  focal objectives**.
- **Genuinely unseen predicates during supervision?** Framed as yes by the
  secondary summary ("zero-shot generalization to unseen predicates" via a
  **contrastive** objective), but this session could not fetch the primary
  PDF to verify the exact split protocol, leakage controls, or whether
  "unseen" here means unseen predicate *class* or unseen predicate
  *combination* (§4 below explains why this distinction is usually
  conflated in this literature). **This is the single most important gap in
  this audit** — see the novelty verdict.
- **Arbitrary text predicates queryable?** `[INFERENCE]` — likely yes, since
  the scoring mechanism is described as embedding-space-based (CLIP
  semantic space), not a fixed classifier, but not directly confirmed.
- **Evaluation / datasets / strongest result**: `[SOURCED-SECONDARY]` only —
  "competitive performance in both Recall@K and Mean Recall@K," no numbers
  extracted.
- **Limitation relative to our direction, and why this is the critical
  finding of this audit**: OV-SGT's description — *"learns relationship
  embeddings within CLIP's semantic space... multi-component loss combining
  contrastive, semantic, triplet... objectives for zero-shot transfer"* — is
  **architecturally and objective-wise very close to the core mechanism**
  proposed in `docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md` (a relation
  representation projected and contrastively aligned to frozen CLIP text
  embeddings, scored via compatibility, evaluated for unseen-predicate
  transfer). **This session could not access OV-SGT's full text**, so the
  degree of overlap with our proposed §1-§2 design (temperature-scaled
  symmetric contrastive loss, hard-negative margin from semantic groups,
  frozen backbone, cosine scoring) cannot be assessed precisely — only that
  the high-level pattern is not obviously novel and a workshop paper doing
  something similar already exists, as of a few months before this audit.
  **Any future novelty claim for our design must first read this paper's
  full text and diff against it directly.**

### 1.3 SDSGG

`[SOURCED]`
- **Title**: *Scene Graph Generation with Role-Playing Large Language
  Models*
- **Authors**: Guikun Chen, Jin Li, Wenguan Wang
- **Year / venue**: **NeurIPS 2024** (arXiv:2410.15364; code:
  `github.com/guikunchen/SDSGG`)
- **Architecture**: an LLM is prompted to "role-play" multiple personas
  (e.g. biologist, engineer) to generate **scene-specific** descriptive text
  for a given image; these descriptions adaptively reweight per-predicate
  **text classifiers** (CLIP text embeddings used as zero-shot classifier
  weights, à la CLIP zero-shot image classification) via a "renormalization
  mechanism," rather than using one static prompt template per predicate
  class for every image. A "mutual visual adapter" refines CLIP's
  subject-object interaction features.
- **Predicate vocabulary mechanism**: CLIP text embeddings as classifier
  weights (standard CLIP zero-shot-classification pattern), but made
  **scene-conditional** (the weights shift per image based on LLM-generated
  scene description) rather than fixed per predicate globally.
- **Genuinely unseen predicates during supervision?** `[SOURCED-SECONDARY]`
  — the paper targets "open-vocabulary SGG" and explicitly critiques prior
  OVSGG work's text classifiers as "scene-agnostic... unchanged across
  contexts," implying the underlying zero-shot-classification mechanism
  (CLIP text embedding as classifier weight, no gradient update needed for
  a new predicate string) is inherited from the standard OVSGG pattern this
  whole line of work uses — not a training-time predicate-disjoint ablation
  with leakage controls.
- **Arbitrary text predicates queryable?** Yes, in the same sense as any
  CLIP-zero-shot-classifier design (a new predicate word/phrase just needs a
  new text embedding) — but the SDSGG-specific innovation (LLM
  role-playing, scene-conditional reweighting) is a *different* mechanism
  from what our design targets (learning where the relation-space
  projection points, not scene-conditioning the text side).
- **Evaluation / datasets**: `[SOURCED-SECONDARY]` "prevalent benchmarks,"
  not itemized in the fetched content.
- **Limitation relative to our direction**: SDSGG's central contribution is
  making the *text/prompt* side scene-adaptive via LLM role-play, not
  training a projection/contrastive alignment on the *visual relation*
  side — a different lever than the one our design pulls (§1 of the design
  doc: `f(r)`, the relation-side projection). Its reliance on an LLM at
  inference time is also a substantially heavier and less controlled
  pipeline than a frozen-CLIP dual-encoder score.

### 1.4 PRISM-0

`[SOURCED]`
- **Title**: *PRISM-0: A Predicate-Rich Scene Graph Generation Framework
  for Zero-Shot Open-Vocabulary Tasks*
- **Authors**: Abdelrahman Elskhawy, Mengze Li, Nassir Navab, Benjamin Busam
- **Year / venue**: arXiv preprint, submitted **April 2025**, revised
  November 2025 (arXiv:2504.00844) — no venue/peer-review status confirmed
  by this audit.
- **Architecture**: a **training-free, bottom-up, modular pipeline**
  chaining an object detector → VLM (produces a natural-language
  description of a detected object pair) → LLM (converts the description
  into fine- and coarse-grained predicate candidates) → VQA model
  (validates/filters the candidates). No predicate classifier or learned
  text-alignment head is trained at all.
- **Predicate vocabulary mechanism**: open-ended natural language generated
  by an LLM from a VLM description, then filtered by VQA — the "vocabulary"
  is whatever predicate phrases the LLM proposes, unconstrained by any fixed
  training-time class list. Reports discovering **132 distinct predicates**
  vs. "typically fewer than 50" in prior closed-vocabulary work.
- **Genuinely unseen predicates during supervision?** Not applicable in the
  usual sense — there is no supervised predicate-classification training
  step at all in this pipeline; every predicate is produced zero-shot by
  off-the-shelf foundation models. This sidesteps the seen/unseen
  supervision question entirely rather than answering it the way a trained
  contrastive-alignment method would.
- **Arbitrary text predicates queryable?** Trivially yes (nothing is
  trained against a fixed vocabulary), but for a different reason than our
  design (no training-time class list exists at all, vs. our design's
  frozen-encoder-plus-trained-projection that must be shown to generalize
  beyond its training set).
- **Evaluation / datasets**: Visual Genome and a Sentence-to-Graph Retrieval
  benchmark `[SOURCED-SECONDARY]`; claims parity with weakly-supervised and
  some supervised methods on specific tasks; reports **+12% predicate
  classification / +8% scene graph detection** over prior zero-shot SGG
  methods on Visual Genome `[SOURCED-SECONDARY, numbers not independently
  re-verified from primary text]`.
- **Limitation relative to our direction**: this is not a representation-
  learning or contrastive-alignment method at all — it is an inference-time
  foundation-model orchestration pipeline. It does not test whether a
  *specific, geometry-repaired relational feature* (our `r`) carries
  transferable semantic content; it bypasses that question by using
  general-purpose VLM/LLM/VQA reasoning per pair. Not directly comparable to
  our architecture, but directly relevant as an alternative philosophy for
  "open vocabulary" that requires no predicate-disjoint training protocol at
  all — worth naming as a different branch of the field in any related-work
  section.

### 1.5 The OvSGTR family (adjacent to, but not named exactly as, the task's list)

`[SOURCED-SECONDARY]` — surfaced while searching for "OV-SGT"; likely
related to, but a **different, earlier and more established** line of work
than §1.2's WACV-2026-workshop OV-SGT. Two papers found, probably by the
same core author group, describing what search summaries called "OvSGTR":

- *Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph
  Generation via Visual-Concept Alignment and Retention* — Zuyao Chen,
  Jinlin Wu, Zhen Lei, Zhaoxiang Zhang, Changwen Chen. arXiv:2311.10988,
  submitted Nov 2023, revised Oct 2024. `[SOURCED]`
- *From Data to Modeling: Fully Open-vocabulary Scene Graph Generation* —
  arXiv:2505.20106 (2025) `[SOURCED-SECONDARY only, not fetched directly
  this session — inferred to be a related/extended work from the same
  OvSGTR line based on search-summary content overlap, not confirmed by
  reading both papers' author lists side by side]`.
- **Architecture** (2311.10988): DETR-like end-to-end transformer, **frozen
  image backbone and text encoder**, visual-concept alignment for both
  object nodes and relation edges, a **relation-aware pretraining strategy**
  using image-caption data (weak supervision, no manual relation
  annotation), and a **knowledge-distillation-based visual-concept
  retention** mechanism to fight catastrophic forgetting when adapting to
  open-vocabulary settings.
- **Predicate vocabulary mechanism**: text-encoder-based (open-ended by
  construction), pretrained on caption-derived weak relation supervision
  rather than a closed 50-class label set.
- **Genuinely unseen predicates during supervision?** `[SOURCED-SECONDARY,
  unresolved]` — not confirmed whether VG150's specific 50-predicate
  vocabulary is ever partitioned into a genuinely-never-supervised held-out
  set for evaluation, or whether "open vocabulary" here is validated
  primarily through the caption-pretraining-then-VG150-finetuning pipeline
  (i.e., broad but uncontrolled exposure to relation language via captions,
  not a clean predicate-disjoint ablation).
- **Evaluation**: VG150-based, across closed-set / OV-object-detection-based
  / OV-relation-based / fully-OV settings, reported as SOTA across all four
  `[SOURCED-SECONDARY, exact numbers not extracted]`.
- **Limitation relative to our direction**: this is the closest thing found
  in this audit to an established, multi-setting open-vocabulary SGG
  benchmark suite — but its "open vocabulary" claim rests on **caption-
  derived weak supervision breadth**, not on a controlled ablation isolating
  whether a *frozen, unmodified relational representation* generalizes to
  predicates excluded by construction from any supervision signal
  whatsoever (weak or strong). This is architecturally the single most
  important prior-art candidate to read in full before finalizing our own
  predicate-disjoint protocol's novelty claim (§3 of the design doc).

---

## 2. Additional modern OV / zero-shot SGG methods (lighter coverage)

- **"Towards Open-vocabulary Scene Graph Generation with Prompt-based
  Finetuning"** — Tao He, Lianli Gao, Jingkuan Song, Yuan-Fang Li,
  **2022**, arXiv:2208.08165. `[SOURCED]` Two-stage: pretrain on
  region-caption data, then prompt-based parameter-efficient finetuning
  (no backbone weight updates). Framed explicitly around **unseen object
  classes** ("trained on a set of base object classes... infer relations
  for unseen target object classes"). **Important nuance**: the sourced
  summary centers on open-vocabulary *objects*; it is not clear from this
  audit whether predicates are held out under the same discipline, or
  whether predicates are treated as closed-set throughout with only the
  object side made open-vocabulary. **Needs a direct read before being
  cited as predicate-side prior art.**
- **"Open-Vocabulary Scene Graph Generation via Synonym-Based Predicate
  Descriptor"** — 2025 (MultiMedia Modeling / Springer).
  `[SOURCED-SECONDARY]` Uses a "Predicate Descriptor" = mean-pooled,
  synonym-augmented text embeddings, plus "Foreground Relation Sampling,"
  to improve zero-shot predicate capability and address class imbalance
  ("class-balance-aware" per one summary). Directly relevant as a sibling
  approach to our design's `e(p)` construction — worth a full read for
  comparison on how it handles synonym leakage (our design's §3.4 concern),
  since a synonym-augmented descriptor could either help or accidentally
  leak information between held-out and seen predicates depending on how
  synonym sets are built.
- **"Zero-Shot Scene Graph Generation via Triplet Calibration and
  Reduction" (T-CAR)** — jkli1998 et al., **ACM TOMM 2023**
  (arXiv:2309.03542, code: `github.com/jkli1998/T-CAR`). `[SOURCED]`
  Confirms the field's **dominant zero-shot definition**: "unseen
  triplets" = novel `(subject, predicate, object)` **combinations** built
  from individually-seen categories, explicitly stated as "several times
  larger than the seen space... a huge number of unrealistic
  compositions" — i.e. the generalization target is compositional recombination, not a predicate class that was itself never supervised. See §4 below — this is the central, load-bearing distinction for this entire audit.
- **"Interaction-Centric Knowledge Infusion and Transfer for
  Open-Vocabulary Scene Graph Generation"** — arXiv:2511.05935 (Nov 2025).
  `[NOT FOUND beyond title]` — identified but not read this session; the
  title suggests direct relevance (predicate/"interaction" transfer for
  OVSGG) and should be read before any novelty claim.
- USRL and ZS-BUS (Unseen Space Reasoning / Hybrid-Attention-Cross-Network
  with Unseen Space Optimization) — `[SOURCED-SECONDARY, titles/mechanism
  sketch only]` both explicitly frame "unseen" as **unseen triplet
  combinations**, reachable via positive-unlabeled learning or learned
  filtering of an implausible-combination space — same compositional
  paradigm as T-CAR, not a predicate-class-disjoint paradigm.

---

## 3. Older zero-shot SGG / semantic-embedding methods

- **"Visual Relationship Detection with Language Priors"** — Cewu Lu,
  Ranjay Krishna, Michael Bernstein, Li Fei-Fei, **ECCV 2016**
  (arXiv:1608.00187). `[SOURCED]` **This is the origin of the field's
  standard "zero-shot visual relationship" evaluation convention** still in
  use today (T-CAR, USRL, ZS-BUS, and essentially every "zR@K"-reporting
  paper trace back to this protocol): individually-seen subject/predicate/
  object categories, recombined into triplets never observed together at
  training time. Trains separate visual detectors for objects and a
  predicate module, then rescales predicted relationship likelihood using
  **word-embedding-based language priors** (project each candidate triplet
  through a language/semantic-embedding space to favor plausible
  combinations). Reports a dramatic zero-shot jump from a visual-only model
  (**1.85% R@100**) to visual+language-prior (**14.70% R@100**) —
  demonstrating decades-old, foundational evidence that a language/semantic
  embedding space carries transferable relational structure beyond what a
  purely visual model captures. **Directly relevant precedent** for the
  general thesis of our design ("semantic embeddings help visual relation
  models generalize") but at a **coarser granularity** (combination-level
  plausibility reweighting via static word embeddings, not a trained,
  temperature-scaled contrastive alignment of a learned relation feature to
  a modern VLM's text space).
- **"Weakly-Supervised Learning of Visual Relations"** — Julia Peyre, Ivan
  Laptev, Cordelia Schmid, Josef Sivic, **ICCV 2017**.
  `[SOURCED]` Discriminative-clustering model learning visual relation
  detectors from **image-level labels only** (no box-level relation
  annotation), with an explicit analogy/compositional-transfer mechanism
  that "significantly improves performance on previously unseen relations."
  Introduces the **UnRel** dataset (unusual/rare relations) for evaluating
  this transfer. Relevant precedent for "compose a rare/unseen relation
  from visual+semantic cues shared with seen ones," conceptually adjacent
  to our design's semantic-group hard-negative structure (§2 of the design
  doc), though the mechanism (weak supervision + discriminative clustering)
  is unrelated to CLIP-style contrastive alignment (CLIP did not exist in
  2017).

---

## 4. The single most important distinction this audit surfaces

Every "zero-shot" or "open-vocabulary" SGG paper found in this audit that
reports a `zR@K`-style number is, by the field's own dominant convention
(traced to Lu et al. 2016, restated explicitly by T-CAR 2023), testing
**unseen *combinations* of individually-seen categories** — the predicate
itself was supervised during training on other (subject, object) pairs; only
the specific triplet was withheld. This is a **materially weaker claim**
than what our Paper C design proposes: a predicate **class** that receives
**zero supervision of any kind** (no positive example, no negative-class
gradient, no synonym leakage) during training, scored purely from its CLIP
text embedding at inference.

`[INFERENCE]`, drawn from the sourced summaries above: among every method
audited, only two plausibly attempt something closer to genuine
predicate-class-level open-vocabulary generalization rather than
combination-level zero-shot: (a) **OV-SGT** (§1.2, WACV 2026 workshop,
explicit "contrastive... for zero-shot transfer" language, but unverified
protocol — full text not accessible this session), and (b) the **OvSGTR
family** (§1.5, caption-pretraining-derived open-ended relation vocabulary,
but validated through *broad weak-supervision exposure* rather than a
*clean predicate-disjoint ablation with leakage controls*). **Neither was
confirmed, from the sources actually read this session, to implement
anything resembling our design's specific anti-leakage protocol** (§3 of
the design doc: prompt-template parity, nearest-neighbor synonym audit,
near-duplicate co-assignment, seen-only vocabulary tensor construction) or
its **falsification battery** (shuffled-text control, random-text control,
geometry-only floor, untrained-`f` ablation, image-free nearest-neighbor
baseline). This absence should be read as "not found in this audit," not as
proof that no such protocol exists anywhere in the literature.

**Direct answer to the task's explicit question** — "does using CLIP text
embeddings already make a method open-vocabulary?": **No, not by itself,
and this is well illustrated by the audited set.** Every single method
audited in §1-§2 (SDSGG, OV-SGT, OvSGTR, the synonym-descriptor method, even
the non-CLIP language-prior method from 2016) uses a semantic/text embedding
space as **initialization or scoring substrate** for predicates. What varies
— and what actually determines whether "open-vocabulary" is a genuine claim
or a semantic-initialization claim — is whether the method's evaluation
protocol ever tests a predicate that received **zero training-time
supervision of any kind**, under leakage controls, versus testing
combination-level recombination of already-supervised predicates (the
overwhelmingly more common, and structurally easier, test). Readout v2 in
this repository (`predicate_prototypes`, closed 51-row matrix,
CLIP-initialized) is a clean illustration of the failure mode from the other
direction: CLIP text embeddings as **initialization** for a fixed-cardinality
learned matrix is not open-vocabulary at all, regardless of the semantic
initialization, because there is no mechanism to score a 52nd predicate
without adding and training a new row (already stated correctly in
`docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md` §1/§8 — this audit corroborates
rather than revises that conclusion).

---

## 5. Frequency-bias / TDE / causal SGG work

- **"Neural Motifs: Scene Graph Parsing with Global Context"** — Rowan
  Zellers, Mark Yatskar, Sam Thomson, Yejin Choi, **CVPR 2018**.
  `[SOURCED]` Origin of the ubiquitous **frequency-prior baseline**
  (predict the most frequent predicate for a given entity-class pair,
  from training-set statistics alone) — this repository's own
  `frequency_prior_train.json` / `freq_bias_*` machinery and its
  `prior_control_wprd` diagnostic are direct descendants of this baseline
  convention. Reports the frequency-only baseline (`FREQ+OVERLAP`) beating
  the prior state of the art by 1.4 mean recall at the time — the
  foundational demonstration that naive predicate models are dominated by
  pair-conditional frequency statistics, which is exactly the confound
  WPRD's `prior_control_wprd = 0.5` design targets.
- **"Unbiased Scene Graph Generation from Biased Training" (TDE)** —
  Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, Hanwang Zhang,
  **CVPR 2020**. `[SOURCED]` Introduces **counterfactual causal
  inference** for SGG debiasing: run the model twice (factual and an
  intervened/counterfactual pass with the biasing input, e.g. context,
  ablated) and use the **Total Direct Effect** (factual minus
  counterfactual) as the final predicate score, explicitly removing the
  "bad bias" (frequency/context shortcut) from the prediction. This is the
  field's canonical reference for **causal frequency-bias control** —
  directly relevant background for our design's insistence (§2, §4 of the
  design doc) that WPRD's prior control must read exactly 0.5 and that no
  training loss may be prior-contaminated. TDE's mechanism (double
  forward pass, subtract a counterfactual branch) is a **different
  technique** from WPRD's (arithmetic cancellation via a fixed
  cell-population/candidate-set construction, no counterfactual forward
  pass) — both target the same confound, by different means.
- Several later debiasing lines were surfaced but not deeply audited this
  session (`Compositional Feature Augmentation for Unbiased SGG`,
  `Probabilistic Debiasing of Scene Graphs`, `Unbiased SGG using Predicate
  Similarities`) — flagged as existing but not read; a full accounting of
  the causal/debiasing sub-literature was not attempted given the task's
  primary focus on open-vocabulary generalization.

---

## 6. WPRD-adjacent diagnostics — is there a direct precedent?

**Direct answer: no exact precedent was found in this audit** for WPRD's
specific combination of (a) a fixed cell population defined by exact
ground-truth-pair identity, (b) macro-averaging over those cells with a
capped per-cell candidate count, and (c) an arithmetic guarantee that a
prior-only control collapses to exactly `0.5` by construction. This is a
"not found," not a "does not exist" — the audit's scope and depth are
limited (§0).

The **closest related, but structurally different**, precedent found:

- **"Rethinking the Evaluation of Unbiased Scene Graph Generation"** — Li,
  Chen, et al., **BMVC 2022** (arXiv:2208.01909). `[SOURCED-SECONDARY —
  primary PDF did not extract to readable text this session; summary drawn
  from search-engine synthesis, re-verify before citing precisely]`.
  Identifies two flaws in ordinary mean Recall@K: it (1) breaks
  **category independence** by jointly ranking all triplet predictions
  across every predicate category together, and (2) over-weights
  "oversimple" high-compositional-diversity predicate categories. Proposes
  **Independent Mean Recall (IMR)** and **weighted IMR (wIMR)**, which
  rank/score **within each predicate category independently** rather than
  in one joint softmax-style ranking across all categories, to fix the
  multi-label ambiguity inherent in scoring one subject-object pair against
  many plausible predicates.
- **How this differs from WPRD**: IMR/wIMR restructure **global** ranking
  statistics (across the whole dataset, per-category independence) to fix a
  labeling-fairness problem in the standard mR@K pipeline. WPRD instead
  defines a **local, per-pair diagnostic cell** (does the model discriminate
  the *correct* predicate from same-pair alternates, for *this exact*
  ground-truth pair) and aggregates those cells with a **built-in null
  baseline that is provably prior-free by construction** (`0.5` exactly).
  These solve related but distinct problems: IMR/wIMR is about **fair
  cross-category ranking at the whole-dataset level**; WPRD is about
  **isolating a per-pair discrimination signal from the frequency prior**,
  closer in spirit to TDE's causal-control goal (§5) than to IMR/wIMR's
  ranking-fairness goal, but achieved by a wholly different (non-
  counterfactual, cell-arithmetic) mechanism. **No source found in this
  audit combines these three specific properties (pair-exact cell identity
  + capped macro aggregation + provably-prior-free arithmetic null) in one
  named metric.**

---

## NOVELTY VERDICT

### A. Established prior art (safe to assume exists, cite accordingly)

1. Using semantic/word/text embeddings to help a visual relation model
   generalize beyond what it was directly trained on — **established since
   2016** (Lu et al.), refined continuously since (Peyre et al. 2017 through
   every audited 2024-2026 method).
2. Using **CLIP specifically** as a frozen text/vision encoder for
   open-vocabulary SGG, including as classifier weights (SDSGG), as an
   embedding space for a learned projection or prototype matrix (OV-SGT,
   this repository's own Readout v0/v2), and as one stage of a larger
   VLM/LLM pipeline (PRISM-0, Pix2Grp/PGSG) — **established, multiple
   independent instances, 2022-2026.**
3. The field's dominant "zero-shot SGG" evaluation convention (`zR@K`) tests
   **unseen triplet combinations of seen categories**, not unseen predicate
   classes — **established since Lu et al. 2016, still the norm through
   T-CAR 2023 and its contemporaries.**
4. Frequency-prior baselines and causal/counterfactual debiasing of SGG
   predicate scores — **established** (Zellers et al. 2018; Tang et al.
   2020, TDE), with a substantial subsequent literature this audit did not
   fully enumerate (§5).
5. A CLIP-space **contrastive** objective specifically for relation-to-text
   alignment aimed at zero-shot predicate transfer — **very likely already
   established** by OV-SGT (§1.2), though this audit could not verify its
   exact protocol from primary text.

### B. Apparent gaps (found, but not confirmed absent — genuinely uncertain)

1. No audited method was confirmed (from primary-text evidence actually
   read this session) to run a **strict predicate-disjoint split** with the
   specific leakage controls our design specifies — prompt-template parity,
   nearest-neighbor synonym audit, near-duplicate co-assignment, seen-only
   vocabulary tensor construction (§3 of the design doc). This is an
   apparent gap, not a confirmed one: OV-SGT's and OvSGTR's primary texts
   were not accessible this session, and either could already do this.
2. No audited method was confirmed to run WPRD's specific style of
   falsification battery (shuffled-text control, random-text control,
   geometry-only floor, untrained-projection ablation, image-free
   nearest-neighbor baseline) as a **pre-registered gate before claiming a
   result**, though individual pieces of this idea have clear ancestors
   (TDE's counterfactual-branch philosophy is conceptually close to a
   falsification-control mindset).
3. No exact precedent found for WPRD's specific pair-exact-cell,
   prior-provably-null mechanism (§6) — genuinely searched for and not
   located, though the search was not exhaustive.

### C. Genuinely unresolved questions (this audit cannot answer them)

1. Does OV-SGT's or OvSGTR's full text already implement a leakage-
   controlled predicate-disjoint protocol equivalent to or stronger than
   ours? **Unknown — requires reading the primary PDFs directly**, which
   this session's fetch tooling could not reliably extract.
2. Is there prior art for WPRD's exact mechanism under a different name,
   in a venue or sub-community this audit's search terms did not reach
   (e.g. a robotics, HOI-detection, or video-SGG paper using an analogous
   pair-conditioned diagnostic)? Not ruled out.
3. Would the specific combination of (PURE's pair-relative-geometry-repaired
   representation) × (a properly-trained, symmetric, temperature-scaled
   contrastive alignment) × (this exact falsification battery) outperform
   OV-SGT or OvSGTR on a shared, predicate-disjoint benchmark? **Cannot be
   answered by literature audit — requires a controlled experiment.**

### D. Possible novelty for this program's work (hedged, contingent)

Given A-C above, the **defensible, narrow** possible-novelty claims are:

- **Not novel**: "using CLIP text embeddings for open-vocabulary predicate
  scoring" (established many times over, §A2).
- **Not safely claimable without further reading**: "a contrastive
  relation-to-text alignment objective for zero-shot predicate transfer"
  (OV-SGT appears to already do something in this space, §A5) — **this
  claim must not appear in any external document until OV-SGT's full text
  has been read and diffed against our design.**
- **Potentially novel, contingent on the unresolved questions in §C**: the
  specific **combination** of (a) a rigorously-controlled predicate-disjoint
  split with the exact leakage checks specified in the design doc, (b) the
  specific falsification battery (especially the geometry-only floor,
  which — unlike anything found in this audit — directly separates
  "spatial-layout shortcut" from "genuine visual-semantic transfer," a
  distinction that matters specifically because this program's own `r` is
  built on a geometry-repaired representation whose weak-positive effect
  size is already precisely characterized), and (c) applying this to a
  representation whose geometry contract has itself been causally
  ablated and replicated twice (`docs/PAPER_C_C1_SEED2_RESULT.md`) — a
  provenance no other audited method shares, simply because no other
  audited method is built on this specific codebase's geometry-repair
  history.

### Experiments required before claiming any novelty externally

1. **Read OV-SGT's full paper** (not just the WACV listing/abstract) and
   produce a section-by-section diff against
   `docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md` §1-§2 — architecture,
   loss terms, and evaluation protocol specifically. This is the single
   highest-priority follow-up from this entire audit.
2. **Read the OvSGTR family's full papers** (arXiv:2311.10988 and
   arXiv:2505.20106) to determine whether their "fully open-vocabulary"
   relation setting already includes a clean predicate-disjoint ablation,
   or is validated only through caption-pretraining breadth.
3. **Run Stage A and Stage B of the experiment ladder** (design doc §6) and
   check whether the resulting `WPRD_unseen`, harmonic mean, and
   falsification-test results are directionally consistent with what a
   from-scratch read of OV-SGT/OvSGTR's own reported numbers would predict —
   if our falsification battery reveals a geometry-only floor or
   shuffled-text collapse that those papers do not report checking for,
   that specific methodological delta (not the base architecture) becomes
   the more defensible novelty claim.
4. Only after (1)-(3): draft any external-facing novelty statement, and
   have it independently checked against this document's §A/§B/§C before
   submission anywhere.

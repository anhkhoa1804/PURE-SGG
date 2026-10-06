# Results

## Predictive decoding and geometry

Recorded WPRD values are A1 .574988, A2 .580128, A3 .585669, A4 .575834, A5a .588144, A5b .596235, A6 .585503, N1 .505451, N2 .500000. A3 exceeds A1 by .0106807; geometry cross-fit exceeds A3 by .0024755 and geometry train-fit by .0105659. These latter differences are arithmetic comparisons of recorded means; no new paired CI is assigned to them. A5b changes fitting population but still evaluates held-out historical validation; it must not be mislabeled an in-sample validation score. The registered C0→C1 change was +.008261, CI [.003831,.012697], below the registered +.010 success target. This supports decodability and competitive geometry, with a qualified architecture improvement, rather than the stronger original target. [E1,E13]

## Held-out residual prediction

On fixed validation, calibrated O LL is 1.6323624327; O+rel_feat is 1.2663293005; delta −.3660331323, image CI [−.3966375880,−.3361649032]. O accuracy/macro recall are .6458541792/.2000605986; combined .6404509372/.1666924060. A reduction in probability loss does not require an increase in argmax accuracy or equal-class recall. Geometry, appearance and scene context remain alternatives to the O residual. [E2]

G+O calibrated LL is 1.3242384157 and the rel_feat combination 1.3191095828; delta −.0051288329, CI [−.0263599217,+.0158628067]. Standalone matched-scale rel_feat is worse at 1.4205308987. G+O accuracy/macro recall are .6561270095/.1218934710; combination .6200386899/.1540253465. The raw comparison worsens, 1.3279949152→1.4190692328, while calibrated loss improves slightly. Thus this particular G+O combination is calibration-sensitive and does not improve argmax accuracy. Earlier statements that both raw loss and accuracy improved for this full contrast are contradicted by the final saved results; the statement holds only for the small N4 pilot. [E2]

The magnitude of the O-relative increment contracts by roughly 98.6% in the G+O comparison. This descriptive calculation is about these estimators and calibrated combinations; it is not a decomposition of causal or mutual information. The G+O interval permits both a modest benefit and a modest loss. No equivalence claim is warranted. [E2]

## Richer nuisance pilot

On CAL-CHECK, N4-FIT calibrated LL is 1.5984459578; its rel_feat combination is 1.5959997348. Delta is −.0024462230, exploratory CI [−.0026268349,−.0022835433]. Alpha is .0057614225; N4/combined temperatures are 2.5614361/2.5656223. Raw LL changes 2.4195268202→2.4193531598 (−.0001736604); accuracy .5566518141→.5582431572 (+.0015913431); macro recall .1386469772→.1395603580. The bootstrap interval excludes zero conditional on this pilot, but does not include seed, model-selection, or fullscale-population uncertainty. A small alpha is not a measure of information, because logit scale, calibration, and collinearity affect it. [E4]

The absolute pilot change is approximately 47.7% of the full G+O reference magnitude. That comparison triggered the recorded resource decision, `PILOT_DOES_NOT_SUPPORT_FULLSCALE`; it is not a statistically matched contraction estimate because the populations, offset orientation, and nuisance/rel_feat fitting scales differ. Full 83,249-image N4-FIT and constructive residual C1 training were not executed. The pilot cannot prove an absence of residual beyond U/S. [E4]

## Sampling, seed and readout diagnostics

The common-validation M2 results are LL 1.662453/1.684642/1.692843/1.685949 for 2.5k/5k/10k/15k-prefix, versus 1.630454 balanced15k. Balanced−prefix delta −.0554957, image CI [−.0824371,−.0307560], while accuracy and macro recall decrease by .0032019 and .0120571. Training rows differ (188,736 prefix versus 206,669 balanced), and sampling was not randomized across replicated fits; no clean scaling or generalization claim follows. [E5]

Cross-seed cosine .940568 and sampled linear CKA .960323 indicate feature similarity. Separate cross-seed decoder transfer loses .019870 and .018640 accuracy, with intervals below zero in both directions. The representation similarity therefore does not guarantee readout invariance. The matched residual and intervention studies have no second-seed replication. Corrected R2c WPRD .573407 is below R0 .574988; no paired R2c CI was found. Historical R2's near-zero gain is fidelity-invalid as a trained-prototype result. [E6,E8]

FULL/A/B smoke M2 validation LL are 1.830500/1.825392/1.875606; native-head LL are 2.697031/2.715586/2.809679. Registered spatial M2 LL are 1.614668/1.635027/1.681200. Those observations are consistent with sensitivity to the intended operations but cannot identify relation semantics, independent nuisance control, or causal effects. No matched seed5678 smoke or B2 was run. [E7]

## Error and population limits

Saved per-predicate results are rendered without new grouping or fitting. “on”, “in”, “has” and “of” dominate total loss largely through prevalence. Rare predicates can have only 4–30 rows, so tail deltas are unstable. No robust tail advantage or interaction-specific residual is established. The whitelist-member predicate union used in some smoke descriptive tables is not the frozen pairwise interactional endpoint; the current residual validation has zero eligible rows for that endpoint. [E0,E7,E10]

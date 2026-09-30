# E2 M2 clean fitting protocol

M2 is a fresh closed-set decoder probe. For the amended E2 protocol it is
named `M2_valheldout`: its fitting data are the cached C1 validation
representation after excluding every image in the frozen E2 packet. This is a
declared representation artifact whose image IDs are disjoint from every E2
image, but it is not a canonical train-derived source. The artifact retains
`rel_feat`, ordered subject/object identity, predicate target, and source image
ID. Standardization, decoder parameters, hyperparameters, and any model
selection are fit only on that artifact.

The fitted decoder is frozen before it sees accepted E2 blocks. No E2 human
truth, block acceptance, nuisance score, or focal-model score may enter fitting
or selection. The implementation is the registered GELU MLP capacity used as
A3, with one declared seed and no architecture search. The exact dump and
packet hashes, excluded image-ID hash, row count, standardization, and fit
hyperparameters are recorded in the fit artifact. The model may support the
amended M2-versus-M3/M4 block comparisons, but it cannot support a claim of
train-split generalization, semantic transfer, or open-vocabulary understanding.

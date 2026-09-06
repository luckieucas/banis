# MitoEM 2.0 SDT--affinity graph-repair protocol

Protocol status: deterministic and learned-gate evaluation rules locked,
September 4, 2026.

## Purpose

This document freezes the protocol used to complete the deterministic BANIS-GR
baseline before developing a learned edge scorer. It prevents the remaining
validation work from silently changing the test evaluation.

## Dense predictor

- Model: BANIS with MedNeXt-B, kernel size 3.
- Outputs: three one-voxel affinities, three ten-voxel affinities, and one SDT.
- Checkpoint: `/projects/weilab/liupeng/runs/banis/mitoem2_sdt_train/checkpoints/mitoem2_all_sdt_mednextB_lr3e-4_s0_b2_k3_ns250000/default/checkpoints/epoch=171-step=250000.ckpt`
- Training: all eight MitoEM 2.0 training domains, seed 0, 250,000 steps.
- Inference: 128-cube overlapping patches, batch size 2, all seven channels.

## SDT fragments

- Foreground: predicted SDT > 0.0.
- Seeds: predicted SDT > 0.5.
- Seed connectivity: 26.
- Watershed EDT downsampling: factor 2.
- Minimum final region: 200 voxels (CLI default).

## Deterministic graph rule

- Affinity channels: all six BANIS channels.
- Offsets in array `(z,y,x)` order: `(1,0,0)`, `(0,1,0)`, `(0,0,1)`,
  `(10,0,0)`, `(0,10,0)`, `(0,0,10)`.
- Candidate edge evidence: affinity values whose endpoints lie in different
  positive fragments.
- Threshold grid: 0.40 to 0.80 in increments of 0.05.
- Minimum edge observations: 8.
- Affinity-positive threshold: 0.50.
- Minimum positive fraction: 0.50.
- Selection objective: pooled instance F1 at IoU 0.5 over validation volumes.
- Tie break: lower threshold, then higher macro F1.
- Block shape: `256 512 512`.

## Validation set

The complete local validation split contains 11 volumes:

- Beta: `me2-beta_train01`
- Jurkat: `me2-jurkat_train02`
- Macro: `me2-macro_train02`
- Mossy: `me2-mossy_train03`
- Podo: `me2-podo_train02`
- Pyra: `me2-pyra_train01`, `me2-pyra_train11`, `me2-pyra_train12`,
  `me2-pyra_train13`
- Sperm: `me2-sperm_train02`
- Stem: `me2-stem_train02`

The first audit used one validation volume per domain. The completion jobs add
the three remaining Pyra volumes. The full 11-volume summary is written to
`heldout_validation/summary_full_validation_11_cases`.

Completion jobs `2962825`, `2962826`, and `2962827` evaluated Pyra train11--13;
all completed successfully. Dependent summary job `2962828` also completed.

## Locked deterministic operating point

The full validation results are:

| Method | Macro F1 | Pooled F1 | Pooled accuracy |
|---|---:|---:|---:|
| SDT fragments | 0.652899 | 0.642715 | 0.473530 |
| Strict local oracle | 0.729282 | 0.727104 | 0.571221 |
| Affinity merge, threshold 0.40 | 0.730373 | 0.737785 | 0.584516 |

Thresholds 0.40 and 0.45 tie exactly for the primary pooled-F1 objective and
macro F1. The predeclared lower-threshold tie break retains 0.40. Threshold
0.55 has the highest macro F1 (0.731186), but a lower pooled F1 (0.736038), so it
is not selected. The locked deterministic operating point is therefore 0.40.

The previous 11-case designated-test evaluation already used threshold 0.40;
no repeat test evaluation is required after completing validation.

## Test-use disclosure

The designated MitoEM 2.0 test outputs were inspected in an early analysis-only
diagnostic before this protocol was frozen. The later threshold-transfer test
used a validation-selected threshold and no per-test-volume tuning, but it is
retrospective rather than pristine blind evidence. It must not be presented as
the final independent confirmatory test. A new external or genuinely untouched
evaluation set will be locked before learned-scorer development is finalized.

## Separation from the learned scorer

The deterministic results are a fixed baseline. A learned edge scorer will be
trained using training-volume graph edges only, tuned/calibrated using validation
volumes only, and evaluated once on a separately declared confirmatory set. The
existing designated-test metrics may be used for retrospective comparison but
not model selection.

## Offset and candidate-graph ablation

The cached per-channel sufficient statistics were re-pooled without rerunning
the dense model. Hyperparameters for each variant were selected on the complete
11-volume validation set and transferred unchanged to the 11 designated test
volumes. The selected settings and results are:

| Variant | Validation setting `(threshold, evidence, high fraction)` | Validation macro / pooled F1 | Test macro / pooled F1 |
|---|---|---:|---:|
| All six offsets | `(0.40, 8, 0.50)` | 0.730373 / 0.737785 | 0.790490 / 0.787215 |
| Long-range only | `(0.40, 1, 0.50)` | 0.730968 / 0.737747 | 0.794154 / 0.789505 |
| Short-range only | `(0.60, 8, 0.50)` | 0.695159 / 0.692998 | 0.710800 / 0.703212 |
| All offsets, local candidates only | `(0.40, 1, 0.50)` | 0.729053 / 0.735165 | 0.789835 / 0.786537 |

The long-only versus all-offset difference is not significant (domain-clustered
95% bootstrap CI for macro F1: -0.00025 to 0.00675; Holm-adjusted exact
sign-flip p=0.625). In contrast, short-only is significantly worse than
all-offset (macro difference -0.07969, CI -0.12008 to -0.02657,
Holm-adjusted p=0.0469). Long-only improves over the unmerged baseline on every
test volume (macro difference +0.05226, CI +0.02523 to +0.09935;
Holm-adjusted p=0.0391). These designated-test statistics remain retrospective
under the disclosure above.

A matched constraint ablation keeps threshold 0.40 and evidence count 8 but
removes the positive-fraction requirement (`minimum_high_fraction=0`). It drops
validation macro/pooled F1 to 0.672660/0.670000 and test macro/pooled F1 to
0.597139/0.618439 (job `2963057`), demonstrating that mean evidence alone can
create catastrophic links.

## Learned selective-gate protocol

- Training edges: predictions on the exact 23-volume complement of the frozen
  validation split within `labelsTr`.
- Target: the strict safe-merge action. A positive edge connects two fragments
  with purity at least 0.90 and the same non-background dominant GT object;
  impure/background edges are explicit unsafe negatives.
- Features: pooled and per-channel affinity moments/support, short/long support,
  physical support distance, voxel spacing/anisotropy, and fragment-size ratio.
- Model: 25-member, volume-bootstrap ensemble of standardized L2 logistic
  regressions, class-balanced, `C=1.0`, seed 0.
- Safety anchor: every accepted edge must still satisfy the frozen deterministic
  rule (`mean >= 0.40`, evidence >= 8, positive fraction >= 0.50). The learned
  probability and ensemble standard deviation only reject candidate merges.
- Validation grid: probability threshold 0.00--0.90 in increments of 0.10 and
  maximum uncertainty in `{0.02, 0.05, 0.10, 0.20, 1.0}`; pooled instance F1 is
  primary. Probability 0 and uncertainty 1 exactly reproduce the deterministic
  method and provide an implementation invariant. Exact metric ties prefer the
  lower probability threshold and then the higher uncertainty limit, so added
  gating is retained only when it measurably improves validation performance.
- The chosen probability/uncertainty pair is frozen before designated-test and
  external-confirmatory evaluation.

Two scorer variants were frozen before either learned-scorer transfer result
was inspected:

- **Unanchored performance scorer:** the scorer alone controls acceptance,
  subject only to evidence >= 8. Validation selected probability >= 0.60 and
  maximum uncertainty 1.0 (validation macro/pooled F1 =
  0.736675/0.747160). The immutable validation record is stored under
  `learned_edge_scorer_stage1/validation_unanchored/seed0`.
- **Safety-anchored scorer:** the deterministic affinity requirements above are
  additionally enforced. Validation selected probability >= 0.70 and maximum
  uncertainty 1.0 (validation macro/pooled F1 = 0.733909/0.743627); this point
  is transferred without modification.

The unanchored and safety-anchored variants answer different questions and are
reported separately. Neither variant may be selected or altered using the
designated-test or external-confirmatory results.

Frozen seed-0 transfer results are:

| Variant | MitoEM2 test macro / pooled F1 | External macro / pooled F1 |
|---|---:|---:|
| Deterministic anchor | 0.790490 / 0.787215 | 0.869187 / 0.890101 |
| Unanchored scorer | 0.793493 / 0.790493 | 0.867583 / 0.887676 |
| Safety-anchored scorer | 0.796607 / 0.791346 | 0.871438 / 0.893551 |

Relative to SDT fragments, the safety-anchored scorer improves every one of the
11 designated-test volumes (macro difference +0.05471, domain-bootstrap 95%
CI +0.02858 to +0.09896; Holm-adjusted exact domain sign-flip p=0.0391). Its
increment over the deterministic anchor is smaller and not significant
(+0.00612; CI -0.00014 to +0.01036; Holm-adjusted p=0.5625).

Seed-1 and seed-2 bootstrap-ensemble replications use the same fixed feature
schema, targets, regularization, estimator count, validation grids, anchors,
and tie breaks. They are stability analyses; they do not trigger a new choice
of operating point for seed 0. Independent dense-predictor seed-1 and seed-2
training runs were also submitted as jobs `2962946` and `2962947` before their
outcomes were available.

All scorer replication jobs completed. Across seeds 0--2, the safety-anchored
scorer obtains MitoEM2 test macro F1 0.79632 +/- 0.00029. On the two
non-overlapping external volumes, all three scorer seeds give macro F1 0.871438;
every anchored seed selects
probability 0.70 and maximum uncertainty 1.0.

## External confirmatory domain-transfer set

Before running the upgraded learned-gate pipeline, three local CellMap-style
test volumes were initially declared as external confirmatory domains:

- Cardiac: `jrc_zf-cardiac-1_recon-1_test1`
- Kidney: `jrc_mus-kidney_recon-1_test1`
- Liver: `jrc_mus-liver_recon-1_test1`

Each volume is `256 x 1024 x 1024` at isotropic 16 nm spacing and provides a
full instance-label array. A subsequent blockwise provenance audit found that
the Kidney annotation is voxel-identical to MitoEM 2.0 Podo-test01 (zero
mismatched voxels and matching canonical SHA-256). Kidney is therefore excluded
from every external aggregate and is not treated as independent evidence. The
remaining Cardiac and Liver volumes are absent from the MitoEM 2.0
dense-predictor and edge-scorer training sets. The SDT decoding parameters,
deterministic affinity rule, and learned probability/uncertainty operating
point were transferred unchanged from MitoEM 2.0 validation. They are an
external test of the current pipeline, not a claim that these public volumes
have never been viewed in any earlier BANIS project.

On the two non-overlapping volumes, the frozen deterministic rule improves
external macro F1 from 0.775052 to 0.869187 and pooled F1 from 0.821326 to
0.890101. Per-volume F1 changes are Cardiac 0.880000 to 0.910788 and Liver
0.670103 to 0.827586. No external setting was tuned.

## Short-affinity connected-components baseline

Before running this baseline, the following protocol is fixed. The first three
one-voxel affinity probability volumes are thresholded at
`{0.40, 0.45, ..., 0.90}`. Positive-direction edges define undirected voxel
connectivity; connected components smaller than 200 voxels are removed. Pooled
instance F1 on the same 11 validation volumes selects one shared threshold,
with macro F1 and then the lower threshold as tie breaks. The selected threshold
is transferred unchanged to all 11 designated-test volumes and the external
volumes. The evaluator uses the identical IoU-0.5 Hungarian metrics
and masks as the SDT graph experiments. No test or external result may alter
this baseline setting.

The validation selector locked threshold `0.60` (validation macro F1
`0.652241`, pooled F1 `0.661683`). Transferred unchanged, short-affinity CC
obtains MitoEM 2.0 test macro F1 `0.654984`, pooled F1 `0.652075`, and macro PQ
`0.510766`. BANIS-GR improves all 11 test volumes relative to this baseline;
the paired macro-F1 difference is `+0.135506` with domain-clustered 95% CI
`[+0.039560, +0.252564]` and Holm-adjusted exact sign-flip `p=0.023438` across
the three declared short-CC comparisons. On the two non-overlapping external
volumes, short-affinity CC obtains macro F1 `0.811804`, pooled F1 `0.833208`,
and macro PQ `0.661172`, below the frozen deterministic BANIS-GR result
(`0.869187` macro F1). All baseline settings remained frozen.

## MitoNet v1 zero-shot reference

Before launching the full 11-case run, MitoNet v1 was fixed as an external
pretrained, zero-shot reference. It is not a training-data-matched baseline:
the released checkpoint was trained on CEM-MitoLab, while BANIS uses the
prespecified MitoEM 2.0 training split. The result must therefore be labelled
`MitoNet v1 (zero-shot)` and cannot support a controlled architecture claim.

The standard repository defaults are fixed for every volume: native resolution;
xy/xz/yz inference; semantic confidence 0.5; center confidence 0.1; minimum
center distance 3; minimum object size 500 voxels; minimum extent 5 voxels; and
three-plane consensus vote threshold 2. There is no threshold sweep,
fine-tuning, test-time selection, or case-specific setting. The released
checkpoint SHA-256 is
`487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04`;
the frozen BANIS configuration SHA-256 is
`2394f8e6e3778dc6ab83688f4b8c1928d7f7d8d85e9959c560b09f132602f962`.
The complete freeze record is stored at
`mitonet_v1_zero_shot/FROZEN_PROTOCOL.md` in the experiment root.

Only `me2-jurkat_test01` was run before the full submission, as a functional
smoke test (job `2963223`). It produced a same-shape uint16 prediction with 126
foreground instances, and the common evaluator completed successfully. Its
metric was not used to alter any setting and remains one member of the final
11-case aggregate. Full MitoEM 2.0 case jobs are `2963226`--`2963236`, with
dependent summary job `2963237`. Confirmatory Cardiac and Liver jobs are
`2963238` and `2963239`, with summary job `2963240`; Kidney remains excluded by
the provenance audit. Evaluation uses the same validity masks, exact overlap
accumulation, IoU-0.5 Hungarian matching, and macro/pooled definitions as the
BANIS experiments. Failed cases must be reported rather than omitted.

The first external run exposed a storage-dtype mismatch: Liver contains values
only in 0--190 but is stored as uint16. The released MitoNet preprocessor scales
normalization by the dtype maximum, so this execution divided by 65535 and
returned zero instances. Jobs `2963238`--`2963240` are retained as invalid audit
artifacts and must not be reported as scientific results. Before rerunning, the
loader was assigned a deterministic lossless rule: wider integer arrays are
cast to uint8 if and only if their complete range lies in 0--255; otherwise the
original dtype is preserved. This rule involves no clipping, percentile
normalization, threshold change, or result-dependent parameter. Three I/O unit
tests cover NIfTI metadata, singleton-channel Zarr/canonicalization, and genuine
uint16 preservation. The corrected Cardiac/Liver jobs are `2963244` and
`2963245`, with dependent summary `2963246`; their outputs are stored in a new
versioned directory so the invalid first run remains auditable.

Subsequent reproducibility auditing found cross-architecture differences
between V100 and A100 executions. The final wrapper fixes seed 0 for Python,
NumPy, and PyTorch, enables strict deterministic tensor algorithms, replaces
the CUDA median-with-indices operation by the mathematically equivalent stable
sort median, fixes Python hashing and OMP/MKL thread counts, and records the GPU
model. Same-GPU job `2963258` produced label arrays with identical canonical
SHA-256 and zero mismatched voxels across two fresh processes; all metric fields
except elapsed time were exactly equal. Final reporting is restricted to the
Tesla V100-SXM2-16GB jobs `2963260`--`2963270` (MitoEM 2.0; summary `2963271`)
and `2963272`--`2963273` (Cardiac/Liver; summary `2963274`). All earlier
MitoNet outputs are retained as implementation-audit artifacts only.

## Matched nnU-Net BANIS-7 backbone control

The architecture control uses the completed
`Dataset304_MitoEM20BANIS7TrainOnly` fold-0 model. Its split contains exactly
the same 23 training volumes and 11 held-out validation volumes as the primary
BANIS experiment; a programmatic audit confirmed disjointness, full coverage
of the 34 available train/validation cases, and exact agreement with the BANIS
validation case list. No designated-test case appears in training or
validation. The frozen split SHA-256 is
`bad0102d0dc2a0d5f94f04e53b6322138e86106a0f81ae84a0ad149fe8fe7acb`.

The model is a six-stage 3-D PlainConvUNet with 31,117,123 parameters, a
96 x 160 x 128 patch, batch size 2, and 400 epochs of 250 iterations. It was
trained with the BANIS-7 objective: binary cross entropy on six short/long
directional affinity logits plus mean squared error on the tanh-activated SDT.
The final checkpoint SHA-256 is
`8cd2e8222dbc573cb79b4906cac76d53c6c1ccb7f2969d6ecdd6d5650ce09b2b`.

Inference retains only the SDT head so this row isolates the dense backbone
under the same downstream SDT decoder. Tile step is 0.5; directional mirroring
is disabled; tanh activation is applied per tile before blending. The decoder
is unchanged from the primary BANIS SDT baseline: seed threshold 0.5,
foreground threshold 0, 26-connectivity, minimum size 200, and factor-2 EDT
before marker watershed. The same blockwise IoU-0.5 evaluator and validity
masks are used. Python, NumPy, and PyTorch use seed 0; deterministic tensor and
cuDNN modes are enabled; all reportable inference jobs use an NVIDIA L40S on
`g013`.

Target/loss tests (2/2) and new SDT loading/decoding/mask tests (2/2) passed.
End-to-end smoke job `2963277` verified checkpoint loading, nnU-Net
preprocessing, source-grid restoration, NPZ export, watershed decoding, TIFF
export, and the common evaluator. Its observed test score did not alter any
setting. The full test jobs are `2963280`--`2963290`, with dependent aggregate
job `2963291`. The complete freeze record is stored at
`nnunet_banis7_matched/FROZEN_PROTOCOL.md` in the experiment root.

All 11 matched nnU-Net jobs and the aggregate completed. nnU-Net--BANIS7 SDT
obtains test macro F1 `0.672799`, pooled F1 `0.641605`, macro PQ `0.527791`,
and counts TP/FP/FN `2415/1930/768`. The primary MedNeXt SDT baseline is higher
by `0.069099` macro F1 and wins 8 of 11 per-volume comparisons. An independent
recomputation from the per-case counts reproduced the aggregate exactly.

## Streaming dense-inference validation

The deployment implementation performs overlap--add one disjoint output block
at a time and writes each normalized channel directly to float16 Zarr. Patches
crossing block boundaries are recomputed, bounding host memory at the cost of
additional inference. Fragment-graph evidence collection was already blockwise.

Unit tests cover block boundaries, padded dimensions, affinity activation, SDT
activation, and float16 persistence. Real-data job `2963823` compared the
legacy dense and streaming paths on the same Stem crop and seed-0 checkpoint.
All seven channels are voxelwise identical after the production float16
conversion. The maximum difference from the transient dense float32 array is
`0.000244140625`; affinity threshold 0.5, SDT foreground threshold 0, and SDT
seed threshold 0.5 each have zero decision mismatches relative to persisted
dense float16. Failed jobs `2963816` (missing root import path) and `2963821`
(incorrect float32-vs-float16 threshold assertion) are retained as audit
artifacts; neither produced a scientific result. The first full-Pyra launcher
job `2963824` exposed and removed an incompatible legacy loader import before
reading the image or creating predictions. The self-contained loader passed a
syntax check, and replacement profiling job `2963825` was submitted without
changing inference settings.

## Current MitoNet deterministic-run status

The final Cardiac/Liver deterministic V100 run is complete: macro/pooled F1 is
`0.536646/0.478058`, macro PQ is `0.448247`, and TP/FP/FN is `256/173/386`.
This remains an external-pretrained zero-shot reference, not a matched baseline.
Ten of 11 final MitoEM 2.0 cases completed under jobs `2963260`--`2963270`.
The original Pyra job `2963268` was canceled while pending because its 180-GB
request could not fit alongside the fixed-node workload; no prediction was
started. The same frozen command was resubmitted at 90 GB as job `2963807` on
the same V100 node, with replacement dependent summary job `2963808`. No model,
input, inference, or evaluation setting changed.

## Region-graph Mutex Watershed control

The reviewer-facing RG-MWS baseline uses the same SDT fragments, affinity
candidate pairs, pooled mean evidence, masks, and evaluator. Signed edge weight
is mean affinity minus beta; positive edges are attractive, negative edges are
mutex, and priority is the absolute signed weight. It does not use BANIS-GR's
minimum-support or positive-fraction safety rules. Affogato 0.4.2 runs in the
isolated `/projects/weilab/liupeng/.codex/envs/affogato` environment. Three
mechanism unit tests passed before evaluation.

The complete 11-volume validation sweep selected beta `0.50` by pooled F1
(macro/pooled F1 `0.729579/0.736604`; jobs `2963839`--`2963849`, summary
`2963850`). Only this beta was transferred. On 11 test volumes, RG-MWS reaches
macro/pooled F1 `0.793183/0.790506`, macro PQ `0.635242`, and TP/FP/FN
`2681/919/502` (jobs `2963876`--`2963886`, summary `2963887`). On external
Cardiac/Liver it reaches macro/pooled F1 `0.868459/0.890272` (jobs
`2963888`--`2963889`, summary `2963890`).

BANIS-GR+ minus RG-MWS test macro F1 is `+0.003424` with domain-bootstrap 95%
CI `[-0.003840, +0.010297]`, wins/ties/losses `6/1/4`, and exact domain
sign-flip `p=0.84375`. The difference is not statistically significant; this
control supports the fragment-graph representation more strongly than a claim
of unique partition-rule superiority. The first validation launch failed only
while recording conda package metadata after metrics were calculated; all cases
were rerun after adding a metadata fallback, and the failed jobs remain audit
artifacts.

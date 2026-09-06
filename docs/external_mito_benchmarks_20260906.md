# Frozen external mitochondrial benchmark extension — 2026-09-06

## Purpose and scope

Evaluate the existing MitoEM2 seed-0, 250,000-step checkpoint without fine-tuning
or selecting thresholds from new labels. Report negative domains as well as
positive results. Dataset count is not a substitute for methodological novelty.
No existing manuscript numbers are replaced with incomplete results.

Primary run root:
`/projects/weilab/liupeng/runs/banis/mitoem2_graph_repair_stage1/external_benchmarks_20260906`

MitoEM-R root: the same path suffixed with `_mitoem_r`.

Each root contains a pre-evaluation `manifest.json`, immutable model/config
SHA256 values, prepared inputs under `data/`, per-case outputs under `by_case/`,
Slurm IDs in `jobs.json`, and explicitly completeness-qualified `summary/`.

## Declared data and provenance

| Cohort | Cases | Native ZYX spacing (nm) | Notes |
|---|---:|---|---|
| CEM/MitoNet 3D benchmark, EMPIAR-10982 | 6 | C. elegans 24; fly brain 12; HeLa 15; glycolytic muscle 18; salivary gland 15; Lucchi++ 5 (isotropic) | Existing local TIFFs; not the CEM-MitoLab training corpus |
| UroCell | 5 | approximately 15,16,16 | Manual instances plus branched/contacting annotations; 287 GT objects |
| MitoEM-R public validation v2 | 1 | 30,8,8 | Original z400:500, full 4096x4096 XY; not hidden test |

Sources:

- https://www.ebi.ac.uk/empiar/EMPIAR-10982/
- https://www.ebi.ac.uk/empiar/api/entry/10982/
- https://github.com/MancaZerovnikMekuc/UroCell/tree/4bab22add57dd00dacb7ecffae2205e6d1422299
- https://mitoem.grand-challenge.org/MitoEM/
- https://huggingface.co/datasets/pytc/EM30/tree/63519706548316c1f00863cbea0ece40358a759a
- https://huggingface.co/datasets/pytc/MitoEM/tree/2105967e61abf02ae1873c514b8ea16390606384
- https://huggingface.co/datasets/pytc/MitoVerse/blob/main/splits/mitoem2.0.json

UroCell uses CC-BY-NC-SA-4.0. Its five crops are treated as one source cluster,
not five independent animals. CEM files were present in 2025 and historical
C. elegans evaluation outputs exist; do not claim they were never seen during
earlier method development. The current checkpoint's archived hparams list only
the eight MitoEM2 training domains. Different source names alone are not a
voxelwise proof of independent acquisition provenance.

Exclude MitoEM-H as an independent new source because ME2-Pyra derives from it.
Exclude the known Kidney/ME2-Podo duplicate. Lucchi++ is included only once,
inside the six-volume CEM benchmark. CellMap and further classic benchmarks are
not silently added to this frozen cohort.

## Input auditing

- TIFF data retain stack ZYX order. NIfTI data use SimpleITK's ZYX array
  convention, consistently for images, instances and attributes.
- Only lossless conversion of integral intensities in [0,255] is allowed.
  No inversion, contrast normalization, resizing, connected-component
  relabeling of GT, or per-domain threshold adjustment is performed.
- Instance IDs are preserved in uint32; every prepared Zarr block is read back
  and compared exactly with its source array. Shape, range, object counts,
  source-file SHA256 and source versions are recorded.
- UroCell NIfTI physical headers are not credible nanometer metadata; preserve
  them in the audit, but use the approximate released spacing from the authors'
  README. Do not accidentally interpret its large header spacing as nm.
- UroCell `fib1-3-3-0` branched mask has one positive voxel outside GT.
  `fib1-4-3-0` has 31 branched positives outside GT and five missing branched
  voxels in GT instance 1; its contacting mask has one positive outside GT.
  Keep GT unchanged. Any incomplete or conflicting within-object attribute is
  unknown, not inferred by majority vote. Outside-GT attribute pixels are only
  logged. Instance 1 of the last crop is excluded from branched/nonbranched
  strata, but remains in overall segmentation and the unknown stratum.
- MitoEM-R uses the official native PNG/TIFF archives, not the browser's JPEG
  images/downsampled segmentation layer. Read only validation members through
  HTTP ranges; ZIP CRC checks plus SHA256 records per original member and
  staged voxel plane support integrity checks. Preserve z-offset 400.

## Frozen protocol

- Seed-0 MedNeXt-B checkpoint: `epoch=171-step=250000.ckpt`.
- Native-resolution images, uint8 / 255, patch size 128, batch size 2,
  edge/center padding and deterministic CUDA inference.
- SDT foreground >0, seeds >0.5, minimum object size 100, EDT downsampling 2.
- GR threshold 0.40, minimum evidence 8, high-affinity fraction >=0.50.
- GR+ seed-0 learned scorer and validation operating point p=0.70, maximum
  uncertainty 1.0, with the same safety constraints. No external fitting.
- Region-graph MWS beta=0.50, run in the existing isolated affogato environment.
- Released MitoNet v1 uses the existing frozen generic configuration, without
  dataset-specific parameter choices from the original MitoNet paper.
  It is externally pretrained, not training matched.
- Common instance F1 and PQ use full-volume one-to-one IoU>=0.50 matching.
  Morphometry compares count error and physical-volume/elongation recovery.
  Every four-decoder F1/PQ is cross-checked against the morphometry evaluator.

For UroCell, use full-volume matches before stratifying GT recall. A significant
piece covers >=10% of a GT object's volume. A split has at least two such
predicted pieces; merge involvement means sharing a significant predicted
piece with another GT object. These descriptive diagnostics can coexist and
are not official UroCell scores. Do not mask away the other class to compute
subgroup F1. No per-crop-as-animal significance claims are made.

## Correctness findings and fixes

1. New preparation tests cover asymmetric TIFF/NIfTI axis order, lossless
   uint8 conversion, integer label IDs >65535, exact Zarr roundtrip, source
   changes, and incomplete attributes.
2. New output wrapper reuses the existing watershed algorithm but persists
   uint32 rather than forcing uint16. Existing predictions are not overwritten.
3. Empty fragment graphs are valid outcomes. The learned scorer now returns
   empty probability/uncertainty arrays rather than crashing in sklearn.
   Nonempty inference and model weights are unchanged.
4. Real 256-cubed UroCell testing exposed a streaming mismatch: selecting and
   rebatching patches per output block altered global batch membership, giving
   3 affinity-threshold mismatches against persisted dense float16 output.
   Streaming now preserves global batch membership (including batch partners
   outside the output block) and the dense CPU-half normalization order.
   A deliberately batch-sensitive synthetic regression test covers this.
5. Corrected real-checkpoint test (job 2971775): all seven 256-cubed channels
   are voxelwise identical after the production float16 cast; zero affinity,
   SDT-foreground or SDT-seed threshold mismatches. The float32-to-float16
   quantization difference itself is expected and is separately reported.
6. Positive-output cases exposed a CSV-selection bug in the new orchestration:
   the learned evaluator writes a baseline row before its learned-scorer row.
   The new adapter now selects the unique `learned_edge_scorer` row at p=0.70,
   uncertainty=1.0 by named fields, not row position. The independent metric
   cross-check caught this before any complete results were published. A
   regression test covers row order and duplicate/missing rows. Five early
   CPU checks were resubmitted; their GPU predictions needed no recomputation.

Pre-fix pilot outputs are preserved under
`diagnostics/pilot_before_streaming_fix` and are not included in the main
external cohort summary. Their embedded historical paths are provenance, not
current-result pointers. The pilot contained zero SDT seeds; this finding is
not a reason to loosen a test-set threshold. Recompute with the corrected
predictor and report all results.

## Commands

```bash
python scripts/prepare_external_mito_benchmarks.py --download-urocell
python scripts/submit_external_mito_benchmarks.py          # read-only plan
python scripts/submit_external_mito_benchmarks.py --submit
python scripts/summarize_external_mito_benchmarks.py
python scripts/prepare_mitoem_r_validation.py
```

Use `/projects/weilab/liupeng/conda/envs/sdt/bin/python`; RG-MWS uses
`/projects/weilab/liupeng/.codex/envs/affogato/bin/python` internally.
The launcher permits at most two sequential GPU lanes per model (four GPUs
total across BANIS and MitoNet), separates CPU decoding from GPU inference,
and refuses duplicate submissions when a ledger already exists. Summary jobs
report missing cases explicitly rather than presenting a partial mean as a
complete result.

## Submitted execution

- Main cohort: original jobs 2971807–2971840 (11 predictions, 11 CPU analyses,
  11 MitoNet references, one summary); exact IDs are in `jobs.json`.
- CPU cross-check retries: 2971852–2971855 and 2971860; current summary job
  2971861. Superseded pending summaries 2971840 and 2971856 were cancelled to
  avoid concurrent writes to the same report; their IDs remain in the ledger.
- MitoEM-R native validation download/preparation: 2971777, completed;
  100x4096x4096, 1,203 GT instances, every source plane checked voxelwise.
- MitoEM-R BANIS prediction / analysis / MitoNet / summary:
  2972265 / 2972266 / 2972267 / 2972268. The two GPU jobs also depend on
  completion of a main-cohort GPU lane (2971837 and 2971839 respectively),
  keeping the combined GPU concurrency bounded.
- Original pending MitoEM-R jobs 2971841–2971844 were cancelled and replaced:
  a post-submission dependency update introduced incompatible GPU-node
  exclusions. Resubmission uses the original resource requests and sets lane
  dependencies directly at submission. The old ledger is retained as
  `jobs_before_dependency_update.json`; no data or completed results were deleted.
- Correctness suite: 32 tests passed in the sdt environment, plus all 3
  RG-MWS tests in the isolated affogato environment. The actual-checkpoint
  dense/streaming equality check also passed as described above.

Preparation/pilot jobs 2971764, 2971766, 2971767, 2971771, 2971772,
2971774 and 2971775 remain in Slurm provenance. Earlier failures correspond
to the explicitly documented environment, empty-graph, annotation-QC or
streaming checks; they are not silently treated as successful final results.

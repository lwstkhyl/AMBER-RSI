# Data and input formats

[English README](../README.md) | [中文README](../README.zh-CN.md)

## Required assets

| Asset | Configuration | Requirement |
|---|---|---|
| Experiment archive | `paths.raw_zip` | `truncated sgRNA.zip` with aligned allele counts and the sample workbook |
| AF3 results | `paths.af3_results` | One matching primary AF3 run for each of 12 targets × 3 length groups |
| Upstream installation | `paths.upstream_root` | Working CRISPRon-BE, its tools and pretrained score models |
| ABE backbones | `paths.pretrained_root` | 25 members named `model_1_1` through `model_5_5` |
| Submitted AF3 JSON, optional | `paths.af3_submitted_json` | Original request directory or ZIP for complete-object comparison |
| Complete historical results, optional | `paths.reference_runs`, `paths.reference_deployment` | Required only when `comparison.enabled=true` |
| Helvetica fonts, optional | `figures.font_dir` | Required only by the Helvetica plotting script |

Experiment-specific data and fitted residual weights are not included in the source repository. This documentation release does not provide public accessions for them. The small reference summaries are comparison aids, not a replacement for training inputs or deployable weights. Upstream software/model acquisition is documented in the [CRISPRon-BE repository](https://github.com/RTH-tools/crispron-BE).

## Experiment archive

The preparation scripts consume precomputed aligned allele-count tables. They do not process FASTQ files. The archive must contain:

- A uniquely identifiable `sgRNA编号.xlsx` workbook with sample ID, editor, guide name, guide sequence and length annotation in the expected first five columns; `prepare_targets.py` reads it with `header=1`.
- Per-sample `Alleles_frequency_table_around_sgRNA_*.txt` tables under the established target/sample directory layout.
- Per-sample `Alleles_frequency_table.zip` files used to reconstruct full reference sequences.

Required allele-table columns include `Reference_Sequence`, `Aligned_Sequence`, `#Reads`, `n_deleted`, `n_inserted` and `n_mutated`. These column and file names specify the input schema; they do not establish which upstream software produced the supplied tables.

The fixed reproduction dataset contains the targets ABE25, ABE8, ABS4, CS1P, HPE6, PT13, PT14, S12, S16, S18, S7 and sgANGPTL3-A6. Length groups are `L20`, `L18_19` and `L16_17`; their target-complementary lengths are 20, 18 and 16 nt. Extra 5′ guide bases remain in AF3 requests and do not change this modeling length definition.

The preparation policy excludes ABE8e samples `6W63` and `6U103`. Expected included sample counts are 108 ABE8e, 111 D699 and 112 TeABE. These are part of the fixed dataset specification. Different targets, sample IDs, schemas or counts require adapting the loader rather than renaming unrelated data to match this archive.

## Prepare AF3 requests without existing results

To generate inputs before an AF3 result archive is available, use the preparation scripts directly. The destination below is a separate preparation directory; keep full-pipeline runs in a fresh output directory.

```bash
python 01_af3/prepare_targets.py \
  --allele-zip "/absolute/path/truncated sgRNA.zip" \
  --outdir /absolute/path/af3_preparation/raw

python 01_af3/write_af3_server_jobs.py \
  --manifest /absolute/path/af3_preparation/raw/af3_manifest.tsv \
  --outdir /absolute/path/af3_preparation/af3_inputs \
  --batch-size 6
```

The first command generates target/sample manifests and the upstream FASTA. The second writes 36 individual requests and batches. Submit the requests externally if new structures are needed; the pipeline does not submit or run AF3 itself.

`bash run.sh af3` also performs request comparison. It therefore needs original submitted JSON or request echoes from existing results; it is not the standalone command for preparing an entirely new AF3 experiment.

## AF3 result layout

`src/extract_distances.py` searches ZIPs recursively beneath `paths.af3_results`, identifies each job through its `_job_request.json`, and matches the recorded protein, RNA and DNA sequences. Each matched job must contain exactly five CIF structures ending in `_model_0.cif` through `_model_4.cif`.

Keep exactly one primary result for each target/length condition. Duplicate matches are rejected. DNA chains are assigned by full sequence, not by fixed chain letters. The CIF reader expects the original AF3 atom-site layout; alternative structural exports may need a different parser.

If `paths.af3_submitted_json` is set, request comparison checks complete JSON objects while ignoring whitespace and key order. Otherwise, comparison checks sequences and chain counts from server echoes and reports server-assigned seeds separately. Echo comparison does not establish complete submitted-object identity.

## Prediction features

The deployment predictor accepts a tab-separated feature table for ABE8e or D699. It does not accept sequence-only input: upstream CRISPRoff/CRISPRon scores and structural distances must already be available.

| Columns | Meaning |
|---|---|
| `outcome_id` | Unique outcome row identifier |
| `condition_id` | Unique target/length condition identifier |
| `target_id`, `length_group_id` | Target identity and one of the three supported length-group labels |
| `target_20mer` | Full 20nt protospacer, used for window coordinates and editable A positions |
| `target_30mer` | 4nt upstream + 20nt protospacer + 3nt PAM + 3nt downstream |
| `CRISPRoff_score`, `CRISPRon_score` | Upstream sequence scores |
| `sgRNA_length_numeric` | Target-complementary guide length, consistent with the group label |
| `outcome_pos3` … `outcome_pos10` | Eight binary indicators of the A→G outcome |
| `distance_pos3` … `distance_pos10` | Eight absolute filtered nearest-C1′ distances in Å |

Provide every nonempty valid A→G outcome for each condition, including outcomes with no expected observed count. With `a` editable A positions, a condition has `2**a - 1` rows. Bits outside actual A positions must be zero. The current predictor requires a nonempty outcome set.

Every target requires an `L20` reference condition so that the predictor can compute Δdistance. Distances, sequences, length and scores must be consistent across outcome rows of the same condition. `outcome_key` is reconstructed from the eight indicators; true efficiency/frequency labels are not required.

A completed deployment verification writes examples such as `validation/deployment/ABE8e_features_only.tsv`. Prediction output contains raw double-head predictions, bounded frequencies and paper-normalized values, together with normalization status. The complement `100 - sum(bounded_pred_freq)` represents no A→G edit in the modeled window, not necessarily absence of all other editing events.

The deployed inference workflow supports new valid feature tables; the fixed 12-target training loader does not become a general training interface through this prediction schema. This package has no all-target TeABE deployment predictor.

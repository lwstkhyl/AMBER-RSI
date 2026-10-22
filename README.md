# LDpred-BE

**English** | [简体中文](README.zh-CN.md)

LDpred-BE is a length- and distance-aware adaptation of CRISPRon-ABE for predicting base-editing efficiency and A-to-G outcome frequencies. It adds trainable residual branches to frozen pretrained networks and evaluates ABE8e and D699 using target-level cross-validation. A separate regression workflow transfers source-model activities to TeABE position-level editing efficiencies.

This repository provides AF3 request preparation, distance extraction from existing AF3 results, model training and evaluation, PDF figure generation, and full-data model fitting and verification.

## Requirements

- Linux; Conda is recommended.
- Python 3.10.13 and TensorFlow/Keras 2.10.0. The training workflow explicitly checks TensorFlow/Keras versions.
- NumPy 1.26.4, pandas 2.2.2, SciPy 1.15.3 and Matplotlib 3.10.5; see [requirements.txt](requirements.txt).
- A working [CRISPRon-BE installation](https://github.com/RTH-tools/crispron-BE), its external tools, and its 25 pretrained ABE models.
- Slurm for cluster execution, or a local Linux environment for sequential execution. The supplied runtime defaults to CPU execution.

## Installation

Run the following from the repository root:

```bash
conda env create -f environment.yml
conda activate ldpred-be
cp config.example.json config.json
```

The environment file installs this project's Python dependencies. Install and test the upstream CRISPRon-BE tools separately, including RNAfold and the upstream model downloads, in the environment used by the pipeline. Upstream installation instructions describe these dependencies. An existing compatible environment can be used instead.

## Data and pretrained models

Full reproduction requires the experiment's aligned allele-count archive (`truncated sgRNA.zip`), 36 matching AF3 result ZIPs, and the upstream pretrained models. Optional historical comparisons additionally require complete reference runs. These large assets are not bundled with the source code, and this pipeline does not download them automatically. Public accessions for the experiment-specific archive, AF3 results and fitted LDpred-BE adapters are not supplied in this documentation release.

See [Data and input formats](docs/DATA.md) for required files, AF3 layouts, prediction input columns and dataset restrictions. The training data loader is specific to the 12-target experiment; applying it to another training dataset requires adapting its manifest and dataset checks.

## Configuration

Edit `config.json` before running the pipeline:

| Setting | Purpose |
|---|---|
| `paths.project_root` | Base directory for `${project_root}` substitutions |
| `paths.work_dir` | Output directory, outside the source tree |
| `paths.raw_zip` | Experiment allele-count archive |
| `paths.af3_results` | Directory containing matching AF3 result ZIPs |
| `paths.upstream_root` | Working upstream CRISPRon-BE installation |
| `paths.pretrained_root` | Directory containing `model_1_1` through `model_5_5` |
| `execution` | Backend, environment activation and Slurm resources |
| `training` | Seed, optimizer settings, epochs and residual bottleneck width |
| `comparison.enabled` | Enable comparisons against complete historical runs |

For the environment created above, set `execution.conda_env` to `ldpred-be`. Set `execution.module` to an empty string if your cluster does not use environment modules. If you activate the environment yourself and do not need activation inside jobs, set `execution.activate_conda` to `false`.

For Slurm, replace the template's partition/account placeholders and adjust CPU, memory and array concurrency settings to your cluster's limits.

**Set `comparison.enabled` to `false` when historical reference runs are unavailable.** Otherwise, configure `paths.reference_runs` and `paths.reference_deployment` with the complete matching reference artifacts. Small tables in `reference/run01_summary/` do not replace complete runs.

`config.example.json` is a template; `config.json` is your editable configuration. On the first execution, the pipeline writes `config.snapshot.json` inside the selected output directory. Keep the snapshot and source files fixed while that run is active. Use a new output directory after changing code or configuration.

## Run the workflow

Inspect the execution plan, run the control tests, then submit the workflow:

```bash
bash run.sh all --dry-run
bash run.sh test
bash run.sh all
bash run.sh status
```

The default backend is Slurm. Submission returns before jobs finish; monitor jobs with `squeue` and inspect the output directory's `logs/`. For local execution, add `--backend local` to execution commands:

```bash
bash run.sh all --backend local
```

Use `--config /absolute/path/config.json` to select another configuration or `--work /absolute/path/output/run01` to override the configured output directory. Replace example absolute paths with your own paths.

| Stage | Operation |
|---|---|
| `af3` | Prepare requests and compare them with submitted JSON or existing AF3 request echoes |
| `distance` | Prepare/check requests, extract structural distances and aggregate filtered distances |
| `train` | Prepare inputs, train four-model LOTO comparisons and fit TeABE transfer regressions |
| `plot` | Prepare plotting tables and generate the main and supplementary PDFs |
| `model` | Fit the full-data ABE8e/D699 ensembles after LOTO analysis; verify saved models |
| `verify` | Repeat label-free reload verification for completed deployment models |
| `compare` | Compare AF3 requests, analysis artifacts and deployment artifacts with references |
| `all` | Run the complete workflow and write the final report |
| `status` | List stage completion and job IDs |
| `test` | Check configuration propagation, orchestration and comparison logic without training |

Individual stages are invoked as `bash run.sh STAGE`. `train` includes input preparation; `plot` and `model` require completed analysis inputs. The `af3` stage performs request verification and therefore requires comparison inputs. To generate JSON before AF3 results exist, use the standalone commands in [Data and input formats](docs/DATA.md#prepare-af3-requests-without-existing-results).

The workflow reuses completed stages when resumed with the same code, configuration and output directory. It does not launch AlphaFold 3: the structural stages consume existing AF3 results.

## Evaluation and reproducibility

The main analysis uses 12 target-level leave-one-target-out folds, with 10 training targets, one validation target and one test target per fold. All lengths and outcomes of a target remain in the same partition. Each candidate uses a 25-member ensemble.

The four main metrics are joint two-output correlations **R₂ and ρ₂** and **outcome-level and position-level MSE**. The subscript 2 denotes the joint two-output definition; these are not squared correlation coefficients. MSE is reported in squared percentage points (pp²). See [Methods](docs/METHODS.md) for normalization, weighting and transfer definitions.

The default seed is 42. For a same-seed repeat, keep settings and the software environment fixed and use a second output directory. For seed-sensitivity analysis, change `training.seed`, use a separate directory, and disable comparisons against references generated with a different seed. Identical seeds alone do not establish stability across seeds or guarantee identical results across hardware and software versions.

After a complete `all` workflow, inspect `reproduction_report.json`. When `comparison.enabled=false`, successful pipeline execution and model verification do not imply that historical artifacts were compared. AF3 verification compares complete JSON objects when `paths.af3_submitted_json` is provided; otherwise it compares chain sequences and counts from existing AF3 request echoes, with server-assigned fields reported separately.

## Figures

The plotting scripts use existing result tables and export PDF only:

| Script | Font behavior |
|---|---|
| `plotting/plot_figures.py` | DejaVu Sans; original font sizes; no external font installation |
| `plotting/plot_figures2.py` | Helvetica; licensed regular/bold font files and an explicit font configuration |

For an existing run:

```bash
python plotting/plot_figures.py \
  --config config.example.json \
  --work /absolute/path/output/run01 \
  --out /absolute/path/output/figures_dejavu
```

If model or canvas parameters differ from the template, pass that run's configuration instead. Both scripts use uppercase panel labels A–D. See [Plotting](docs/PLOTTING.md) for Helvetica setup, input tables, individual figures and output paths. The unified pipeline calls `plot_figures.py`; it does not select another script based on the configured font family.

## Full-data models and prediction

`model` fits ABE8e and D699 on all 12 targets, with 25 members per editor. Training epochs for each member and stage are fixed to the ceiling of the median best epoch across its 12 LOTO folds. These fits are deployment models; paper performance is evaluated using held-out LOTO predictions.

After `model` completes, predict with a prepared feature table:

```bash
bash run.sh predict \
  --editor ABE8e \
  --input /absolute/path/features.tsv \
  --out /absolute/path/predictions.tsv
```

Use the same configuration and output directory as the completed deployment run. Add `--work` if the run was created with an output override. Prediction requires the original pretrained backbones, residual weights and saved scalers, plus correctly prepared sequence scores and absolute distances; residual NPZ files are not standalone models. Input labels are not required. See [Prediction features](docs/DATA.md#prediction-features).

## Outputs and repository layout

| Path | Contents |
|---|---|
| `01_af3/`, `src/`, `hpc/` | Request preparation, scientific implementation and scheduling |
| `deployment/`, `plotting/`, `tests/` | Deployment, figure scripts and control tests |
| `docs/` | Data requirements, methods and plotting instructions |
| `reference/` | Small reference tables and source integrity metadata |
| Output: `summary/` | Main metrics and held-out ensemble predictions |
| Output: `transfer/summary/` | TeABE position-level transfer results |
| Output: `figures/` | Main and supplementary PDF figures |
| Output: `plot_data/`, `provenance/` | Plot values, inputs and execution records |
| Output: `deployment/run01/` | Fitted residuals, scalers and deployment records |
| Output: `validation/deployment/` | Reload verification and label-free feature examples |

Keep source code, tests, configuration templates and public documentation in Git. Local configurations, datasets, model weights, generated outputs and personal working notes are excluded by `.gitignore`. Upstream CRISPRon-BE tools and models remain external dependencies; follow the upstream project's attribution and asset-use instructions.

# RLDpred-BE: methods and implementation

[English README](../README.md) | [中文README](../README.zh-CN.md) | [Input formats](DATA.md)

RLDpred-BE is an sgRNA length- and structural distance-aware post-training framework for predicting the editing efficiency of specific base editors. This document specifies the implemented analysis. File paths below refer to source files in the repository, not to a particular author's filesystem or archived execution.

## Coordinates and labels

All positions use one-based coordinates of the full 20nt protospacer, including for truncated guides. Main-model outcomes cover A→G combinations at positions 3–10. Length groups `L20`, `L18_19` and `L16_17` use numeric target-complementary lengths 20, 18 and 16 nt.

Each editor has 36 target/length conditions and 984 nonempty candidate outcomes, including candidates with zero observed frequency. Main-model position-level evaluation contains 159 condition/A-position records per editor.

In `src/rebuild_joint_labels.py`, each sample's aligned allele counts are converted to percentages and samples within a condition are averaged equally. Editing efficiency is the proportion with a nonzero provided `n_deleted`, `n_inserted` or `n_mutated` status. Outcome frequency is the proportion of all reads with a particular window A→G combination; deletions or uncovered window sites are treated as unedited at that site. Thus overall efficiency and the sum of window A→G frequencies have different definitions.

TeABE labels cover actual A positions across the full 20nt protospacer. Each sample uses reads covering all 20 positions as its denominator, followed by equal sample averaging. This yields 312 condition/A-position records. These denominators also underlie full-window observed profiles prepared by `plotting/prepare_plot_data.py`.

## Structural distances

Implementation: `01_af3/prepare_targets.py`, `01_af3/write_af3_server_jobs.py`, `src/extract_distances.py` and `src/collect_distances.py`.

AF3 inputs contain Cas9, sgRNA and two 60nt DNA strands. The DNA strand containing the protospacer and the opposite strand are assigned by sequence. The geometry is shared across editors using this Cas9 complex; the feature is a cross-DNA distance, not a distance from a deaminase active site.

For protospacer position $j$, the main absolute feature is the shortest distance from its C1′ atom $a_j$ to any C1′ atom $b_k$ on the opposite 60nt DNA strand:

$$
d_j=\min_{1\leq k\leq60}\lVert a_j-b_k\rVert_2.
$$

Distances are in Å. The search is not restricted to the nominal paired nucleotide or the opposite 20nt window. Four definitions are retained:

| Field | Definition |
|---|---|
| `paired_c1` | C1′ to the nominal opposite paired residue's C1′ |
| `nearest_c1` | C1′ to the nearest opposite-strand C1′; model input |
| `c1_nearest_carbon` | C1′ to the nearest opposite-strand carbon atom |
| `nucleotide_nearest_carbon` | Minimum distance between carbons in the target nucleotide and opposite DNA; structural filtering |

There are five AF3 structures per condition. A structure is excluded from the clean aggregation if any position has `nucleotide_nearest_carbon < 2 Å`. The entire structure is excluded, rather than only the flagged position. The remaining distances are averaged by condition/position; sample SD is reported separately. The clean table has 36 × 20 = 720 rows.

For target $t$, length $L$ and position $j$:

$$
\Delta d_{t,L,j}=d_{t,L,j}-d_{t,20,j}.
$$

The network uses the eight values at positions 3–10, divided by training-condition SD without mean subtraction. This retains exactly zero Δdistance for L20. Length is standardized by the training-condition mean and SD. Scales are fitted only on the 10 training targets of each fold; validation/test data use the saved scales. Near-zero SD is replaced by one. Implementation: `src/common.py` and `src/prepare.py`.

## Frozen-backbone adaptation

Implementation: `src/network.py`, `src/train.py` and `src/joint_training.py`.

Each pretrained ABE member is frozen. Let $z_0$ be its 900-dimensional Dense2 preactivation. The length branch is a linear residual $z_L$; the distance branch projects normalized Δdistance through a tanh bottleneck and a linear projection to $z_{\Delta d}$. The combined network uses:

$$
z=z_0+z_L+z_{\Delta d}.
$$

The default bottleneck width is four. These residuals enter the existing Dense2 activation and downstream frozen layers.

| Candidate | Training |
|---|---|
| `original` | Original 25 backbones; no adapter fitting |
| `length` | Fit length residual only |
| `single_delta_nearest` | Fit distance residual with the length residual fixed to zero |
| `delta_nearest` | Fit length first, freeze it, then fit distance residual |

Input encoding follows the original A/T/G/C channel order and uses dataset weights `[0, 0, 0, 0, 1]`. The added length input is a standardized scalar, not a one-hot encoding.

## Target-level training and validation

There are 12 folds. For fold $f$ (one-based), the test target is `sorted(targets)[f-1]`. Remove that target and choose validation as `remaining[(f-1) % 11]`; the other 10 targets train the adapters. All lengths and outcomes of a target stay together. Implementation: `src/common.py::split`.

All 25 members use the same target partitions. Training minimizes the raw, percentage-scale double-output MSE:

$$
\mathcal L=\frac{1}{2N}\sum_{i=1}^N\left[(\hat E_i-E_i)^2+(\hat F_i-F_i)^2\right].
$$

Outcome rows and output heads are equally weighted; targets with different numbers of outcomes need not have equal total training weight. Defaults are Adam learning rate $10^{-4}$, batch size 16, at most 2,000 epochs and patience 100. The checkpoint is selected by raw joint MSE on the validation target. Epoch zero is not a candidate checkpoint. Test metrics are not used for selection.

For two editors, 25 members, three trained candidates and 12 folds, the analysis produces 1,800 adapter checkpoints. Frozen features are cached to accelerate training without changing the trainable parameters or loss.

## Ensemble evaluation

Implementation: `src/summarize.py`, `src/metrics.py` and `src/evaluation.py`.

First align outcome IDs and average the 25 members' raw double outputs. Apply postprocessing to this average, not independently to each member before averaging. Pool the held-out target predictions across folds for the main metrics.

### Joint correlations

For R₂ and ρ₂, negative predicted frequencies are set to zero; condition-level efficiency is averaged across outcome rows, and frequencies are rescaled to sum to that efficiency. The efficiency head is not clipped. An all-zero positive-frequency denominator is reported as a normalization failure.

Let $A$ and $B$ be the true/predicted $N\times2$ matrices after centering each column. Then:

$$
R_2=\frac{\sum_{i,k}A_{ik}B_{ik}}{\sqrt{\sum_{i,k}A_{ik}^2\sum_{i,k}B_{ik}^2}}.
$$

ρ₂ applies the same calculation after ranking each column separately with average ranks. These metrics are neither squared correlations nor averages of per-head correlations. Output columns are `paper_R_2` and `paper_rho_2`.

### Bounded MSE

For the two main MSE measures, use a separate fixed postprocessing rule: clip individual outcome frequencies to [0, 100]; if their condition sum exceeds 100, scale them proportionally to sum to 100. Otherwise retain their sum.

Outcome MSE averages squared frequency errors over all 984 nonempty candidate outcomes, including zero-frequency candidates. Position MSE first sums frequencies of outcomes containing an edit at each actual A position, then averages squared marginal errors over the 159 condition/A-position records. Both use pp². Fields are `bounded_outcome_mse_pp2` and `bounded_position_mse_pp2`.

The complement of bounded outcome frequencies is the probability of no A→G edit in the modeled window. It need not equal the absence of all substitutions or indels.

## TeABE position-level transfer

Implementation: `src/transfer.py` and `src/regression.py`.

For each outer fold, the same fold's ABE8e and D699 combined ensembles predict all targets. For each editor, bounded outcome frequencies are converted to actual-A marginals at positions 3–10 and averaged over those A positions. The two resulting source activities are predictions, not observed editing labels.

For each actual TeABE A position, features are the logits of those two source activities, the absolute filtered nearest-C1′ distance at that position, normalized position $x=(j-10.5)/9.5$, and $x^2$. Source probabilities are clipped to [$10^{-4}$, $1-10^{-4}$] before taking logits. The `without_distance` comparator removes the explicit absolute-distance feature but still receives activities from distance-aware source models.

Fit a fractional-logit regression to continuous labels $y_i\in[0,1]$:

$$
z_i=\beta_0+\boldsymbol\beta^\top\tilde{\boldsymbol x}_i,\qquad
\mathcal L=\sum_i w_i[\log(1+e^{z_i})-y_i z_i]+\frac{\lambda}{2}\lVert\boldsymbol\beta\rVert_2^2.
$$

The intercept is unpenalized. Weights give equal contributions to targets, then to lengths within target, then to A positions within condition. Feature means/SD are fitted with these training weights. SciPy L-BFGS-B fits the 10 training targets. Validation-target macro position MSE selects λ from 0.001, 0.01, 0.1 and 1. The selected fit is retained without adding validation or test targets to training.

Predictions are `100 * sigmoid(z)` in percentage points. The model is multiplicative on the odds scale; source probabilities are not directly multiplied. The transfer is supervised by TeABE training-target labels over positions 1–20. It is not a new TeABE neural network or a full-window joint outcome distribution model.

The 12 folds produce 312 held-out A-position predictions per regression variant. Transfer MSE is averaged within condition, across lengths within target, then across targets. The two variants together produce 624 test rows.

## Full-data deployment and verification

Implementation: `deployment/manage.py`, `deployment/train_all.py`, `deployment/predict.py` and `deployment/verify_final_model.py`.

Refit only the combined ABE8e/D699 models on all 12 targets: 25 members per editor, with length fitted before distance. Recompute scales over all training conditions and fix each member/stage's epoch count to:

$$
T=\left\lceil\operatorname{median}_{f=1}^{12}T_f^{\mathrm{best}}\right\rceil.
$$

There is no held-out validation set in this refit and no training-error-based selection of the epoch count. Full-data training predictions are fit diagnostics, not generalization estimates. The package does not fit an all-target TeABE deployment regressor.

Residual NPZ files store adapter weights and input scales and require the original frozen backbones. Verification reloads these artifacts, removes label columns from prediction inputs, checks scaling and frequency constraints, and compares reloaded predictions with fit-time predictions using the configured round-trip tolerance. This checks implementation and serialization, not accuracy on unseen targets.

## Randomness and execution records

The pipeline fixes Python/NumPy/TensorFlow seeds, deterministic TensorFlow operations, CPU selection and thread counts before scientific execution. The default base seed is 42; LOTO uses `seed + 100*fold`, with distance bottleneck initialization at one higher. Full-data stages reset to the base seed and use the corresponding saved training settings.

Record software/hardware versions, configuration snapshots, source fingerprints and reports together with outputs. Same-seed repeats test execution reproducibility. Multi-seed held-out runs test sensitivity to initialization and shuffling; they should retain the same splits and metric definitions and report all runs rather than selecting the best seed.

# 距离、模型训练、迁移及最终部署的实现说明

整合版本：2026-10-20。依据为用户上传的主分析代码、最终绘图代码，以及此前run01/run02和全数据部署的验证报告。本文件描述当前整合包的行为；用户入口与路径设置见根目录`README.md`。

## 1. 本次整合的范围与核对依据

`run.sh`调用`pipeline.py`，从单一`config.json`生成不可变运行快照，再通过`hpc/job.sbatch`启动各阶段。Slurm脚本保留`#!/bin/bash`解释器修复，资源、路径和训练参数不再散落在多份shell配置中。

AF3准备、距离提取/汇总、标签重建、target划分、训练循环、checkpoint选择、评价和迁移流程中的13个科学核心文件与上传源码逐字节相同，清单见`reference/scientific_core_sha256.json`。其余必要修改为配置传递、网络瓶颈宽度可配置、任务调度、输出路径和复现报告；默认训练值保持原设置。

历史分析报告显示run01/run02的7070个科学文件逐值相同、最大差0、容差0；历史全数据部署两轮404个文件逐值相同。历史报告与主分析summary保存在`reference/`供追溯。它们不证明新整合入口已经在HPC执行；本次需用新工作目录重建并对照旧结果，最后核对`reproduction_report.json`。

原始数据、36个AF3任务结果、原25个骨干和上游CRISPRon-BE工具均为外部依赖。不要覆盖原代码或旧结果。运行中的代码、配置、数据和模型保持固定；结果目录与代码目录分开。

## 2. 数据和坐标约定

主模型有12个target，每个target对应`L20`、`L18_19`、`L16_17`三个长度组；网络的长度数值分别为20、18、16。12个target按字符串排序为：ABE25、ABE8、ABS4、CS1P、HPE6、PT13、PT14、S12、S16、S18、S7、sgANGPTL3-A6。

主模型编辑窗口为完整20nt protospacer的第3—10位，位置从1开始；不是按截短后的sgRNA重新编号。每个编辑器有36个条件、984个非空候选A→G outcome，零实测频率的候选也保留。主模型位置指标只统计窗口中实际为A的位置，共159条条件×位置记录。

标签实现见`src/rebuild_joint_labels.py`：

- 每个样本先用自身总reads作分母算百分比，然后同一条件内样本等权平均；不是合并reads后重新加权。
- `editing_efficiency`使用CRISPResso报告的任意替换/插入/缺失reads比例。
- `outcome_frequency`只编码第3—10位A→G组合，占该样本全部reads的比例。缺失或未覆盖位置按该窗口位未发生A→G处理。
- 因而总编辑效率不必等于窗口A→G outcome频率之和。
- TeABE迁移标签另取完整第1—20位的实际A。每个样本先限制到完整覆盖20位的reads，再计算各位A→G比例，最后样本等权平均，共312条条件×A位置记录。不能把这个分母偷偷换成主模型的全部reads分母。

可核对文件：`data/joint_sample_qc.tsv`、`joint_condition_labels.tsv`、`joint_sample_outcomes.tsv`、`TeABE_sample_positions.tsv`及`TeABE_positions.tsv`。最后一个样本级文件记录`full20_covered_percent`，可检查分母覆盖情况。

## 3. 距离如何计算

### 3.1 从AF3结构得到绝对距离

实现：`01_af3/prepare_targets.py`、`01_af3/write_af3_server_jobs.py`、`src/extract_distances.py::atom_table()/extract()`。

12×3个AF3条件各使用既有的5个CIF结构。双链DNA各60nt；通过完整DNA序列匹配分配DNA1和对侧DNA2，不依赖“A链/B链”这样的固定链名。下文距离起点均为包含`target_20mer`的DNA1链。原始guide序列、scaffold和Cas9输入保持既定定义；两种编辑器共用这套Cas9-DNA几何，不能把它描述成两个脱氨酶活性中心的距离。

设DNA1第j个protospacer核苷酸的C1′坐标为a_j，对侧60nt DNA的C1′坐标为b_1…b_60，主特征是：

`d_j = min(k=1…60) ||a_j - b_k||₂`，单位Å。

对应实际代码：

```python
opposite_c1 = np.array([opposite[i]['atoms']["C1'"] for i in range(1, 61)])
start = row.DNA1_af3.index(row.target_20mer)  # 0-based
residue = start + position                  # CIF residue is 1-based
c1 = chains[own][residue]['atoms']["C1'"]
nearest_c1 = np.linalg.norm(opposite_c1 - c1, axis=1)
distance = float(nearest_c1.min())
```

最近点搜索遍历整条对侧60nt链，不限于对侧20nt窗口，也不限于名义互补配对核苷酸。提取器同时保存4种距离，但主模型只用`nearest_c1`：

| 字段 | 定义 | 用途 |
|---|---|---|
| `paired_c1` | 目标C1′到对侧名义配对残基C1′；配对残基号为61−目标残基号 | 保留供审查 |
| `nearest_c1` | 目标C1′到对侧60个C1′的最短距离 | 主模型和迁移使用 |
| `c1_nearest_carbon` | 目标C1′到对侧DNA全部碳原子的最短距离 | 保留供审查 |
| `nucleotide_nearest_carbon` | 目标核苷酸全部碳原子与对侧DNA全部碳原子的最短距离 | 结构碰撞筛查 |

审查文件：`distance_parts/*_distances.tsv`记录目标/对侧残基、最近原子和各距离；`*_audit.tsv`记录链分配、定位起点和CIF SHA256。

### 3.2 结构筛选和五结构汇总

实现：`src/collect_distances.py`。

共36×5×20=3600条位置记录。如果某结构任一位置的`nucleotide_nearest_carbon < 2Å`，则整个结构从clean均值中排除。不是仅删去那个位置，也不是按`nearest_c1 < 2Å`筛选。

每个条件/位置对保留结构的距离取算术平均；同时报告样本SD（`ddof=1`），SD不作为主模型输入。既有数据中PT14/L20的model1和S12/L16_17的model3被排除，clean每个条件使用4或5个结构。

核对：`distances/flagged_structures.tsv`、`distance_per_model.tsv`、`shared_distance_all.tsv`和`shared_distance_clean.tsv`。后两表各有720行。主分析只读取clean版本。

### 3.3 从绝对距离得到Δdistance

实现：`src/prepare.py`选取`nearest_c1`第3—10位；`src/common.py::condition_features()/arrays()`形成8维输入。

对相同target和相同位置：`Δd(t,L,j) = d_clean(t,L,j) − d_clean(t,L20,j)`。因此L20的8维Δdistance严格为0。

每一LOTO折只在10个训练target的30个条件上计算每一维Δdistance的总体SD，输入为`Δd / SD_train`，**不再减去均值**；这使零点保持L20参考意义。小于1e−8的SD替换为1。长度输入则为`(L − mean_train) / SD_train`。验证和测试target只应用训练得到的尺度。

可在已激活的`crispronbe`环境运行以下只读核对：

```bash
python - <<'PY'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/public/home/GENE_proc/wth/crispron-BE/LDpred_BE_complete_20261020/src')
from common import read_input, condition_features, split
w = Path('/public/home/GENE_proc/wth/crispron-BE/res/LDpred_BE_reproduction_20261020/run01')
f = read_input(w, 'ABE8e')
c, delta = condition_features(f)
print('targets/conditions/outcomes:', f.target_id.nunique(), len(c), len(f))
print('max |L20 delta|:', np.abs(delta.loc[c.loc[c.length_group_id.eq('L20'), 'condition_id']]).to_numpy().max())
print('fold1 train/validation/test:', split(f.target_id.unique(), 1))
PY
```

预期为12/36/984，L20差值为0，fold1测试ABE25、验证ABE8、训练其余10个target。

## 4. 主模型怎样训练

### 4.1 四个模型的差别

实现：`src/network.py::Backbone.build()`与`src/train.py`。

| 图中名称 | 代码candidate | 可训练内容 |
|---|---|---|
| 原模型 | `original` | 不训练；25个预训练模型直接预测 |
| length only | `length` | 长度到Dense2的线性残差，1800参数 |
| Δdistance only | `single_delta_nearest` | 长度残差固定为0，仅训练距离残差，3632参数 |
| length&Δdistance | `delta_nearest` | 先训练length；固定该length残差，再训练距离残差，3632参数 |

对每个原始成员，30×4序列one-hot采用A/T/G/C通道顺序；额外输入为8位outcome编码、CRISPRoff分数、CRISPRon分数和固定数据集向量`[0,0,0,0,1]`。原序列编码器、Dense2主干权重、Dense3及最终双输出头全部冻结。ABE8e和D699分别训练，不是同一个模型里学习编辑器embedding。

残差加在Dense2激活函数**之前**：

```text
z0 = frozen_Dense2(sequence_encoding, outcome, off_score, on_score, dataset)
z_length = W_length × standardized_length + b_length
z_delta = W_projection × tanh(W_bottleneck × standardized_delta)
z = z0 + z_length + z_delta
prediction = frozen_Output(frozen_Dense3(original_Dense2_activation(z)))
```

距离瓶颈为8→4，tanh、无bias；投影4→900、无bias。长度层为1→900，有bias。长度初始权重为0；距离投影初始为0，故新增分支初始输出等于前一阶段输出。图中保留的`LengthGateRaw`冻结为0，`LengthGate=1+tanh(0)=1`，不是一个额外学习到的门控。

### 4.2 LOTO划分与checkpoint

实现：`src/common.py::split()`、`src/train.py::main()`、`src/joint_training.py::train_joint()/BranchCheckpoint`。

- 共12折，target整体划分，同一target的所有长度和outcome不跨训练/验证/测试。
- 第f折测试排序后的第f个target；在其余11个中取索引`(f−1)%11`作验证；其余10个训练。
- 每个编辑器有25个原始预训练成员，每个成员独立拟合残差；所有成员使用相同target划分。
- Adam，学习率1e−4，batch16，最多2000epoch，patience100。
- 优化原始百分比尺度的双输出MSE：`loss = sum_i[(Ehat_i−E_i)^2+(Fhat_i−F_i)^2] / (2N)`。
- outcome行与两输出等权；条件的outcome数量不同会影响其训练总权重。这不是target等权的神经网络训练。
- checkpoint只按验证target的原始双输出MSE选择，恢复最优已训练epoch；epoch0不参与选择。测试指标、相关系数和归一化后的频率均不用于选checkpoint。
- 冻结序列特征缓存到Dense2线性主干输出，避免每轮重复计算CNN。缓存仅加速前向计算，不改变损失或可训练参数。

总计2编辑器×25成员×3训练candidate×12折=1800个checkpoint；原模型不额外训练。核对：`summary/fold_checkpoints.tsv`；成员下`history.tsv`、`initial_weights/*.npz`、`checkpoints/*.npz`、`branch_weights/*.npz`和`test_predictions.tsv`。

### 4.3 最终四指标的定义

实现：`src/summarize.py::summarize()`、`src/metrics.py`和`src/evaluation.py`。

先按outcome ID对齐25个成员，平均两个**原始输出**，再后处理；不能先逐成员归一化再平均。汇总12折各自留出target的预测后计算指标。

| 必须报告的指标 | 主表字段 | 定义 |
|---|---|---|
| R_2 | `paper_R_2` | 双输出各列中心化后合并内积的联合Pearson相关；不是R平方，也不是两个相关系数的均值 |
| ρ_2 | `paper_rho_2` | 每一输出列先各自用平均秩处理，再按上述联合相关公式计算 |
| position-level MSE | `bounded_position_mse_pp2` | 有界outcome频率按8位编码求各实际A位的边际效率，159条记录的平方误差均值 |
| outcome-level MSE | `bounded_outcome_mse_pp2` | 984个非空候选outcome的频率平方误差均值，包括实测为0的候选 |

R_2/ρ_2使用原论文后处理：负频率置0，同一条件的效率头取平均，再把频率按总和缩放到该效率。效率头不裁剪。对N×2的真值Y、预测P，令各列中心化后为A、B，则`R_2 = sum(A*B) / sqrt(sum(A²)*sum(B²))`。ρ_2先逐列rank再作同一运算。

两个主MSE使用另一套已固定的bounded处理：频率逐项裁剪到0—100；仅当条件频率总和超过100时同比缩小到100。不要把paper后处理的MSE与bounded主MSE混用。单位均为百分比点平方（pp²），不是0—1概率平方。TV/MAE等额外指标仍保留，但不替代这四项。

已有`summary/main_four_models.tsv`即最终作图数据源，每个编辑器四行。当前结果中D699的组合模型四项均最好；ABE8e组合模型outcome MSE最低，但相关性和位置MSE并非全部最优，结论需按实际指标表述。

## 5. TeABE模型迁移如何实现

实现：`src/transfer.py::source_activity()/run_fold()`、`src/regression.py::feature_matrix()/objective()/fit()/predict()`。

这一步迁移的是ABE8e和D699预测出的活性特征，再拟合一个小型TeABE位置回归模型；没有重新训练TeABE神经网络，也没有生成TeABE全窗口outcome联合分布。

### 5.1 源模型特征

每一外层fold使用该折训练出来的25个ABE8e组合模型和25个D699组合模型，为全部target生成原始预测，保存为`delta_nearest/source_predictions/foldNN.npz`。对回归训练、验证和测试target，都使用**同一个源模型fold**；不能拼接每行各自的OOF源模型输出，否则会破坏这个外层划分关系。

每个编辑器先对25成员原始输出求均值，bounded处理outcome频率，按outcome位编码求第3—10位边际A→G效率，再仅对该窗口实际A位取均值，得到`source_ABE8e_activity_pp`和`source_D699_activity_pp`。不是读取实测源编辑效率，也不是直接使用神经网络的效率输出头。

### 5.2 位置回归与“乘性”的含义

TeABE每个实际A位置的原始特征为：

```python
source = frame[['source_ABE8e_activity_pp', 'source_D699_activity_pp']].to_numpy() / 100
x = (frame.position.to_numpy() - 10.5) / 9.5
features = [logit(clip(source_ABE8e, 1e-4, 1-1e-4)),
            logit(clip(source_D699, 1e-4, 1-1e-4)),
            absolute_nearest_c1_at_this_position, x, x*x]
```

这里是当前长度的**绝对**`nearest_c1`，不是主神经网络的L20差值；覆盖完整第1—20位。with_distance有5个特征+截距=6个系数；without_distance删除显式绝对距离，剩5个系数。without_distance仍使用结构源神经网络，因此它不是“整个流程完全未用距离”的对照。

只用10个训练target计算加权均值和SD，再把特征标准化。权重为`1/(训练target数×该target长度条件数×该条件A位数)`，使target等权、target内部长度等权、条件内部A位等权。

连续标签为`y = true_position_pp / 100`，拟合fractional logit：

```text
z = beta0 + sum(beta_k * standardized_feature_k)
loss = sum_i w_i [log(1+exp(z_i)) - y_i*z_i] + lambda/2 * sum_k>0 beta_k²
pred_position_pp = 100 * sigmoid(z)
```

截距不惩罚；SciPy L-BFGS-B优化。λ候选为0.001、0.01、0.1、1，由该折验证target的macro位置MSE选择。选好后保留训练10个target得到的系数，不把验证或测试重新加入该折训练。

“乘性”指odds尺度：`p/(1−p)=exp(beta0)×product(exp(beta_k*z_k))`；不是把两个源模型的概率直接相乘。它用连续效率标签，不是把位点标签二值化。

迁移同样采用10训练/1验证/1测试的12折target划分。其他训练target提供TeABE完整20位监督，所以不能宣称窗口外位点完全无TeABE监督。只统计各实际A位误差；主要指标先平均条件内A位，再平均target内长度，再平均target。312个留出位点×2个回归版本=624条测试预测。

核对文件：`transfer/foldNN/source_activities.tsv`、`*_model.json`、`regression_selection.tsv`、`regression_coefficients.tsv`、`test_position_predictions.tsv`，以及`transfer/summary/overall_metrics.tsv`。模型JSON保存尺度、系数、λ、训练和验证target。迁移只输出位置指标；不应补算不存在的TeABE outcome-level MSE或双输出R_2/ρ_2。

## 6. 现在训练全12个target的最终模型

可以进行。现有12折模型用于评价；全12个target refit用于后续预测。架构、候选模型和训练规则应在refit前确定。训练集内拟合值不能加入主结果图作为泛化性能；现有LOTO四模型比较继续作为结果图。可以在方法或模型发布说明里写明最终部署模型使用全部12个target；如另有独立新target数据，可再作独立测试。[1][2]

`bash run.sh model`仅refit两种源编辑器的主模型length&Δdistance：2编辑器×25成员，每个成员先length、后冻结length训练Δdistance。它读取当前已完成的四模型LOTO分析，不改写TeABE的12折迁移模型。完整`all`入口则先完成LOTO再执行部署。若还要发布一个“全12-target TeABE部署回归器”，需另行固定正则强度及全数据源特征生成规则后拟合，不能把现有12折系数直接平均冒充全数据模型。

由于全12个target都参与拟合，不再留独立验证target，不能原样调用验证集early stopping。这里预先采用：

`epochs(editor,member,stage) = ceil(median(该成员该阶段12个LOTO最佳epoch))`。

这个中位数规则是本项目新增的、可审查的部署规则，不是某种保证最优的定理，也不是原LOTO流程已有行为；不同fold的最优轮数差异可能较大。训练轮数先写入`epochs.tsv`和`plan.json`，全数据训练不再按训练误差挑轮数或挑种子。也不额外按12/10放大轮数。

按上传的run01汇总，中位数向上取整后，25成员范围为：

| 编辑器 | length轮数范围 | Δdistance轮数范围 |
|---|---:|---:|
| ABE8e | 42—442 | 2—41 |
| D699 | 69—466 | 7—22 |

实现对应：

| 当前代码文件 | 作用 |
|---|---|
| `deployment/manage.py::prepare()` | 从已完成CV读取epoch证据和输入，固定50个任务及代码/数据指纹 |
| `deployment/train_all.py::train()/fit_fixed()` | 在全部36条件/984个outcome重新计算尺度，重新初始化残差，固定epoch拟合两阶段 |
| `src/network.py::Backbone.build()` | 直接复用既有网络，冻结原25骨干；没有复制另一套网络定义 |
| `deployment/predict.py::predict()` | 加载25个骨干、分支参数和尺度，恢复组合模型，原始双输出先平均再后处理 |
| `deployment/manage.py::finish()`、`src/reproduce.py::compare_deployment()` | 要求50成员齐全，与历史部署逐值比较参数、初始值、过程和拟合预测 |
| `deployment/verify_final_model.py::verify()` | 删除输入真值列后重载预测，独立检查尺度、保存/加载和频率边际约束 |

最终权重保存为每成员`weights.npz`，包含length kernel/bias、距离bottleneck/projection kernel和输入尺度。原始25骨干仍需保留；这些NPZ是残差权重，不是完整独立SavedModel。`training_fit_predictions.npz`明确只是训练集拟合，不输出“全数据测试指标”。

## 7. 固定随机数与稳定性应怎样验证

`pipeline.py::environment()`与`hpc/job.sbatch`在导入TensorFlow前固定Python hash seed、确定性开关、CPU模式及线程数；`train.py`再调用`tf.keras.utils.set_random_seed()`和`enable_op_determinism()`。默认基础seed42，每折为`42+100*fold`，距离瓶颈显式seed为该折seed+1；数据shuffle显式给seed并每epoch重排。[3]

部署refit使用同一配置中的seed，默认42；两个阶段分别重置seed42，距离瓶颈用43，shuffle42。这里已经没有fold，因此不再加100×fold。全部36条件重新计算尺度；优化器/损失/batch与原模型相同，变化仅是训练target数、尺度拟合范围及预先固定epoch。

需要区分：

- **同seed重复两轮**：检验执行可重复性。旧分析与旧部署已分别通过；本次是新入口，仍需从头执行并与两份旧分析及旧部署比较。需要新包自身再重复时，用同一配置执行`all`并指定另一个新`--work`；两轮各自对共同参照严格通过，即证明其受检科学产物一致。
- **不同seed重复3—5轮LOTO**：检验泛化结论对随机初始化和shuffle的敏感性。修改`training.seed`，固定target划分、其余超参数和评价定义，用独立目录报告四项指标均值/SD及趋势；不能挑最好seed作为主结果。与seed42历史逐值比较需要显式关闭，最终报告会注明未验证历史复现。
- **不同seed重复全数据refit**：只能检查参数/预测敏感性，不能替代不同seed的留出评价。当前统一配置的seed同时控制主分析和部署，避免隐含第二套种子；不同配置一律使用新结果目录。

当前交付先固定seed42完成整合包与历史结果的验证。是否增加多seed LOTO取决于论文是否需要进一步支持稳定性结论。同seed重复不等于多seed稳定性；相同seed也不保证跨软件版本或硬件完全相同，TF官方要求相同环境、硬件及确定性数据管线。[3]

## 8. 绘图与最终验证对应代码

`plotting/plot_figures.py`直接复用用户确认的最终版式，保留颜色、模型顺序、曲线、误差线、symlog和主附图结构。唯一格式为PDF。主图b读取同次`summary/total_results.tsv`中的四指标，c读取`position_predictions.tsv`，d读取`target_metrics.tsv`中的原模型减组合模型的target级outcome MSE。

`plotting/prepare_plot_data.py`从原始ZIP为三种编辑器重建完整20位实际A位置效率：每个样本的A→G reads除以该样本完整覆盖20位的reads，先样本等权平均，再对有A的target取均值/SEM。附图a从同次clean距离表计算绝对距离及相对L20的差；附图b使用上述实测值；附图c/d使用同次TeABE的`with_distance`留出预测，不使用全数据模型训练拟合值。

`src/reproduce.py::figure_audit()`核对32个主图指标与当前汇总表、八个输入表的路径，以及绘图目录仅含两份PDF。输入SHA256、逐项作图数值和运行记录位于`plot_data/`、`provenance/`，不混入`figures/`。`compare_analysis()`核对两份旧分析，`compare_deployment()`核对旧全数据模型；`final_report()`集中记录AF3比较范围、科学结果比较、模型重载和PDF检查。

若只用AF3结果包回写请求作比较，程序只证明链类型/顺序/序列/拷贝数相符，并另外报告服务器seed和附加字段；原始完整提交JSON的一致性需配置实际提交文件后另行严格比较。

## 9. 依据

1. [scikit-learn：交叉验证与训练/测试分离](https://scikit-learn.org/stable/modules/cross_validation.html)：训练集上的成绩不能充当测试性能，变换尺度只在训练集拟合。
2. [scikit-learn：GridSearchCV的refit参数](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)：交叉验证确定设置后，在全数据上重新拟合是常见发布方式；本项目并未改用GridSearchCV。
3. [TensorFlow：确定性执行与随机种子](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)：同时设置Python/NumPy/TF种子、确定性算子和确定性输入管线，并保持运行环境一致。

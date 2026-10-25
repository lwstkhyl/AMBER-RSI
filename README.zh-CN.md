# RLDpred-BE

[English](README.md) | **简体中文**

RLDpred-BE（an sgRNA length- and structural distance-aware post-training framework for predicting the editing efficiency of specific base editors）是一个感知sgRNA长度和结构距离、用于预测特定碱基编辑器编辑效率的后训练框架。

RLDpred-BE在冻结的CRISPRon-ABE预训练网络上增加长度和结构距离残差分支，预测编辑效率及A→G产物频率。主分析对ABE8e和D699进行按target划分的交叉验证；独立的回归流程将源模型活性迁移至TeABE的位点编辑效率预测。

本项目提供AF3请求准备、已有AF3结果的距离提取、模型训练与评价、PDF绘图，以及全数据模型拟合和验证。

## 运行要求

- Linux；推荐使用Conda。
- Python3.10.13、TensorFlow/Keras2.10.0。训练流程会明确检查TensorFlow/Keras版本。
- NumPy1.26.4、pandas2.2.2、SciPy1.15.3、Matplotlib3.10.5，见[requirements.txt](requirements.txt)。
- 已安装并可运行的[CRISPRon-BE](https://github.com/RTH-tools/crispron-BE)、相应外部工具及25个预训练ABE模型。
- 集群执行需要Slurm；也支持在普通Linux环境中顺序运行。默认采用CPU计算。

## 安装

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate rldpred-be
cp config.example.json config.json
```

环境文件安装本项目的Python依赖。还需按照上游CRISPRon-BE说明，在流程实际使用的环境中安装并测试RNAfold等工具，下载上游模型。也可以使用已有的兼容环境。

## 数据与预训练模型

完整复现需要本实验的已比对等位序列计数归档`truncated sgRNA.zip`、36个匹配的AF3结果ZIP，以及上游预训练模型。可选的历史结果比较还需要完整参考运行。这些大文件不随源码分发，流程也不会自动下载。本次文档发布尚未提供实验归档、AF3结果及已拟合RLDpred-BE残差权重的公开下载编号。

文件要求、AF3布局、预测输入列和数据集限制见[数据与输入格式](docs/DATA.md)。训练数据读取器针对本实验的12个target编写；用于其他训练数据集时，需要相应修改manifest和数据检查规则。

## 配置

运行前编辑`config.json`：

| 配置项 | 用途 |
|---|---|
| `paths.project_root` | `${project_root}`变量替换的基准目录 |
| `paths.work_dir` | 结果目录，须位于源码目录之外 |
| `paths.raw_zip` | 本实验的等位序列计数归档 |
| `paths.af3_results` | 匹配AF3结果ZIP所在目录 |
| `paths.upstream_root` | 可运行的上游CRISPRon-BE目录 |
| `paths.pretrained_root` | 包含`model_1_1`至`model_5_5`的目录 |
| `execution` | 后端、环境激活方式和Slurm资源 |
| `training` | 随机种子、优化器参数、训练轮数和残差瓶颈宽度 |
| `comparison.enabled` | 是否与完整历史运行比较 |

使用上面新建的环境时，将`execution.conda_env`设为`rldpred-be`。集群不使用module时，将`execution.module`留空。若已自行激活环境且任务内无需切换，将`execution.activate_conda`设为`false`。

使用Slurm时，替换模板中的分区和账户占位值，并根据集群限制调整CPU、内存和数组任务并发数。

**没有历史参考运行时，必须将`comparison.enabled`设为`false`。**需要比较时，在`paths.reference_runs`和`paths.reference_deployment`中配置完整匹配的参考结果。`reference/run01_summary/`中的小型表格不能替代完整运行。

`config.example.json`是模板，`config.json`是本机可编辑配置。流程首次执行时，会在选定结果目录生成`config.snapshot.json`。运行期间保持快照和源码固定；修改代码或配置后使用新的结果目录。

## 执行流程

先查看执行计划、运行控制测试，再提交完整流程：

```bash
bash run.sh all --dry-run
bash run.sh test
bash run.sh all
bash run.sh status
```

默认使用Slurm。提交命令返回时任务通常尚未结束，可通过`squeue`和结果目录中的`logs/`查看进度。本地顺序执行时，在执行命令中添加`--backend local`：

```bash
bash run.sh all --backend local
```

通过`--config /absolute/path/config.json`选择其他配置，通过`--work /absolute/path/output/run01`覆盖配置中的结果目录。示例绝对路径均需替换为自己的路径。

| 阶段 | 功能 |
|---|---|
| `af3` | 准备请求，并与原始提交JSON或已有AF3请求回写记录比较 |
| `distance` | 准备和核对请求，提取结构距离并汇总筛选后的距离 |
| `train` | 准备输入、完成四模型LOTO比较并拟合TeABE迁移回归 |
| `plot` | 准备绘图表，输出主图和附图PDF |
| `model` | 在LOTO分析后拟合全数据ABE8e/D699模型并验证保存结果 |
| `verify` | 对已完成的部署模型重新进行无标签重载验证 |
| `compare` | 比较AF3请求、主分析及部署模型参考产物 |
| `all` | 执行完整流程并生成最终报告 |
| `status` | 查看阶段完成情况与任务ID |
| `test` | 检查配置传递、调度和比较逻辑，不训练模型 |

单独阶段的调用形式为`bash run.sh STAGE`。`train`包含输入准备；`plot`和`model`依赖已完成的分析输入。`af3`阶段包含请求核对，需要相应比较输入；若AF3结果尚未产生，只想先生成JSON，请使用[独立请求准备命令](docs/DATA.md#prepare-af3-requests-without-existing-results)。

以相同代码、配置和结果目录续跑时，流程会复用已完成阶段。流程不启动AlphaFold3，结构相关阶段读取已有AF3结果。

## 评价与可重复性

主分析采用12折按target留一验证，每折10个训练target、1个验证target、1个测试target。同一target的所有长度和outcome始终位于同一分区，每个候选采用25成员集成。

四个主要指标是联合双输出相关**R₂、ρ₂**及**outcome-level MSE、position-level MSE**。下标2表示联合双输出定义，并非相关系数平方。MSE单位为百分比点平方pp²。归一化、加权和迁移定义见[方法说明](docs/METHODS.md)。

默认随机种子为42。同seed重复时，保持设置和软件环境一致，使用第二个结果目录。评估seed敏感性时，修改`training.seed`、使用独立目录，并关闭与不同seed历史结果的比较。同seed重复不能替代多seed稳定性分析，也不能保证跨硬件和软件版本逐值一致。

完整`all`执行后检查`reproduction_report.json`。`comparison.enabled=false`时，流程与模型验证成功不代表执行了历史产物比较。设置`paths.af3_submitted_json`时核对完整JSON对象；未设置时核对已有AF3请求回写记录中的链序列和拷贝数，并单独报告服务器生成字段。

## 绘图

绘图脚本读取已有结果表，仅输出PDF：

| 脚本 | 字体行为 |
|---|---|
| `plotting/plot_figures.py` | DejaVu Sans、原始字号，无需额外安装字体 |
| `plotting/plot_figures2.py` | Helvetica，需要合法常规体/粗体文件及明确的字体配置 |

对已有运行重新画图：

```bash
python plotting/plot_figures.py \
  --config config.example.json \
  --work /absolute/path/output/run01 \
  --out /absolute/path/output/figures_dejavu
```

如果模型或画布参数与模板不同，应传入该次运行的配置。两版子图标号均为大写A–D。Helvetica设置、输入表、单图输出和结果位置见[绘图说明](docs/PLOTTING.md)。统一入口固定调用`plot_figures.py`，不会依据字体配置自动选择其他脚本。

将已有结果重画为RLDpred-BE名称、Helvetica字体版本：

```bash
bash plotting/replot_helvetica.sh \
  /absolute/path/output/run01 \
  /absolute/path/output/RLDpred_BE_figures_helvetica \
  /absolute/path/to/Helvetica_fonts
```

此命令读取旧运行的`config.snapshot.json`，在新输出目录创建独立绘图配置，设置新名称与Helvetica字体，并在其`figures/`下生成`Fig1_RLDpred-BE.pdf`和`FigS1_RLDpred-BE.pdf`。本项目已记录路径对应的完整命令见[服务器重画说明](docs/REPLOT_HELVETICA.zh-CN.md)。

## 全数据模型与预测

`model`使用全部12个target拟合ABE8e和D699模型，每个编辑器25成员。各成员、各训练阶段的轮数固定为其12折LOTO最佳epoch中位数向上取整。这些模型用于部署；论文性能使用留出target的LOTO预测评价。

`model`完成后，对准备好的特征表执行：

```bash
bash run.sh predict \
  --editor ABE8e \
  --input /absolute/path/features.tsv \
  --out /absolute/path/predictions.tsv
```

使用部署运行对应的配置和结果目录；如果此前覆盖过结果目录，此处也需添加相应`--work`。预测依赖原预训练骨干、残差权重、保存的尺度，以及正确准备的序列分数和绝对距离。残差NPZ不能单独构成完整模型，输入无需真实标签。详见[预测特征](docs/DATA.md#prediction-features)。

## 输出与仓库结构

| 路径 | 内容 |
|---|---|
| `01_af3/`、`src/`、`hpc/` | 请求准备、科学实现与调度 |
| `deployment/`、`plotting/`、`tests/` | 部署、绘图及控制测试 |
| `docs/` | 数据要求、方法和绘图说明 |
| `reference/` | 小型参考表和源码完整性记录 |
| 结果：`summary/` | 主指标及留出集成预测 |
| 结果：`transfer/summary/` | TeABE位点迁移结果 |
| 结果：`figures/` | 主图与附图PDF |
| 结果：`plot_data/`、`provenance/` | 绘图数值、输入及执行记录 |
| 结果：`deployment/run01/` | 已拟合残差、尺度与部署记录 |
| 结果：`validation/deployment/` | 重载验证及无标签特征示例 |

Git保留源码、测试、配置模板和公开文档。本机配置、数据、权重、生成结果和个人工作记录由`.gitignore`排除。上游CRISPRon-BE工具与模型仍是外部依赖，引用及资源使用方式遵循上游说明。

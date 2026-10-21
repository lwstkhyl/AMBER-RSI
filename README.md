# LDpred-BE

LDpred-BE在冻结的CRISPRon-ABE骨干上学习长度与Δdistance残差，用于ABE8e和D699的效率及第3–10位A→G产物频率预测。本项目还包含TeABE第1–20位的位置回归迁移、论文PDF绘图和全12-target部署模型。

本公开代码副本以已验收的`LDpred_BE_complete_20261020`为基础；2026-10-21更新提供DejaVu Sans稳定绘图版和可选Helvetica绘图版，支持独立绘图输出路径，并调整发布文件组织及调度测试。两版主图和附图的子图标号均为A、B、C、D。已完成的模型、数据划分、损失和评价定义保持不变。字体更新不表示已经完成新一轮模型训练。

## 安装与配置

优先沿用此前成功运行的`crispronbe`环境。Python计算版本为3.10.13、TensorFlow/Keras2.10.0、NumPy1.26.4、pandas2.2.2、SciPy1.15.3；本次验收的绘图库为Matplotlib3.10.5。

需要新建环境时：

```bash
conda env create -f environment.yml
conda activate ldpred-be
cp config.example.json config.json
```

使用已有环境时，只需复制配置模板。编辑`config.json`，设置数据、AF3结果、上游工具、原25个骨干、结果目录和Slurm资源；若新建了`ldpred-be`环境，也将`execution.conda_env`改为该名称。集群不需要module时将`execution.module`留空；不需要环境切换时设置`activate_conda=false`。

完整流程使用本机的`config.json`，GitHub保留配置模板`config.example.json`。统一入口首次正式初始化运行时，会在`--work`指定的结果目录生成`config.snapshot.json`，保存本次实际使用的配置；它不随代码包分发，单独运行绘图Python脚本也不会生成它。已生成的配置快照和代码指纹不可修改；参数或代码变化后使用新工作目录。仅重画已有结果时，按下文直接调用绘图脚本即可。

以下依赖不随Git代码仓库分发：

- 原始`truncated sgRNA.zip`；
- 原先36个AF3任务的结构结果ZIP；不重新运行AF3；
- 已可运行的上游CRISPRon-BE项目，包括其RNAfold等外部依赖；
- 原25个冻结骨干；
- 如需逐值核对，旧分析run01/run02和旧部署run01的完整结果；
- 仅Helvetica绘图版需要合法的Helvetica常规体、粗体字体文件；DejaVu Sans稳定版无需额外字体文件。

数据、模型的公开下载地址应在发布前由项目作者补充，并遵循原始授权。上游方法见https://github.com/RTH-tools/crispron-BE及其对应论文。

## 执行

```bash
bash run.sh all --dry-run
bash run.sh all
bash run.sh status
```

默认后端为Slurm，`all`会提交依赖链；命令返回不等于训练结束。普通Linux环境可以加`--backend local`顺序执行。执行前需按配置设置相应环境。

| 阶段       | 内容                                                    |
| ---------- | ------------------------------------------------------- |
| `af3`      | 由原始ZIP准备12个target、36个AF3请求并比较历史请求      |
| `distance` | 提取已有结构距离，排除异常结构并汇总clean距离           |
| `train`    | 包含前序准备、上游分数、标签、四模型12折LOTO及TeABE迁移 |
| `plot`     | 从同次运行的数据生成主图和附图PDF                       |
| `model`    | 训练50个全12-target组合模型成员，汇总、比较并重载验证   |
| `verify`   | 重做已完成部署模型的无标签重载预测验证                  |
| `compare`  | 重做历史AF3、主分析及部署比较                           |
| `all`      | 全部执行并写总报告                                      |
| `status`   | 查看完成标记与jobID                                     |
| `test`     | 运行控制、配置传递和比较逻辑检查，不训练模型            |

各阶段可指定`--config`、`--work`和`--dry-run`。已有运行中断后，采用相同配置和相同目录继续，已完成阶段会跳过。代码或配置变更后须选择新结果目录。

历史逐值比较默认开启。只有同时拥有完整历史参考结果时才能执行此检查；没有参考结果的公开用户可设置`comparison.enabled=false`，输出报告会注明没有验证历史逐值复现。

`paths.af3_submitted_json`填写实际原始JSON目录或ZIP时，核对完整任务对象；留空时仅核对已有AF3结果回写请求的链类型、序列和拷贝数，并另报服务器seed等字段。后者不能声明完整原始提交JSON一致。

## 方法与评价

主分析按target做12折LOTO，每折10个训练、1个验证、1个测试target，25成员平均原始双输出后评价。四个candidate为原模型、length only、Δdistance only、length+Δdistance。默认seed42，Adam学习率0.0001、batch16、最多2000epoch、patience100，以验证target的联合MSE选择checkpoint。

主结果包含联合双输出相关R₂、ρ₂（不是R平方），以及bounded outcome-level MSE和position-level MSE，单位pp²。TeABE使用同折源模型预测和绝对距离等特征拟合fractional logit位置映射，不额外训练TeABE神经网络，也不输出TeABE outcome联合分布。具体公式和实现见`docs/METHODS.md`。

全12-target模型是部署用重拟合：ABE8e、D699各25成员，先length、后Δdistance。每成员各阶段轮数为该成员12折最佳epoch中位数向上取整。论文性能图保持LOTO结果，训练集拟合不作为泛化成绩。

## 绘图：DejaVu Sans稳定版与Helvetica版

### 文件放置与版本选择

将两版脚本按下表放入代码包。下文命令按“代码包内文件名”执行；无需修改`run.sh`或`pipeline.py`。

| 绘图版本                             | 提供的脚本文件       | 代码包内文件名              |
| ------------------------------------ | -------------------- | --------------------------- |
| DejaVu Sans稳定版，无需安装Helvetica | `plot_figures(2).py` | `plotting/plot_figures.py`  |
| Helvetica版，需要真实Helvetica字体   | `plot_figures2.py`   | `plotting/plot_figures2.py` |

先激活此前成功运行的`crispronbe`环境（或新建的`ldpred-be`环境），以下命令均在代码包根目录执行。两版都直接读取`--work`下已经生成的汇总表，不重新训练模型或运行AF3。已有运行需包含`summary/`、`transfer/summary/`和`plot_data/`中的绘图输入表；仅有模型权重时，应先完成汇总和绘图数据准备。

### 不使用Helvetica：稳定版

稳定版固定使用Matplotlib自带的DejaVu Sans，恢复此前定稿图的原始字号；忽略配置中的`figures.font_family`、`min_font_size`和`font_dir`。即使模板中仍填写Helvetica，也不会加载或检查Helvetica字体。

复用已验收run01重画示例：

```bash
python plotting/plot_figures.py \
  --config config.example.json \
  --work ../res/LDpred_BE_reproduction_20261020/run01 \
  --out ../res/LDpred_BE_figures_stable_20261021
```

这里使用包内已有的`config.example.json`即可，无需额外创建`config.plot.local.json`或寻找`config.snapshot.json`。直接绘图只读取绘图配置和机制图中的`training.bottleneck`，不会使用模板中的Slurm或数据路径占位值。若实际训练的bottleneck或绘图画布参数曾改动，应改用对应运行的配置文件，以确保机制图标注和画布设置正确。

这版保留原始字号，不实施Helvetica版的12pt字号下限。

### 使用Helvetica：准备字体后运行专用脚本

1. 将合法的Helvetica常规体和粗体`.ttf`、`.otf`或`.ttc`文件放入代码包的`fonts/Helvetica/`目录；脚本会动态加载，无需管理员安装。两份字体的内部家族名称应为`Helvetica`，并覆盖绘图所需符号。
2. 创建独立绘图配置。以下命令从包内模板生成`config.plot.local.json`，不会修改完整流程的`config.json`：

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("config.example.json").read_text(encoding="utf-8"))
config["figures"].update(
    font_family="Helvetica",
    min_font_size=12,
    font_dir="fonts/Helvetica_extracted",
)
Path("config.plot.local.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
```

若已有自定义的运行或画布配置，可将上面读取的`config.example.json`替换为相应`config.json`或结果目录中的`config.snapshot.json`。`font_dir`可使用绝对路径；使用相对路径时，相对于配置文件所在目录解析。

3. 调用Helvetica脚本，并使用单独的输出目录：

```bash
python plotting/plot_figures2.py \
  --config config.plot.local.json \
  --work ../res/LDpred_BE_reproduction_20261020/run01 \
  --out ../res/LDpred_BE_paper_fonts_20261021
```

Helvetica版会检查常规体、粗体、符号覆盖以及普通文字和数学下标的字号；缺少字体或所需符号时会明确报错，不静默换成其他字体。该版将数值与下标保持在至少12pt，标题和轴标签使用更大的字号。

12pt下限适用于原始PDF尺寸。默认画布宽42.672cm，缩放到18cm时，原始12pt约等于5.06pt；若要求最终论文版面中的字仍至少12pt，需要按最终宽度重新安排布局。

### 参数、输出及统一入口

- `--config`原来就有，用于指定要读取的配置文件；`config.plot.local.json`是上面明确创建的文件名，仅Helvetica示例需要它。
- `--work`指定已有分析结果所在目录，所有绘图输入均从这里读取。
- `--out`指定新的输出根目录；省略时输出写回`--work`。重画已验收结果时应指定新的`--out`。
- 默认绘制主图和附图；加`--figure main`或`--figure supp`可只画其中一张。

默认绘制两图时，输出根目录下的`figures/`生成`Fig1_LDpred-BE.pdf`和`FigS1_LDpred-BE.pdf`；指定单图时只生成对应PDF，不输出PNG或SVG。绘图数值另存于`plot_data/plotted_values/`，默认两图一起绘制时的来源、版本和输出记录写入`provenance/plot_run_all.json`。两版PDF文件名相同，建议分别使用上面示例中的稳定版和Helvetica版输出目录。

统一入口的`plot`及`all`阶段固定调用`plotting/plot_figures.py`，不会根据字体配置自动切换到`plot_figures2.py`。按本节命名放置脚本后，完整流程默认采用DejaVu Sans稳定版；Helvetica图使用上述专用命令另行生成。

对已验收run01仅重画时，使用独立代码副本中的脚本和新的`--out`，无需修改旧快照、旧代码或重新训练。更换脚本后直接用`run.sh plot`重用旧运行目录，可能因代码指纹不一致被拒绝；此时使用本节的独立Python命令。

## 输出、验证与预测

主结果在`summary/`，TeABE结果在`transfer/summary/`；作图数值和来源在`plot_data/`、`provenance/`；全数据模型在`deployment/run01/`。`reproduction_report.json`集中记录完成情况、历史比较、模型重载和绘图检查。

部署模型由25个原骨干、50份残差NPZ和保存的尺度组成。`model_manifest.json`记录成员与文件指纹；残差NPZ不能脱离原骨干单独预测。前台预测入口为：

```bash
bash run.sh predict --editor ABE8e --input /absolute/path/features.tsv --out /absolute/path/predictions.tsv
```

输入无需真实效率或频率标签，但需所有有效非空A→G outcome、上游序列分数、绝对距离及每个target的L20参考条件。参照已验证运行的`validation/deployment/ABE8e_features_only.tsv`。

小型参考结果和历史报告可保留在`reference/`，实际运行结果与大文件不提交Git。`tests/`及验证源代码应保留；具体上传清单见`docs/GITHUB_UPLOAD.md`。

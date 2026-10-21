# GitHub上传范围与操作

建议保留验证源代码，忽略运行产生的大量结果。`tests/`、`src/reproduce.py`和`deployment/verify_final_model.py`都有明确用途；其中部分被统一入口直接调用，删除后会破坏复现、比较或模型验证阶段。

## 1. 上传与忽略的具体范围

| 内容 | 是否上传 | 原因 |
|---|---|---|
| `run.sh`、`pipeline.py`、`hpc/` | 上传 | 统一入口和Slurm调度 |
| `01_af3/`、`src/`、`plotting/`、`deployment/` | 上传 | 完整方法、训练、绘图、推理及必要验证实现 |
| `tests/` | 上传 | 小型自动检查，测试配置传递、任务依赖和比较逻辑；不包含训练结果 |
| `README.md`、`docs/`、`environment.yml`、`requirements.txt` | 上传 | 使用、方法与依赖说明 |
| `config.example.json` | 上传 | 不含个人路径的配置模板；普通用户复制后编辑 |
| 本机`config.json` | gitignore | 包含本机数据路径、队列和账户配置 |
| `reference/scientific_core_sha256.json` | 上传 | 自动检查科学核心实现是否改变，需要该文件 |
| `reference/run01_summary/` | 建议上传 | 体积较小的论文参考结果，可用于核对指标；本包中约4MB未压缩 |
| `reference/historical_checks/` | 建议上传 | 小型历史比较报告，可帮助审查复现结论 |
| `reference/approved_layout/`及`approved_plot_data_manifest.json` | gitignore | 早期版式与数据来源参考，部分数值早于最终结果，容易误当成最终结果 |
| 根目录`VALIDATION.json` | gitignore | 旧代码包中是当时的本地检查状态；本次完整HPC验收已另有报告 |
| 根目录`SHA256SUMS` | gitignore | 归档包生成物，代码修改后原值失效；发布ZIP时再生成 |
| `res/`、日志、缓存、中间距离、每折模型及重复运行结果 | gitignore | 可由代码生成，数量多且不适合逐次提交 |
| 原始reads、AF3结构与模型权重 | gitignore | 作为独立数据/模型发布，并在README提供位置、版本和校验信息 |
| Helvetica字体文件 | gitignore | 使用本机合法字体；不随代码默认分发 |
| 最终两份论文PDF | 可选上传至`docs/figures/` | 少量展示用文件可保留；根目录生成的`figures/`仍忽略 |

不要用`*.json`、`*.tsv`、`*.pdf`这样的全局规则，它们会连配置模板、小型参考数据与说明图一起忽略。也不要忽略整个`deployment/`，该目录包含源代码，不是模型结果目录。

大规模输入和权重适合用论文数据仓库、项目Release等方式发布；README应明确原始25个骨干仍是预测依赖，不能只发布50份残差NPZ就称为独立模型。公开数据或权重应遵循其原有许可与项目授权；本更新包不替上游指定许可证。

## 2. 使用独立副本，保留已验收目录

本代码包已经是完整的独立副本。已验收的代码会被旧运行和部署plan按指纹核对，请保留原`LDpred_BE_complete_20261020`，把新包解压到并列目录：

```bash
cd /public/home/GENE_proc/wth/crispron-BE
unzip LDpred_BE_github_20261021.zip
cd LDpred_BE_github_20261021
```

新包包含全部源代码、完整修正版绘图脚本、`.gitignore`、公开配置模板、README和说明。它不包含个人`config.json`，需要按下文创建。`requirements.txt`将绘图库固定到实际验收记录中的Matplotlib3.10.5，其余计算依赖保留既有版本。调度测试中的队列、CPU和并发断言改为读取配置，以适配其他集群；6项测试已通过。

本机实际配置不会被提交。首次使用时执行：

```bash
cp config.example.json config.json
```

然后修改路径、队列、账户和字体目录。README应把这一步放在首次运行之前；`run.sh`和原有测试默认读取`config.json`。

完整训练比较需要旧run01/run02和旧部署的完整结果。如果公开用户没有这些参考文件，可设置`comparison.enabled=false`完成新的训练与模型重载验证；总报告会明确未执行历史逐值比较，不能把它解释成已复现旧结果。

## 3. 提交前检查

在新副本根目录执行：

```bash
git init
git add .
git status --short
git diff --cached --stat
git check-ignore -v config.json
```

确认暂存内容只有源代码、文档、模板、测试和选定的小型参考文件，再按自己的GitHub仓库配置提交与推送。

如果文件此前已被Git跟踪，新增`.gitignore`不会自动停止跟踪。先按文件或目录取消索引，例如：

```bash
git rm --cached config.json
git rm -r --cached res
```

仅对确实已经被跟踪、且本次决定忽略的路径执行；`--cached`保留磁盘文件。不要为了更新规则统一清空整个索引。

## 4. 发表与归档分开保存

GitHub保存可读的源代码与必要的小型复现材料。完整run01、原始输入、25个冻结骨干、50份残差权重、实际环境和验收包应另外完整归档。README可给出论文最终结果和数据/模型下载入口；尚未建立的链接不要编造。

官方说明：https://docs.github.com/en/get-started/git-basics/ignoring-files

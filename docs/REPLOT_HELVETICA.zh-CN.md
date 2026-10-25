# RLDpred-BE：服务器重新绘图（Helvetica）

模型名称统一为 **RLDpred-BE**，描述为 **an sgRNA length- and structural distance-aware post-training framework for predicting the editing efficiency of specific base editors**。主图标题和图例、附图涉及模型的图例、输出PDF名称及新生成的模型清单均使用新名称。

先把本次更新后的代码同步到服务器的独立代码副本。下列命令沿用项目笔记中的代码位置、已完成run01及Helvetica字体位置；若实际位置不同，替换对应路径。

```bash
module load miniconda3/base
conda activate crispronbe
cd /public/home/GENE_proc/wth/crispron-BE/LDpred_BE_github_20261021

bash plotting/replot_helvetica.sh \
  /public/home/GENE_proc/wth/crispron-BE/res/LDpred_BE_reproduction_20261020/run01 \
  /public/home/GENE_proc/wth/crispron-BE/res/RLDpred_BE_paper_fonts_20261025 \
  /public/home/GENE_proc/wth/crispron-BE/LDpred_BE_github_20261021/fonts/Helvetica_extracted
```

脚本从run01读取`config.snapshot.json`，保留实际bottleneck、画布及其他参数，在新输出目录中写入`config.plot.local.json`，明确设置`model_name="RLDpred-BE"`、`font_family="Helvetica"`、`min_font_size=12`和绝对字体路径，然后调用`plotting/plot_figures2.py`。即使旧快照仍记录旧模型名，新图也使用新名称。

若不使用module，省略`module load`一行；继续使用此前成功运行的`crispronbe`环境即可。字体目录需要含内部家族为Helvetica的常规体及粗体文件。该目录中的字体不随Git提交，须在服务器上已有或单独同步。

新文件位于：

- `RLDpred_BE_paper_fonts_20261025/figures/Fig1_RLDpred-BE.pdf`
- `RLDpred_BE_paper_fonts_20261025/figures/FigS1_RLDpred-BE.pdf`
- `RLDpred_BE_paper_fonts_20261025/provenance/plot_run_all.json`
- `RLDpred_BE_paper_fonts_20261025/plot_data/plotted_values/`

命令只读取旧运行，绘图配置和输出写到新目录；无需重新训练或运行AF3。旧结果路径中的`LDpred_BE`保留为实际历史目录名。不要修改旧快照，也不要用更新后的`run.sh plot`续跑旧目录，以免触发代码指纹检查。

脚本需要[绘图说明](PLOTTING.md#required-tables)列出的8份表。若提示缺少输入，先确认指定的是包含`summary/`、`transfer/summary/`、`plot_data/`及`config.snapshot.json`的完整run01目录。

生成后可检查字体和绘图记录：

```bash
out=/public/home/GENE_proc/wth/crispron-BE/res/RLDpred_BE_paper_fonts_20261025
pdffonts "$out/figures/Fig1_RLDpred-BE.pdf"
pdffonts "$out/figures/FigS1_RLDpred-BE.pdf"
python - "$out" <<'PY'
import json
from pathlib import Path
import sys

record = json.loads((Path(sys.argv[1]) / "provenance/plot_run_all.json").read_text(encoding="utf-8"))
print("Model:", record["config"]["model_name"])
print("Font:", record["typography"]["family"])
print("Glyph sizes:", record["typography"]["figures"])
print("Layout warnings:", record["layout_warnings"])
PY
```

`pdffonts`由Poppler提供，若未安装可先检查JSON记录。应看到新模型名、Helvetica、最小字形字号至少12pt以及空的`layout_warnings`，随后打开两份PDF核对标题、图例和子图标号。12pt下限对应原始PDF尺寸，缩放到论文版面后需再按实际尺寸核对。

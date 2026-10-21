# 字体核查与修正版绘图

## 1. 现有最终PDF不满足Helvetica、最小12pt

直接检查用户上传的最终两份PDF，实际字体均为嵌入的`DejaVuSans`、`DejaVuSans-Bold`，没有Helvetica。

| PDF | 普通文字最小字号 | 包含数学下标的最小实际字号 |
|---|---:|---:|
| 主图 | 10pt | 约7.7pt |
| 附图 | 10pt | 10pt |

这与原脚本的DejaVu Sans、10–11pt设置一致。数学表达式会缩小下标，因此只把`font.size`改成12仍不能保证下标达到12pt。

## 2. 修正版做了什么

本包提供完整的`plotting/plot_figures.py`，保留原数据读取、模型顺序、颜色、面板与统计口径，新增：

- 默认请求真实Helvetica，并支持从`figures.font_dir`加载用户已有的常规体、粗体字体。
- 字体查找关闭默认替代；缺少字体或所需字符时明确报错，不把Arial、Nimbus Sans或DejaVu冒充Helvetica。
- 普通文字至少12pt；另外检查数学表达式中各个字形的实际字号，必要时提高表达式字号，使下标也至少12pt。
- 数学字体与普通字体采用同一家族；记录字体文件SHA256、各图最小字形字号及画布尺寸。
- 仅输出PDF，使用嵌入字体。新增`--out`，可读取旧run01的数据并把新图写到另一个目录。

配置模板新增的字段为：

```json
"font_family": "Helvetica",
"min_font_size": 12,
"font_dir": "/absolute/path/to/your/Helvetica_fonts"
```

本地缺少Helvetica文件，因此本更新包不包含“已验证的Helvetica成品PDF”。真实字体下的最终排版仍需在字体可用后核查。字体文件应覆盖图中的Δ、ρ、ℓ、→、−、′、²等字符；缺少符号时脚本会提示，不自动混入其他字体。

## 3. 字号还取决于论文中最终放置宽度

最小12pt应明确是原始PDF尺寸，还是论文中的最终尺寸。当前画布宽16.8英寸，即42.672cm。把它缩放到18cm宽时，原始12pt会变成约5.06pt：

`最终字号 = PDF原字号 × 最终放置宽度 / PDF原宽度`

若严格要求18cm宽的最终图中仍至少12pt，那么原始42.672cm画布上的最小字号要约28.45pt，且需要重新安排面板和间距。不能只把一个字体参数改成12就称为满足最终排版要求。

本修正版保证的是**原始输出PDF中的最小字形字号**。保持现有已确认的面板布局时，最终缩放后的要求需按投稿版面宽度另行核查。

## 4. 只重画，不改动已验收运行

按`GITHUB_UPLOAD.md`先建立独立代码副本，在副本的`config.json`中设置字体目录，再在原计算环境中执行：

```bash
module load miniconda3/base
conda activate crispronbe
python /public/home/GENE_proc/wth/crispron-BE/LDpred_BE_github_20261021/plotting/plot_figures.py \
  --config /public/home/GENE_proc/wth/crispron-BE/LDpred_BE_github_20261021/config.json \
  --work /public/home/GENE_proc/wth/crispron-BE/res/LDpred_BE_reproduction_20261020/run01 \
  --out /public/home/GENE_proc/wth/crispron-BE/res/LDpred_BE_paper_fonts_20261021
```

不要覆盖已验收目录里的源代码后再对旧run01执行`run.sh plot`：旧目录会检查代码指纹。上面的独立绘图命令使用同次旧run01的已验证数据，输出到新目录，模型不需重训。

生成后检查新目录的`provenance/plot_run_all.json`中`typography`和`layout_warnings`，并实际查看两份PDF，特别留意公式、四模型图例、柱顶数字和target名称。可另外执行：

```bash
pdffonts /public/home/GENE_proc/wth/crispron-BE/res/LDpred_BE_paper_fonts_20261021/figures/Fig1_LDpred-BE.pdf
pdffonts /public/home/GENE_proc/wth/crispron-BE/res/LDpred_BE_paper_fonts_20261021/figures/FigS1_LDpred-BE.pdf
```

应看到Helvetica家族及嵌入字体标记。Matplotlib官方字体说明：https://matplotlib.org/stable/users/explain/fonts.html

数学字体设置说明：https://matplotlib.org/3.10.5/users/explain/text/mathtext.html

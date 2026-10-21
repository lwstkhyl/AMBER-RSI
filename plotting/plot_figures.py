#!/usr/bin/env python3
"""Plot the approved LDpred-BE figures with the original DejaVu Sans typography."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import FixedFormatter, FixedLocator

# The pipeline supplies its immutable snapshot; figures always use this run's tables.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config.json')
parser.add_argument('--work', type=Path, required=True)
parser.add_argument('--out', type=Path, help='Separate output directory; defaults to --work')
parser.add_argument('--figure', choices=['all', 'main', 'supp'], default='all')
args = parser.parse_args()
config_path = args.config.resolve()
full_config = json.loads(config_path.read_text(encoding='utf-8'))
config = full_config['figures']
root = args.work.resolve()
output_root = args.out.resolve() if args.out else root
out_dir = output_root / 'figures'
out_dir.mkdir(parents=True, exist_ok=True)
plot_data_dir = output_root / 'plot_data/plotted_values'
plot_data_dir.mkdir(parents=True, exist_ok=True)
dpi = config['dpi']
formats = ['pdf']
model_name = config['model_name']
inputs = {key: root / value for key, value in {
    'main_summary': 'summary/total_results.tsv', 'main_positions': 'summary/position_predictions.tsv',
    'main_targets': 'summary/target_metrics.tsv', 'structure_distances': 'plot_data/distance_per_target.tsv',
    'observed_windows': 'plot_data/observed_window_summary.tsv',
    'transfer_conditions': 'transfer/summary/condition_metrics.tsv',
    'transfer_curves': 'transfer/summary/editing_window_curve.tsv',
    'transfer_by_length': 'transfer/summary/metrics_by_length.tsv'}.items()}
input_hashes = {}
saved_files = []
layout_warnings = []

# Fixed model identities, colors, and ordering used in the approved figures.
editors = ['ABE8e', 'D699']
lengths = ['L16_17', 'L18_19', 'L20']
length_labels = dict(zip(lengths, ['16/17 nt', '18/19 nt', '20 nt']))
length_colors = dict(zip(lengths, ['#B96BA7', '#D69B12', '#1686A8']))
order = ['delta_nearest', 'single_delta_nearest', 'length', 'original']
colors = {'original': '#9CA9B3', 'length': '#4D84B3', 'single_delta_nearest': '#62A49D', 'delta_nearest': '#D27B3D'}
labels = {'original': 'CRISPRon-ABE', 'length': f'{model_name}\nLength mode',
    'single_delta_nearest': f'{model_name}\nΔdistance mode',
    'delta_nearest': f'{model_name}\nLength + Δdistance mode'}
observed_color = '#7A7A7A'
# Preserve the original font sizes, including normal mathematical subscripts.
# Ignore figures.font_family, min_font_size and font_dir in Helvetica configs.
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
    'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#65727A', 'axes.linewidth': .8, 'xtick.color': '#34434D',
    'ytick.color': '#34434D', 'text.color': '#263743', 'axes.labelcolor': '#34434D',
    'legend.frameon': False, 'savefig.facecolor': 'white', 'mathtext.default': 'regular',
    'mathtext.fontset': 'dejavusans', 'pdf.use14corefonts': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def read_table(key):
    path = inputs[key]
    input_hashes[key] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return pd.read_csv(path, sep='\t')

def style(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#E6EBEF', lw=.7)
    ax.tick_params(labelsize=10, length=3)

def prediction_style(model):
    color = observed_color if model == 'observed' else colors[model]
    return dict(color=color, marker='o', ls='-', lw=1.55, ms=4,
        markerfacecolor=color, markeredgewidth=.95, markeredgecolor=color)

def diagram_box(ax, x, y, width, height, label):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=.006,rounding_size=.012',
        facecolor='white', edgecolor='#869099', lw=1.1))
    ax.text(x + width / 2, y + height / 2, label, ha='center', va='center', fontsize=11)

def diagram_nodes(ax, x, y, count, color):
    ys = np.array([y]) if count == 1 else np.linspace(y - .064, y + .064, count)
    ax.scatter(np.full(count, x), ys, s=24, facecolor='white', edgecolor=color, lw=1, zorder=4)
    return ys

def diagram_arrow(ax, points, color):
    p = np.asarray(points)
    ax.plot(p[:-1, 0], p[:-1, 1], color=color, lw=1.6)
    ax.add_patch(FancyArrowPatch(p[-2], p[-1], arrowstyle='-|>', mutation_scale=10, color=color, lw=1.6))

def save_figure(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            if box.x0 < 0 or box.x1 > fig.bbox.width or box.y0 < 0 or box.y1 > fig.bbox.height:
                layout_warnings.append({'figure': stem, 'text': label.get_text()})
    for extension in formats:
        path = out_dir / f'{stem}.{extension}'
        fig.savefig(path, dpi=dpi, facecolor='white')
        saved_files.append(str(path))
        print(f'Saved {path}')
    plt.close(fig)

if args.figure in ['all', 'main']:
    summary = read_table('main_summary')
    positions = read_table('main_positions')
    targets = read_table('main_targets')
    fig = plt.figure(figsize=config['main_size_inches'], dpi=dpi, facecolor='white')
    for label, y in [('a', .997), ('b', .799), ('c', .547), ('d', .202)]:
        fig.text(.014, y, label, fontsize=21, fontweight='bold', va='top')
    fig.text(.068, .991, f'{model_name}: a length- and distance-aware post-training framework', fontsize=13.2, va='top')
    # Main a: add adapters to Dense2 preactivation; the source network is frozen.
    ax = fig.add_axes([.045, .806, .93, .168])
    ax.set(xlim=(0, 1.035), ylim=(0, 1))
    ax.axis('off')
    grey, blue, orange = '#7D878F', colors['length'], colors['delta_nearest']
    for x, y, width, height, color in [(0, .66, .53, .33, '#F0F2F4'),
        (.006, .30, .53, .32, '#EEF6FA'), (.006, .005, .53, .29, '#FBF2E9'),
        (.647, .36, .382, .35, '#F0F2F4')]:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0,rounding_size=.004',
            facecolor=color, edgecolor='none', zorder=0))
    ax.text(.017, .925, 'CRISPRon-ABE (frozen)', fontsize=11)
    diagram_box(ax, .022, .735, .175, .15, 'Original inputs\n+ pretrained encoder')
    diagram_box(ax, .236, .735, .12, .15, 'Dense2\n(linear)')
    diagram_arrow(ax, [(.205, .81), (.23, .81)], grey)
    diagram_arrow(ax, [(.363, .81), (.40, .81)], grey)
    diagram_nodes(ax, .414, .81, 5, '#869099')
    ax.text(.414, .925, r'$z_0$ (900)', ha='center', fontsize=11)
    diagram_arrow(ax, [(.428, .81), (.55, .81), (.55, .595), (.607, .545)], grey)
    ax.text(.025, .59, 'Length adapter', color=blue, fontsize=12.2, weight='bold', va='top')
    ax.text(.065, .47, r'Length $\ell$', color=blue, fontsize=11, ha='center')
    ax.text(.25, .47, 'Linear (1 → 900)', color=blue, fontsize=11, ha='center')
    diagram_nodes(ax, .065, .40, 1, blue)
    length_nodes = diagram_nodes(ax, .452, .40, 5, blue)
    for y in length_nodes:
        ax.plot([.072, .445], [.40, y], color=blue, alpha=.22, lw=.55)
    ax.text(.452, .52, r'$z_\ell$ (900)', ha='center', color=blue, fontsize=11)
    diagram_arrow(ax, [(.465, .40), (.57, .40), (.607, .49)], blue)
    ax.text(.025, .29, 'Δdistance adapter', color=orange, fontsize=12.2, weight='bold', va='top')
    distance_input = diagram_nodes(ax, .065, .085, 5, orange)
    distance_hidden = diagram_nodes(ax, .258, .085, 4, orange)
    distance_output = diagram_nodes(ax, .452, .085, 5, orange)
    for x0, x1, left_nodes, right_nodes in [(.072, .251, distance_input, distance_hidden),
        (.265, .445, distance_hidden, distance_output)]:
        for y0 in left_nodes:
            for y1 in right_nodes:
                ax.plot([x0, x1], [y0, y1], color=orange, alpha=.22, lw=.55)
    for x, label in [(.065, 'Δd (8)'), (.258, f'tanh ({full_config["training"]["bottleneck"]})'), (.452, 'Linear (900)')]:
        ax.text(x, .18, label, ha='center', color=orange, fontsize=11)
    diagram_arrow(ax, [(.465, .085), (.623, .085), (.623, .485)], orange)
    ax.scatter([.623], [.52], s=310, facecolor='white', edgecolor='#536979', lw=1.2, zorder=5)
    ax.text(.623, .52, '+', ha='center', va='center', fontsize=18, color='#536979', zorder=6)
    ax.text(.623, .75, r'$z=z_0+z_\ell+z_{\Delta d}$', ha='center', fontsize=15)
    diagram_arrow(ax, [(.64, .52), (.66, .52)], grey)
    diagram_box(ax, .665, .445, .105, .15, 'Dense2\nactivation')
    diagram_arrow(ax, [(.78, .52), (.798, .52)], grey)
    dense_nodes = diagram_nodes(ax, .812, .52, 5, '#869099')
    output_nodes = [.475, .565]
    ax.scatter([.872, .872], output_nodes, s=24, facecolor='white', edgecolor='#869099', lw=1, zorder=4)
    for y0 in dense_nodes:
        for y1 in output_nodes:
            ax.plot([.819, .865], [y0, y1], color='#A9B6C0', alpha=.3, lw=.55)
    ax.text(.812, .66, 'Dense3 (200)', ha='center', fontsize=11)
    ax.text(.888, .565, 'Editing efficiency', va='center', fontsize=11)
    ax.text(.888, .475, 'Outcome frequency', va='center', fontsize=11)
    # Main b: preserve the original four reported metrics and reverse model order.
    fig.legend(handles=[Patch(facecolor=colors[m], label=labels[m]) for m in order], loc='center',
        bbox_to_anchor=(.53, .774), ncol=4, fontsize=11, handlelength=1.4, columnspacing=3.3)
    fields = ['paper_R_2', 'paper_rho_2', 'bounded_outcome_mse_pp2', 'bounded_position_mse_pp2']
    metric_labels = [r'$R_2$', r'$\rho_2$', 'Outcome MSE (pp²)', 'Position MSE (pp²)']
    limits = [(-.18, .77), (0, .79), (0, 95), (0, 2420)]
    metric_rows = []
    for j, (field, label, lim) in enumerate(zip(fields, metric_labels, limits)):
        ax = fig.add_axes([.13 + j * .214, .579, .181, .165])
        for editor, ys in zip(editors, [np.array([7.4, 6.4, 5.4, 4.4]), np.array([2.6, 1.6, .6, -.4])]):
            values = summary[summary.editor == editor].set_index('candidate').loc[order, field].to_numpy()
            best = np.argmax(values) if j < 2 else np.argmin(values)
            ax.barh(ys, values, height=.66, color=[colors[m] for m in order], zorder=3)
            for i, (y, value) in enumerate(zip(ys, values)):
                gap = (lim[1] - lim[0]) * .016
                ax.text(value + gap if value >= 0 else value - gap, y, f'{value:.3f}' if j < 2 else f'{value:.2f}',
                    ha='left' if value >= 0 else 'right', va='center', fontsize=11, weight='bold' if i == best else 'normal')
                metric_rows.append({'editor': editor, 'candidate': order[i], 'metric': field, 'value': value})
        ax.axhline(3.5, color='#E1E7EC', lw=.9)
        ax.axvline(0, color='#A9B2B9', lw=.8)
        ax.set(ylim=(-1.2, 8.1), xlim=lim, yticks=[], xlabel=label)
        ax.spines['left'].set_visible(False)
        style(ax, 'x')
    for name, y in [('ABE8e', .705), ('D699', .620)]:
        fig.text(.08, y, name, ha='center', va='center', fontsize=13, weight='bold')
    for x, label in [(.3275, 'higher is better'), (.7555, 'lower is better')]:
        fig.text(x, .547, label, ha='center', va='center', fontsize=10.5, color='#34434D')
    pd.DataFrame(metric_rows).to_csv(plot_data_dir / 'main_b_metrics.tsv', sep='\t', index=False)
    # Main c: position-wise means, solid lines, filled circles, and common titles.
    curve_rows = []
    for row, editor in enumerate(editors):
        for col, length in enumerate(lengths):
            ax = fig.add_axes([.087 + col * .304, .405 - row * .142, .257, .112])
            base = positions[(positions.editor == editor) & (positions.processing == 'bounded') & (positions.length_group_id == length)]
            truth = base[base.candidate == 'original'].groupby('position').true_position_efficiency.mean()
            ax.plot(truth.index, truth, zorder=3, **prediction_style('observed'))
            curve_rows.extend([{'editor': editor, 'length_group_id': length, 'series': 'observed', 'position': p, 'editing_pp': v}
                for p, v in truth.items()])
            for model in ['original', 'length', 'delta_nearest']:
                line = base[base.candidate == model].groupby('position').predicted_position_efficiency.mean()
                ax.plot(line.index, line, zorder=4, **prediction_style(model))
                curve_rows.extend([{'editor': editor, 'length_group_id': length, 'series': model, 'position': p, 'editing_pp': v}
                    for p, v in line.items()])
            ax.set(xlim=(2.8, 10.2), ylim=(-2, 107), xticks=np.arange(3, 11), yticks=[0, 25, 50, 75, 100],
                title=f'{editor} | {length_labels[length]}')
            if col:
                ax.tick_params(axis='y', labelleft=False)
            style(ax)
    fig.text(.056, .390, 'A-to-G editing (%)', rotation=90, ha='center', va='center', fontsize=11, color='#34434D')
    fig.text(.5195, .2432, 'Protospacer position', ha='center', va='center', fontsize=11)
    legend_order = ['delta_nearest', 'length', 'original', 'observed']
    handles = [Line2D([], [], label='Observed' if m == 'observed' else labels[m], **prediction_style(m)) for m in legend_order]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.53, .217), ncol=4, fontsize=10.8, columnspacing=2.4)
    pd.DataFrame(curve_rows).to_csv(plot_data_dir / 'main_c_curves.tsv', sep='\t', index=False)
    # Main d: original minus full-model outcome MSE; positive values are improvements.
    target_order = sorted(targets.target_id.unique())
    difference_rows = []
    linthresh = config['symlog_linthresh_pp2']
    for col, editor in enumerate(editors):
        ax = fig.add_axes([.087 + col * .46, .081, .405, .096])
        table = targets[(targets.editor == editor) & targets.candidate.isin(['original', 'delta_nearest'])]
        table = table.pivot(index='target_id', columns='candidate', values='bounded_outcome_mse_pp2').loc[target_order]
        delta = table.original - table.delta_nearest
        x = np.arange(len(target_order))
        color = np.where(delta >= 0, '#388774', '#BE5C70')
        ax.axhline(0, color='#7F8C94', lw=1)
        ax.axhspan(-linthresh, linthresh, color='#F4F6F7', zorder=0)
        ax.vlines(x, 0, delta, color=color, lw=1.6)
        ax.scatter(x, delta, s=32, color=color, zorder=4)
        ax.set_yscale('symlog', base=10, linthresh=linthresh, linscale=1)
        ax.set(xlim=(-.6, len(target_order) - .4), ylim=config['symlog_ylim_pp2'], title=editor, xticks=x,
            xticklabels=target_order, ylabel='Δoutcome MSE (pp²; symlog)')
        ax.yaxis.set_major_locator(FixedLocator([-1000, -100, -10, 0, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(FixedFormatter(['−1000', '−100', '−10', '0', '10', '100', '1000']))
        if col:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')
        style(ax)
        difference_rows.extend([{'editor': editor, 'target_id': target, 'original_mse_pp2': table.loc[target, 'original'],
            'ldpred_be_mse_pp2': table.loc[target, 'delta_nearest'], 'difference_pp2': value} for target, value in delta.items()])
    fig.text(.5195, .0234, 'Target', ha='center', va='center', fontsize=11)
    pd.DataFrame(difference_rows).to_csv(plot_data_dir / 'main_d_differences.tsv', sep='\t', index=False)
    save_figure(fig, 'Fig1_LDpred-BE')

if args.figure in ['all', 'supp']:
    geometry = read_table('structure_distances')
    window = read_table('observed_windows')
    transfer_conditions = read_table('transfer_conditions')
    transfer_curves = read_table('transfer_curves')
    transfer_lengths = read_table('transfer_by_length')
    fig = plt.figure(figsize=config['supplement_size_inches'], dpi=dpi, facecolor='white')
    for label, x, y in [('a', .028, .982), ('b', .028, .658), ('c', .028, .337), ('d', .325, .337)]:
        fig.text(x, y, label, fontsize=21, fontweight='bold', va='top')
    # Supplement a: mean ± SD across targets, using the distances already supplied.
    structure_rows = []
    for j, delta in enumerate([False, True]):
        ax = fig.add_axes([.08 + j * .48, .737, .39, .205])
        ax.axvspan(2.5, 10.5, color='#EFF2F5', zorder=0)
        field = 'delta_nearest_c1' if delta else 'nearest_c1'
        for length in lengths:
            line = geometry[geometry.length_group_id == length].groupby('position')[field].agg(['mean', 'std'])
            ax.plot(line.index, line['mean'], '-o', lw=1.7, ms=3.7, color=length_colors[length])
            ax.fill_between(line.index, line['mean'] - line['std'], line['mean'] + line['std'],
                facecolor=length_colors[length], alpha=.13, lw=0)
            structure_rows.extend([{'length_group_id': length, 'quantity': field, 'position': p, 'mean': r['mean'], 'sd': r['std']}
                for p, r in line.iterrows()])
        if delta:
            ax.axhline(0, ls='--', color='#87929B', lw=.8)
        ax.set(xlim=(.7, 20.3), xticks=[1, 5, 10, 15, 20],
            ylabel='Δdistance (Å)' if delta else 'Nearest C1′–C1′ distance (Å)')
        style(ax)
    fig.text(.515, .7059, 'Protospacer position', ha='center', va='center', fontsize=11)
    fig.legend(handles=[Line2D([], [], color=length_colors[k], marker='o', lw=1.7, label=length_labels[k]) for k in lengths],
        loc='center', bbox_to_anchor=(.53, .685), ncol=3, fontsize=11, columnspacing=3)
    pd.DataFrame(structure_rows).to_csv(plot_data_dir / 'supplement_a_distance_profiles.tsv', sep='\t', index=False)
    # Supplement b: experimental profiles with the supplied SEM error bars.
    editor_colors = {'ABE8e': '#28343B', 'D699': '#4D91C2', 'TeABE': '#EF8B25'}
    for col, length in enumerate(lengths):
        ax = fig.add_axes([.08 + col * .307, .423, .264, .208])
        ax.axvspan(2.5, 10.5, color='#EFF2F5', zorder=0)
        for editor in ['ABE8e', 'TeABE', 'D699']:
            line = window[(window.editor == editor) & (window.length_group_id == length)].sort_values('position')
            ax.errorbar(line.position, line.observed_mean_pp, yerr=line.observed_sem_pp, color=editor_colors[editor],
                marker='o', lw=1.7, ms=3.6, elinewidth=.8, capsize=2)
        ax.set(xlim=(.6, 20.4), ylim=(-3, 106), xticks=[1, 5, 10, 15, 20], yticks=[0, 25, 50, 75, 100],
            title=length_labels[length])
        if col == 0:
            ax.set_ylabel('Observed A-to-G editing (%)')
        else:
            ax.tick_params(axis='y', labelleft=False)
        style(ax)
    fig.text(.519, .3919, 'Protospacer position', ha='center', va='center', fontsize=11)
    fig.legend(handles=[Line2D([], [], color=editor_colors[e], marker='o', lw=1.7,
        label='TeABE (49–64)' if e == 'TeABE' else e) for e in ['ABE8e', 'TeABE', 'D699']],
        loc='center', bbox_to_anchor=(.53, .367), ncol=3, fontsize=11, columnspacing=3)
    # Supplement c: each target's position MSE and the supplied target-macro mean.
    ax = fig.add_axes([.079, .085, .205, .217])
    means = transfer_lengths[transfer_lengths.model == 'with_distance'].set_index('length_group_id').loc[lengths]
    for i, length in enumerate(lengths):
        values = transfer_conditions[(transfer_conditions.model == 'with_distance') &
            (transfer_conditions.length_group_id == length)].sort_values('target_id').mse_pp2.to_numpy()
        ax.scatter(i + np.linspace(-.12, .12, len(values)), values, s=24, color='#7A929F',
            edgecolors='white', linewidths=.4, alpha=.85, zorder=3)
        mean = means.loc[length, 'macro_mse_pp2']
        ax.plot([i - .25, i + .25], [mean, mean], color=colors['delta_nearest'], lw=3.2, zorder=4)
        ax.text(i, mean + 23, f'{mean:.2f}', ha='center', fontsize=10, color=colors['delta_nearest'], weight='bold')
    ax.set(xticks=[0, 1, 2], xticklabels=[length_labels[k] for k in lengths], xlim=(-.5, 2.5), ylim=(0, 620),
        ylabel='Position MSE (pp²)', xlabel='sgRNA length')
    style(ax)
    # Supplement d: full-window TeABE observed and transferred predictions.
    for col, length in enumerate(lengths):
        ax = fig.add_axes([.362 + col * .205, .085, .175, .217])
        ax.axvspan(2.5, 10.5, color='#EFF2F5', zorder=0)
        line = transfer_curves[(transfer_curves.model == 'with_distance') &
            (transfer_curves.length_group_id == length)].sort_values('position')
        ax.plot(line.position, line.true_mean_pp, '-o', color=observed_color, lw=1.55, ms=3.4)
        ax.plot(line.position, line.pred_mean_pp, '-D', color=colors['delta_nearest'], lw=1.55, ms=3.2)
        ax.set(xlim=(.6, 20.4), ylim=(-2, 105), xticks=[1, 5, 10, 15, 20], yticks=[0, 25, 50, 75, 100],
            title=length_labels[length])
        if col == 0:
            ax.set_ylabel('A-to-G editing (%)')
        else:
            ax.tick_params(axis='y', labelleft=False)
        style(ax)
    fig.text(.6545, .0539, 'Protospacer position', ha='center', va='center', fontsize=11)
    fig.legend(handles=[Line2D([], [], color='#7A929F', marker='o', lw=0, label='Target'),
        Line2D([], [], color=colors['delta_nearest'], lw=3, label='Mean')], loc='center',
        bbox_to_anchor=(.185, .023), ncol=2, fontsize=10.5, columnspacing=1.5)
    fig.legend(handles=[Line2D([], [], color=observed_color, marker='o', lw=1.55, label='Observed'),
        Line2D([], [], color=colors['delta_nearest'], marker='D', lw=1.55, label='Predicted')], loc='center',
        bbox_to_anchor=(.653, .023), ncol=2, fontsize=11, columnspacing=2.5)
    save_figure(fig, 'FigS1_LDpred-BE')

# A compact run record supports later integration with the training/summary workflow.
record = {'figure': args.figure, 'config': config, 'dpi': dpi, 'formats': formats, 'inputs': input_hashes,
    'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
    'model_order': order, 'main_c_legend_order': ['delta_nearest', 'length', 'original', 'observed'],
    'main_d_difference': 'original bounded outcome MSE minus delta_nearest bounded outcome MSE',
    'layout_warnings': layout_warnings, 'outputs': saved_files,
    'typography': {'family': 'DejaVu Sans', 'size_mode': 'approved_original'}}
(output_root / 'provenance').mkdir(exist_ok=True)
(output_root / 'provenance' / f'plot_run_{args.figure}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
if layout_warnings:
    print('Text outside the canvas:', layout_warnings)

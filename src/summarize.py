#!/usr/bin/env python3
"""Average all 25 members' two raw outputs, then score all held-out targets together."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import MEMBERS, EDITORS, CANDIDATES, RAW, read_input, split, save_table, save_json
from evaluation import evaluate_frame


def summarize(work):
    tables = {name: [] for name in ['ensemble_predictions', 'position_predictions', 'condition_metrics',
        'full_window_distributions', 'fold_checkpoints', 'metrics_by_length', 'target_metrics', 'member_inventory']}
    results = []
    for editor in EDITORS:
        labels = read_input(work, editor).set_index('outcome_id').sort_index()
        common = labels[['condition_id', 'target_id', 'length_group_id', 'outcome_key']].copy()
        common['true_editing_efficiency'] = labels.editing_efficiency
        common['true_outcome_frequency'] = labels.outcome_frequency
        splits = {split(labels.target_id.unique(), f)[2]: (f, split(labels.target_id.unique(), f)[1]) for f in range(1, 13)}
        common['fold'] = common.target_id.map({t: v[0] for t, v in splits.items()})
        common['test_target'] = common.target_id
        common['validation_target'] = common.target_id.map({t: v[1] for t, v in splits.items()})
        for candidate in ['zero', *CANDIDATES]:
            values = []
            if candidate != 'zero':
                for member in MEMBERS:
                    stage = work / 'runs' / editor / member / ('length' if candidate == 'original' else candidate)
                    identity = json.loads((stage / 'completed.json').read_text())
                    if identity != dict(editor=editor, member=member, candidate='length' if candidate == 'original' else candidate, n_folds=12):
                        raise ValueError(f'Incomplete or wrong training stage: {stage}')
                    name = 'original_predictions.tsv' if candidate == 'original' else 'test_predictions.tsv'
                    pred = pd.read_csv(stage / name, sep='\t', dtype={'outcome_key': str}).set_index('outcome_id').sort_index()
                    metadata = ['condition_id', 'target_id', 'length_group_id', 'outcome_key', 'fold', 'test_target', 'validation_target']
                    if not pred.index.equals(common.index) or not pred[metadata].equals(common[metadata]):
                        raise ValueError(f'Missing or misassigned held-out outcomes: {stage}')
                    values.append(pred[RAW].to_numpy(float))
                    tables['member_inventory'].append(pd.DataFrame([dict(editor=editor, backbone=member,
                        candidate=candidate, n_folds=12, n_outcomes=len(pred))]))
                    if candidate != 'original':
                        folds = pd.read_csv(stage / 'folds.tsv', sep='\t')
                        if len(folds) != 12 or not folds.checkpoint_metric.eq('joint_mse').all() or (folds.best_epoch < 1).any():
                            raise ValueError(f'Expected 12 trained joint-MSE checkpoints: {stage}')
                        tables['fold_checkpoints'].append(folds.assign(editor=editor, backbone=member, candidate=candidate))
            frame = common.reset_index()
            frame[RAW] = np.stack(values).mean(axis=0) if values else np.zeros((len(frame), 2))
            score, pred, pos, cond, full = evaluate_frame(frame)
            identity = dict(editor=editor, candidate=candidate, n_backbones=25 if values else 0)
            results.append(dict(**identity, **score))
            for name, table in [('ensemble_predictions', pred), ('position_predictions', pos),
                                ('condition_metrics', cond), ('full_window_distributions', full)]:
                tables[name].append(table.assign(**identity))
            for key, name in [('length_group_id', 'metrics_by_length'), ('target_id', 'target_metrics')]:
                for value, group in frame.groupby(key, sort=True):
                    scores, *_ = evaluate_frame(group)
                    tables[name].append(pd.DataFrame([dict(**identity, **{key: value}, **scores)]))
    output = work / 'summary'
    total = pd.DataFrame(results)
    save_table(output / 'total_results.tsv', total)
    for name, frames in tables.items():
        save_table(output / f'{name}.tsv', pd.concat(frames, ignore_index=True))
    columns = ['editor', 'candidate', 'paper_R_2', 'paper_rho_2', 'bounded_position_mse_pp2',
               'bounded_outcome_mse_pp2', 'bounded_outcome_mae_pp', 'macro_distribution_tv_pp',
               'n_backbones', 'n_outcomes_mse_denominator', 'n_positions_mse_denominator']
    main = total[total.candidate.isin(CANDIDATES)][columns].copy()
    main['candidate'] = pd.Categorical(main.candidate, categories=CANDIDATES, ordered=True)
    main = main.sort_values(['editor', 'candidate'])
    if not np.isfinite(main[columns[2:8]].to_numpy(float)).all():
        raise ValueError('A required metric is nonfinite')
    save_table(output / 'main_four_models.tsv', main)
    save_json(output / 'evaluation_settings.json', dict(editors=EDITORS, candidates=CANDIDATES,
        n_members=25, n_folds=12, ensemble='mean both raw outputs before postprocessing',
        correlation='two-output R_2/rho_2; not R-squared', mse='bounded outcome and position frequencies in pp²',
        n_outcomes_per_editor=984, n_positions_per_editor=159, window='positions 3-10'))
    print(main.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    summarize(p.parse_args().work)

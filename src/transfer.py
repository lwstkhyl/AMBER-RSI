#!/usr/bin/env python3
"""TeABE full20 position transfer; fit only the established small fractional logit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import MEMBERS, EDITORS, KEY, BITS, RAW, split, read_input, save_table, save_json, metric_tables
from metrics import bound_frequencies
from regression import fit, predict


def source_activity(work, fold):
    """Use one fixed source fold for ALL targets, including regression training targets."""
    rows = []
    for editor in EDITORS:
        frame = read_input(work, editor).sort_values(KEY + ['outcome_key']).reset_index(drop=True)
        _, validation, test = split(frame.target_id.unique(), fold)
        total = np.zeros((len(frame), 2))
        for member in MEMBERS:
            path = work / 'runs' / editor / member / 'delta_nearest/source_predictions' / f'fold{fold:02d}.npz'
            with np.load(path, allow_pickle=False) as saved:
                if str(saved['test_target'][0]) != test or str(saved['validation_target'][0]) != validation:
                    raise ValueError(f'Source and transfer target splits differ: {path}')
                raw = pd.DataFrame(saved['raw'], index=saved['outcome_id'], columns=RAW).reindex(frame.outcome_id)
                if not np.isfinite(raw.to_numpy()).all():
                    raise ValueError(f'Incomplete source predictions: {path}')
                total += raw.to_numpy() / len(MEMBERS)
        frame[RAW] = total
        for _, group in frame.groupby('condition_id', sort=True):
            marginal = bound_frequencies(group.raw_pred_freq) @ group[BITS].to_numpy(float)
            editable = np.array(list(group.target_20mer.iloc[0][2:10])) == 'A'
            rows.append(dict(target_id=group.target_id.iloc[0], length_group_id=group.length_group_id.iloc[0],
                             source_editor=editor, activity_pp=float(marginal[editable].mean())))
    activity = pd.DataFrame(rows).pivot(index=KEY, columns='source_editor', values='activity_pp')
    return activity.rename(columns={'ABE8e': 'source_ABE8e_activity_pp', 'D699': 'source_D699_activity_pp'}).reset_index()


def run_fold(work, fold):
    labels = pd.read_csv(work / 'data/TeABE_positions.tsv', sep='\t')
    train_targets, validation, test = split(labels.target_id.unique(), fold)
    activity = source_activity(work, fold)
    frame = labels.merge(activity, on=KEY, validate='many_to_one')
    frame['partition'] = 'train'
    frame.loc[frame.target_id.eq(validation), 'partition'] = 'validation'
    frame.loc[frame.target_id.eq(test), 'partition'] = 'test'
    frame['fold'] = fold
    folder = work / 'transfer' / f'fold{fold:02d}'
    predictions, selections, coefficients = [], [], []
    for name, with_distance in [('with_distance', True), ('without_distance', False)]:
        model, trials = fit(frame[frame.partition.eq('train')], frame[frame.partition.eq('validation')], with_distance)
        save_json(folder / f'{name}_model.json', model)
        predictions.append(frame.assign(model=name, pred_position_pp=predict(frame, model), selected_penalty=model['penalty']))
        selections.extend(dict(model=name, fold=fold, selected=t['penalty'] == model['penalty'], **t) for t in trials)
        for i, feature in enumerate(['intercept', *model['features']]):
            coefficients.append(dict(model=name, fold=fold, feature=feature,
                coefficient_standardized=model['coefficients_standardized'][i],
                coefficient_raw=model['intercept_raw'] if i == 0 else model['coefficients_raw'][i - 1],
                selected_penalty=model['penalty'], validation_mse_pp2=model['validation_mse_pp2']))
    prediction = pd.concat(predictions, ignore_index=True)
    for part in ['train', 'validation', 'test']:
        save_table(folder / f'{part}_position_predictions.tsv', prediction[prediction.partition.eq(part)])
    save_table(folder / 'source_activities.tsv', activity)
    save_table(folder / 'regression_selection.tsv', pd.DataFrame(selections))
    save_table(folder / 'regression_coefficients.tsv', pd.DataFrame(coefficients))
    save_json(folder / 'completed.json', dict(fold=fold, train_targets=train_targets, validation_target=validation,
        test_target=test, n_train_targets=10, no_TeABE_neural_training=True))
    print(f'Transfer fold {fold}: test={test}, validation={validation}', flush=True)


def summarize(work):
    predictions, selections, coefficients = [], [], []
    for fold in range(1, 13):
        folder = work / 'transfer' / f'fold{fold:02d}'
        for name, destination in [('test_position_predictions', predictions),
                                   ('regression_selection', selections), ('regression_coefficients', coefficients)]:
            table = pd.read_csv(folder / f'{name}.tsv', sep='\t')
            if name == 'test_position_predictions':
                _, _, test = split(pd.read_csv(work / 'data/TeABE_positions.tsv', sep='\t').target_id.unique(), fold)
                if set(table.target_id) != {test} or set(table.fold) != {fold} or set(table.partition) != {'test'}:
                    raise ValueError(f'Wrong held-out identity: {folder}')
            else:
                table['test_target'] = predictions[-1].target_id.iloc[0]
            destination.append(table)
    result = pd.concat(predictions, ignore_index=True)
    labels = pd.read_csv(work / 'data/TeABE_positions.tsv', sep='\t')
    expected = labels.set_index(KEY + ['position']).true_position_pp.sort_index()
    if set(result.model) != {'with_distance', 'without_distance'} or result.duplicated(['model', *KEY, 'position']).any():
        raise ValueError('Expected both complete transfer models')
    for _, group in result.groupby('model'):
        actual = group.set_index(KEY + ['position']).true_position_pp.sort_index()
        if not actual.index.equals(expected.index) or not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError('Missing transfer positions or changed labels')
    conditions, targets = metric_tables(result)
    overall = targets.groupby('model', as_index=False).agg(macro_mse_pp2=('mse_pp2', 'mean'),
        macro_mae_pp=('mae_pp', 'mean'), n_targets=('target_id', 'nunique'))
    overall['macro_rmse_pp'] = np.sqrt(overall.macro_mse_pp2)
    for i, row in overall.iterrows():
        group = result[result.model.eq(row.model)]
        overall.loc[i, 'pooled_mse_pp2'] = np.mean((group.pred_position_pp - group.true_position_pp) ** 2)
        overall.loc[i, 'pooled_mae_pp'] = np.mean(np.abs(group.pred_position_pp - group.true_position_pp))
        overall.loc[i, 'n_A_positions'] = len(group)
    lengths = conditions.groupby(['model', 'length_group_id'], as_index=False).agg(
        macro_mse_pp2=('mse_pp2', 'mean'), macro_mae_pp=('mae_pp', 'mean'),
        n_targets=('target_id', 'nunique'), n_A_positions=('n_A_positions', 'sum'))
    curves = result.groupby(['model', 'length_group_id', 'position'], as_index=False).agg(
        true_mean_pp=('true_position_pp', 'mean'), pred_mean_pp=('pred_position_pp', 'mean'),
        n_A_targets=('target_id', 'nunique'))
    gains = targets.pivot(index=['fold', 'target_id'], columns='model', values=['mse_pp2', 'mae_pp'])
    gains.columns = ['_'.join(c) for c in gains.columns]
    gains = gains.reset_index()
    gains['mse_improvement_pp2'] = gains.mse_pp2_without_distance - gains.mse_pp2_with_distance
    gains['mae_improvement_pp'] = gains.mae_pp_without_distance - gains.mae_pp_with_distance
    for name, table in [('test_position_predictions', result), ('overall_metrics', overall),
                        ('condition_metrics', conditions), ('target_metrics', targets), ('metrics_by_length', lengths),
                        ('editing_window_curve', curves), ('distance_target_gains', gains),
                        ('regression_selection', pd.concat(selections, ignore_index=True)),
                        ('regression_coefficients', pd.concat(coefficients, ignore_index=True))]:
        save_table(work / 'transfer/summary' / f'{name}.tsv', table)
    save_json(work / 'transfer/summary/summary_settings.json', dict(primary_model='with_distance',
        primary_metric='macro position MSE: mean A sites, then lengths, then targets',
        n_positions_per_model=len(expected), n_folds=12, n_source_members=25,
        source='same-fold freshly trained ABE8e/D699 Length+delta-nearest predictions',
        no_TeABE_neural_training=True, no_outcome_distribution=True))
    print(overall.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--fold', type=int, choices=range(1, 13))
    a = p.parse_args()
    run_fold(a.work, a.fold) if a.fold else summarize(a.work)

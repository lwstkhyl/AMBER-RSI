#!/usr/bin/env python3
"""Train one editor/backbone/branch through all 12 held-out targets."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tensorflow as tf
from common import (SEED, EPOCHS, PATIENCE, BATCH_SIZE, LEARNING_RATE, RAW,
                    read_input, condition_features, arrays, split, save_table, save_json)
from joint_training import train_joint, selected_validation_metrics
from network import Backbone
from state import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--editor', choices=['ABE8e', 'D699'], required=True)
    parser.add_argument('--member', required=True)
    parser.add_argument('--candidate', choices=['length', 'single_delta_nearest', 'delta_nearest'], required=True)
    a = parser.parse_args()
    config = json.loads((a.work / 'run_config.json').read_text())
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    frame = read_input(a.work, a.editor)
    conditions, delta = condition_features(frame)
    backbone = Backbone(str(Path(config['pretrained_root']) / a.member))
    base = a.work / 'runs' / a.editor / a.member
    out = base / a.candidate
    if (out / 'completed.json').exists():
        raise ValueError(f'This stage is already complete; use a fresh run directory: {out}')
    (out / 'branch_weights').mkdir(parents=True, exist_ok=True)
    worker = runtime()
    expected = json.loads((a.work / 'environment.json').read_text())
    if any(worker[k] != expected[k] for k in worker if k not in ['host', 'cpu']):
        raise ValueError('Worker software or deterministic settings differ from preparation')
    save_json(out / 'worker_environment.json', worker)
    args = SimpleNamespace(seed=SEED, outdir=str(out), epochs=EPOCHS, patience=PATIENCE,
                           learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    prediction_rows, original_rows, histories, folds = [], [], [], []
    for fold in range(1, 13):
        train_targets, validation, test = split(frame.target_id.unique(), fold)
        train_table = conditions[conditions.target_id.isin(train_targets)]
        mean = float(train_table.sgRNA_length_numeric.mean())
        sd = float(train_table.sgRNA_length_numeric.std(ddof=0))
        sd = sd if sd >= 1e-8 else 1.0
        scale = delta.loc[train_table.condition_id].to_numpy(float).std(axis=0, ddof=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        structural = a.candidate != 'length'
        data = []
        for cid in conditions.condition_id:
            g = frame[frame.condition_id.eq(cid)].sort_values('outcome_key').reset_index(drop=True)
            data.append(dict(frame=g, inputs=[tf.constant(v) for v in arrays(
                g, mean, sd, delta if structural else None, scale)],
                true_eff=tf.constant(g.editing_efficiency.to_numpy(np.float32)),
                true_freq=tf.constant(g.outcome_frequency.to_numpy(np.float32))))
        training = [c for c in data if c['frame'].target_id.iloc[0] in train_targets]
        valid = [c for c in data if c['frame'].target_id.iloc[0] == validation]
        held = [c for c in data if c['frame'].target_id.iloc[0] == test]
        fold_seed = SEED + 100 * fold
        tf.keras.utils.set_random_seed(fold_seed)
        length_weights = None
        if structural:
            with np.load(base / 'length/branch_weights' / f'length_fold{fold:02d}.npz') as saved:
                if str(saved['test_target'][0]) != test or str(saved['validation_target'][0]) != validation:
                    raise ValueError('Length and distance stages have different target splits')
                length_weights = [saved['length_kernel'], saved['length_bias']]
                if a.candidate == 'single_delta_nearest':
                    length_weights = [np.zeros_like(v) for v in length_weights]
        model = backbone.build(fold_seed, length_weights)
        if structural:
            if a.candidate == 'single_delta_nearest':
                expected_raw = backbone.original(held[0]['inputs'][:5], training=False).numpy()
            else:
                previous = pd.read_csv(base / 'length/test_predictions.tsv', sep='\t').set_index('outcome_id')
                expected_raw = previous.loc[held[0]['frame'].outcome_id, RAW].to_numpy(float)
            initial_raw = model(held[0]['inputs'], training=False).numpy()
            if np.max(np.abs(initial_raw - expected_raw)) > 1e-3:
                raise ValueError('Zero distance projection must reproduce its preceding model')
        if not structural:
            before = model(held[0]['inputs'], training=False).numpy()
            original = backbone.original(held[0]['inputs'][:5], training=False).numpy()
            if np.max(np.abs(before - original)) > 1e-4:
                raise ValueError('Zero length branch must reproduce the original backbone')
        print(f'Fold {fold}: test={test}, validation={validation}', flush=True)
        history, score, best_epoch = train_joint(model, training, valid, fold_seed, args)
        histories.append(history.assign(fold=fold, test_target=test, validation_target=validation))
        parameters = dict(length_mean=np.array([mean]), length_sd=np.array([sd]),
                          test_target=np.array([test]), validation_target=np.array([validation]))
        if structural:
            parameters.update(bottleneck_kernel=model.get_layer('DeltaDistanceBottleneck').get_weights()[0],
                              projection_kernel=model.get_layer('DeltaDistanceToDense2').get_weights()[0],
                              delta_sd=scale)
        else:
            kernel, bias = model.get_layer('LengthToDense2').get_weights()
            parameters.update(length_kernel=kernel, length_bias=bias)
        np.savez_compressed(out / 'branch_weights' / f'fold{fold:02d}.npz', **parameters)
        if not structural:
            # Keep the familiar length filename for the following two structural branches.
            (out / 'branch_weights' / f'fold{fold:02d}.npz').rename(
                out / 'branch_weights' / f'length_fold{fold:02d}.npz')
        all_source = []
        for c in data if a.candidate == 'delta_nearest' else held:
            g = c['frame']
            raw = model(c['inputs'], training=False).numpy()
            p = g[['outcome_id', 'condition_id', 'target_id', 'length_group_id', 'target_30mer', 'outcome_key']].copy()
            p[['true_editing_efficiency', 'true_outcome_frequency']] = np.column_stack([
                c['true_eff'].numpy(), c['true_freq'].numpy()])
            p[RAW] = raw
            p['fold'], p['test_target'], p['validation_target'] = fold, test, validation
            if g.target_id.iloc[0] == test:
                prediction_rows.append(p)
            if a.candidate == 'delta_nearest':
                all_source.append(p[['outcome_id', *RAW]])
            if not structural:
                original = p.copy()
                original[RAW] = backbone.original(c['inputs'][:5], training=False).numpy()
                original_rows.append(original)
        if all_source:
            source = pd.concat(all_source, ignore_index=True).sort_values('outcome_id')
            folder = out / 'source_predictions'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f'fold{fold:02d}.npz',
                outcome_id=source.outcome_id.to_numpy(str), raw=source[RAW].to_numpy(float),
                test_target=np.array([test]), validation_target=np.array([validation]))
        folds.append(dict(fold=fold, test_target=test, validation_target=validation, train_targets=10,
                          length_mean=mean, length_sd=sd, best_epoch=best_epoch, epochs_ran=len(history),
                          checkpoint_metric='joint_mse', best_validation_joint_mse=score,
                          **selected_validation_metrics(model, valid)))
        del model
        gc.collect()
    save_table(out / 'test_predictions.tsv', pd.concat(prediction_rows, ignore_index=True))
    if original_rows:
        save_table(out / 'original_predictions.tsv', pd.concat(original_rows, ignore_index=True))
    save_table(out / 'history.tsv', pd.concat(histories, ignore_index=True))
    save_table(out / 'folds.tsv', pd.DataFrame(folds))
    save_json(out / 'completed.json', dict(editor=a.editor, member=a.member, candidate=a.candidate, n_folds=12))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Restore a 25-member full-data model; input requires all nonempty A3-10 outcomes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from manage import load_plan, read_json


def predict(run, editor, input_path, output_path):
    plan = load_plan(run)
    import tensorflow as tf
    from common import BITS, DIST, KEY, RAW, arrays, save_table
    from metrics import bound_frequencies, paper_normalize
    from network import Backbone
    from state import sha
    if read_json(run / 'completed.json').get('complete') is not True:
        raise ValueError('Finish all 50 deployment members before inference')
    tf.keras.utils.set_random_seed(plan['seed'])
    tf.config.experimental.enable_op_determinism()
    frame = pd.read_csv(input_path, sep='\t', dtype={'outcome_key': str})
    frame['outcome_key'] = frame[BITS].astype(int).astype(str).agg(''.join, axis=1)
    if frame.outcome_id.duplicated().any():
        raise ValueError('outcome_id must be unique')
    columns = ['condition_id', *KEY, 'target_20mer', 'target_30mer', 'CRISPRoff_score',
               'CRISPRon_score', 'sgRNA_length_numeric', *DIST]
    conditions = frame[columns].drop_duplicates()
    if conditions.condition_id.duplicated().any() or conditions.duplicated(KEY).any():
        raise ValueError('Each target/length must have one consistent feature vector')
    reference = conditions[conditions.length_group_id.eq('L20')].set_index('target_id')[DIST]
    delta = pd.DataFrame(conditions[DIST].to_numpy(float) -
                         reference.reindex(conditions.target_id).to_numpy(float),
                         index=conditions.condition_id, columns=DIST)
    if not np.isfinite(delta.to_numpy()).all():
        raise ValueError('Provide finite distances and the L20 reference for every target')
    groups = []
    for cid in conditions.condition_id:
        g = frame[frame.condition_id.eq(cid)].sort_values('outcome_key')
        bits = g[BITS].to_numpy(float)
        editable = np.array(list(g.target_20mer.iloc[0][2:10])) == 'A'
        if (not np.isin(bits, [0, 1]).all() or (bits.sum(axis=1) == 0).any() or
                (bits[:, ~editable] != 0).any() or g.outcome_key.duplicated().any() or
                len(g) != 2 ** int(editable.sum()) - 1):
            raise ValueError(f'Provide all nonempty, valid A3-10 outcomes: {cid}')
        groups.append(g)
    ordered = pd.concat(groups, ignore_index=True)
    predictions = []
    for member in plan['members']:
        for name, digest in plan['source_environment']['pretrained_files'].items():
            if name.startswith(member + '/') and sha(Path(plan['pretrained_root']) / name) != digest:
                raise ValueError(f'Original backbone changed: {name}')
        with np.load(run / 'members' / editor / member / 'weights.npz', allow_pickle=False) as w:
            model = Backbone(str(Path(plan['pretrained_root']) / member)).build(
                plan['seed'], [w['length_kernel'], w['length_bias']])
            model.get_layer('DeltaDistanceBottleneck').set_weights([w['bottleneck_kernel']])
            model.get_layer('DeltaDistanceToDense2').set_weights([w['projection_kernel']])
            inputs = [arrays(g, float(w['length_mean'][0]), float(w['length_sd'][0]), delta, w['delta_sd'])
                      for g in groups]
            raw = [model(values, training=False).numpy().astype(float) for values in inputs]
            predictions.append(np.concatenate(raw))
        tf.keras.backend.clear_session()
    result = ordered[['outcome_id', 'condition_id', *KEY, 'outcome_key']].copy()
    result[RAW] = np.stack(predictions).mean(axis=0)
    result['bounded_pred_freq'] = np.nan
    for _, g in result.groupby('condition_id', sort=False):
        result.loc[g.index, 'bounded_pred_freq'] = bound_frequencies(g.raw_pred_freq)
    try:
        paper = paper_normalize(result.rename(columns={'raw_pred_eff': 'pred_eff', 'raw_pred_freq': 'pred_freq'}))
        result['paper_pred_eff'], result['paper_pred_freq'] = paper.pred_eff, paper.pred_freq
        result['paper_normalization_status'] = 'ok'
    except ZeroDivisionError:
        result['paper_pred_eff'] = result.groupby('condition_id').raw_pred_eff.transform('mean')
        result['paper_pred_freq'] = np.nan
        result['paper_normalization_status'] = 'failed_zero_positive_frequency_denominator'
    result['editor'] = editor
    save_table(output_path, result)
    print(f'Saved full-data ensemble predictions: {output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--editor', choices=['ABE8e', 'D699'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    predict(a.run, a.editor, a.input, a.out)

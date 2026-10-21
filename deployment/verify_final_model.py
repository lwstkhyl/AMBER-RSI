#!/usr/bin/env python3
"""Read-only deployment audit and label-free prediction round trip; no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

RAW = ['raw_pred_eff', 'raw_pred_freq']
RAW_ATOL = 1e-3  # percentage points; inference uses different batch sizes from refit.


def check_predictions(frame, prediction, saved_mean):
    expected_ids = pd.Index(frame.outcome_id)
    if prediction.outcome_id.duplicated().any() or set(prediction.outcome_id) != set(expected_ids):
        raise ValueError('Restored predictions have missing or duplicate outcome IDs')
    actual = prediction.set_index('outcome_id').loc[expected_ids]
    expected = saved_mean.loc[expected_ids, RAW].to_numpy(float)
    raw = actual[RAW].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite restored or saved raw predictions')
    difference = float(np.max(np.abs(raw - expected)))
    if difference > RAW_ATOL:
        raise ValueError(f'Restored raw predictions differ by {difference:.9g}pp; limit={RAW_ATOL}pp')
    bounded = actual.bounded_pred_freq.to_numpy(float)
    if not np.isfinite(bounded).all() or (bounded < -1e-8).any() or (bounded > 100 + 1e-8).any():
        raise ValueError('Bounded outcome frequencies are not finite percentages')
    values = actual.reset_index()
    sums = values.groupby('condition_id').bounded_pred_freq.sum()
    if (sums > 100 + 1e-8).any():
        raise ValueError('Nonempty outcome frequencies exceed 100%')
    bits = [f'outcome_pos{i}' for i in range(3, 11)]
    metadata = frame.set_index('outcome_id').loc[actual.index]
    if not np.array_equal(actual.condition_id.to_numpy(), metadata.condition_id.to_numpy()):
        raise ValueError('Restored outcome-to-condition assignments changed')
    for cid in metadata.condition_id.unique():
        mask = metadata.condition_id.eq(cid).to_numpy()
        marginal = bounded[mask] @ metadata.loc[mask, bits].to_numpy(float)
        if (marginal < -1e-8).any() or (marginal > 100 + 1e-8).any():
            raise ValueError('Position marginals are outside 0-100%')
    return dict(n_outcomes=len(raw), n_conditions=len(sums), raw_roundtrip_pass=True,
        raw_identical_values=bool(np.array_equal(raw, expected)), raw_max_abs_difference_pp=difference,
        bounded_frequencies_valid=True, min_nonempty_frequency_sum_pp=float(sums.min()),
        max_nonempty_frequency_sum_pp=float(sums.max()),
        paper_normalization_ok=bool(actual.paper_normalization_status.eq('ok').all()))


def verify(code, run, output):
    # Import the released entry points without adding files to their directories.
    sys.path.insert(0, str(code))
    from manage import load_plan, read_json
    plan = load_plan(run)
    global RAW_ATOL
    from settings import CONFIG
    RAW_ATOL = float(CONFIG['comparison']['roundtrip_atol_pp'])
    from common import BITS, DIST, KEY, read_input
    from predict import predict
    completed = read_json(run / 'completed.json')
    if completed.get('complete') is not True or completed.get('n_members') != 50:
        raise ValueError('Expected a complete, 50-member deployment run')
    # predict() itself verifies all 25 original-backbone fingerprints per editor.
    output.mkdir(parents=True, exist_ok=False)
    report = dict(purpose='implementation audit; training inputs are not held-out tests',
        code=str(code), run=str(run), seed=plan['seed'], raw_atol_pp=RAW_ATOL, raw_rtol=0,
        note='Small floating-point differences may arise from condition-wise versus full-table inference batches.',
        editors={})
    features = ['outcome_id', 'condition_id', *KEY, 'target_20mer', 'target_30mer',
                'CRISPRoff_score', 'CRISPRon_score', 'sgRNA_length_numeric', *BITS, *DIST]
    for editor in plan['editors']:
        frame = read_input(run, editor)
        conditions = frame[['condition_id', *KEY, 'sgRNA_length_numeric', *DIST]].drop_duplicates()
        if conditions.condition_id.duplicated().any() or len(conditions) != 36 or frame.target_id.nunique() != 12:
            raise ValueError('Expected 36 consistent conditions on all 12 targets')
        reference = conditions[conditions.length_group_id.eq('L20')].set_index('target_id')[DIST]
        delta = conditions[DIST].to_numpy(float) - reference.reindex(conditions.target_id).to_numpy(float)
        if not np.isfinite(delta).all() or not np.all(delta[conditions.length_group_id.eq('L20')] == 0):
            raise ValueError('Invalid distances or nonzero L20 delta')
        mean = float(conditions.sgRNA_length_numeric.mean())
        sd = float(conditions.sgRNA_length_numeric.std(ddof=0))
        sd = sd if sd >= 1e-8 else 1.0
        scale = delta.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-8, 1, scale)
        expected_index = pd.Index(frame.outcome_id)
        member_raw = []
        for task in [t for t in plan['tasks'] if t['editor'] == editor]:
            folder = run / 'members' / editor / task['member']
            for branch in ['length', 'delta']:
                history = pd.read_csv(folder / f'{branch}_history.tsv', sep='\t')
                epochs = task[f'{branch}_epochs']
                if not np.array_equal(history.epoch.to_numpy(), np.arange(1, epochs + 1)) or not np.isfinite(history.train_joint_mse).all():
                    raise ValueError(f'Wrong training history: {editor}/{task["member"]}/{branch}')
            with np.load(folder / 'weights.npz', allow_pickle=False) as w:
                expected_parameters = dict(length_mean=[mean], length_sd=[sd], delta_sd=scale,
                    seed=[plan['seed']], length_epochs=[task['length_epochs']], delta_epochs=[task['delta_epochs']])
                if any(not np.isfinite(w[k]).all() for k in w.files):
                    raise ValueError(f'Nonfinite weights: {folder}')
                for name, expected in expected_parameters.items():
                    if w[name].shape != np.asarray(expected).shape or not np.allclose(w[name], expected, atol=1e-10, rtol=1e-10):
                        raise ValueError(f'Scaler/seed/epoch mismatch: {folder}/{name}')
            with np.load(folder / 'training_fit_predictions.npz', allow_pickle=False) as saved:
                if list(saved['raw_columns']) != RAW or saved['raw'].shape != (984, 2):
                    raise ValueError(f'Wrong saved raw prediction format: {folder}')
                member = pd.DataFrame(saved['raw'], index=saved['outcome_id'], columns=RAW)
                if member.index.has_duplicates or set(member.index) != set(expected_index):
                    raise ValueError(f'Wrong saved outcome IDs: {folder}')
                raw = member.loc[expected_index].to_numpy(float)
                if not np.isfinite(raw).all():
                    raise ValueError(f'Nonfinite saved predictions: {folder}')
                member_raw.append(raw)
        if len(member_raw) != 25:
            raise ValueError('Expected all 25 members for each editor')
        expected = pd.DataFrame(np.stack(member_raw).mean(axis=0), index=expected_index, columns=RAW)
        input_path, prediction_path = output / f'{editor}_features_only.tsv', output / f'{editor}_restored.tsv'
        frame[features].to_csv(input_path, sep='\t', index=False)
        predict(run, editor, input_path, prediction_path)
        restored = pd.read_csv(prediction_path, sep='\t', dtype={'outcome_key': str})
        report['editors'][editor] = dict(n_members=25, n_train_targets=12,
            scalers_and_epoch_counts_valid=True, L20_delta_zero=True, inference_without_true_labels=True,
            **check_predictions(frame, restored, expected))
        print(editor, json.dumps(report['editors'][editor]), flush=True)
    report['all_passed'] = True
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'PASS: {output / "verification.json"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verify(a.code.resolve(), a.run.resolve(), a.out.resolve())

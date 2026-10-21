"""Fixed evaluation definitions; training and checkpoint selection never read these test scores."""
import numpy as np
import pandas as pd
from metrics import custom_metrics, paper_metrics, paper_normalize

def positions(frame, prediction):
    rows = []
    for cid, g in frame.groupby('condition_id', sort=True):
        bits = np.array([[int(x) for x in key] for key in g.outcome_key], dtype=int)
        truth = g.true_outcome_frequency.to_numpy(float) @ bits
        values = g[prediction].to_numpy(float) @ bits
        for j in np.where(bits.any(axis=0))[0]:
            rows.append(dict(condition_id=cid, target_id=g.target_id.iloc[0],
                length_group_id=g.length_group_id.iloc[0], fold=int(g.fold.iloc[0]), position=j + 3,
                true_position_efficiency=float(truth[j]), predicted_position_efficiency=float(values[j])))
    return pd.DataFrame(rows)

def evaluate_frame(frame):
    frame = frame.copy().reset_index(drop=True)
    if not np.isfinite(frame[['raw_pred_eff', 'raw_pred_freq', 'true_editing_efficiency', 'true_outcome_frequency']]).all().all():
        raise ValueError('Nonfinite raw predictions or truth')
    raw = frame.rename(columns={'raw_pred_eff': 'pred_eff', 'raw_pred_freq': 'pred_freq'})
    raw_scores = paper_metrics(raw)
    custom, conditions, bounded = custom_metrics(raw, bounded=True)
    try:
        official = paper_normalize(raw)
        status = 'ok'
    except ZeroDivisionError:
        official = raw.copy()
        official['pred_eff'] = official.groupby('condition_id').pred_eff.transform('mean')
        official['pred_freq'] = np.nan
        status = 'failed_zero_positive_frequency_denominator'
    paper = paper_metrics(official)
    exported = frame.copy()
    exported['custom_bounded_pred_freq'] = bounded.pred_freq.to_numpy()
    exported['paper_pred_eff'] = official.pred_eff.to_numpy()
    exported['paper_pred_freq'] = official.pred_freq.to_numpy()
    exported['paper_normalization_status'] = status
    position_tables = []
    score = dict(**custom, paper_R_2=paper['R_2'], paper_rho_2=paper['rho_2'],
        paper_outcome_mae_pp=paper['outcome_mae_pp'], paper_outcome_mse_pp2=paper['outcome_mse'],
        paper_efficiency_mae_pp=paper['efficiency_mae_pp'], paper_efficiency_rmse_pp=paper['efficiency_rmse_pp'],
        raw_joint_mse_pp2=raw_scores['joint_mse'], raw_outcome_mse_pp2=raw_scores['outcome_mse'],
        raw_outcome_mae_pp=raw_scores['outcome_mae_pp'],
        bounded_outcome_mse_pp2=float(np.mean((bounded.pred_freq - bounded.true_outcome_frequency) ** 2)),
        macro_outcome_mse_pp2=float(conditions.groupby('target_id').mse.mean().mean()),
        paper_normalization_status=status)
    for processing, column in [('raw', 'raw_pred_freq'), ('bounded', 'custom_bounded_pred_freq'), ('paper', 'paper_pred_freq')]:
        pos = positions(exported, column)
        error = pos.predicted_position_efficiency - pos.true_position_efficiency
        score[f'{processing}_position_mae_pp'] = float(abs(error).mean())
        score[f'{processing}_position_mse_pp2'] = float(np.mean(error.to_numpy() ** 2))
        score['n_positions'] = len(pos)
        pos['processing'] = processing
        position_tables.append(pos)
    score['n_outcomes_mse_denominator'] = len(frame)
    score['n_positions_mse_denominator'] = score['n_positions']
    score['bounded_outcome_mae_pp'] = score['pooled_mae_pp']
    full = exported[['outcome_id', 'condition_id', 'target_id', 'length_group_id', 'outcome_key', 'fold',
                     'true_outcome_frequency', 'custom_bounded_pred_freq']].copy()
    complements = []
    for cid, g in full.groupby('condition_id', sort=True):
        complements.append(dict(outcome_id=cid + '|00000000', condition_id=cid, target_id=g.target_id.iloc[0],
            length_group_id=g.length_group_id.iloc[0], outcome_key='00000000', fold=int(g.fold.iloc[0]),
            true_outcome_frequency=max(0., 100 - g.true_outcome_frequency.sum()),
            custom_bounded_pred_freq=max(0., 100 - g.custom_bounded_pred_freq.sum())))
    full = pd.concat([full, pd.DataFrame(complements)], ignore_index=True)
    return score, exported, pd.concat(position_tables, ignore_index=True), conditions, full

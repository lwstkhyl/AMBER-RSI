"""Two-output paper metrics and the project's percentage-scale distribution metrics."""
import numpy as np
import pandas as pd

MAE_COLUMNS = ['pooled_mae_pp', 'macro_mae_pp', 'macro_frequency_weighted_mae_pp']

def correlation(x, y, rank=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(x) < 2:
        return np.nan
    if rank:
        x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    x, y = x - x.mean(), y - y.mean()
    denominator = np.sqrt(np.square(x).sum() * np.square(y).sum())
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan

def joint_correlations(truth, prediction):
    """Paper R_2/rho_2: equal dimension weights, centered within each column."""
    x, y = np.asarray(truth, float), np.asarray(prediction, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('Joint metrics require matching N-by-2 tables')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {'R_2': np.nan, 'rho_2': np.nan}
    result = {}
    for name, rank in [('R_2', False), ('rho_2', True)]:
        a, b = x.copy(), y.copy()
        if rank:
            a, b = pd.DataFrame(a).rank().to_numpy(), pd.DataFrame(b).rank().to_numpy()
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        denominator = np.sqrt(np.square(a).sum() * np.square(b).sum())
        result[name] = float((a * b).sum() / denominator) if denominator > 0 else np.nan
    return result

def paper_normalize(frame):
    """Match official normalize_dataset after averaging both raw ensemble outputs.

    Negative frequencies are zeroed, efficiency is averaged across outcomes,
    then frequencies are scaled to that efficiency. Efficiency is not clipped.
    A zero positive-frequency sum raises, as in the official implementation.
    """
    out = frame.copy()
    out['pred_eff'] = out.groupby('condition_id').pred_eff.transform('mean')
    frequency = out.pred_freq.clip(lower=0)
    total = frequency.groupby(out.condition_id).transform('sum')
    if total.eq(0).any():
        raise ZeroDivisionError('Official normalization has a zero positive-frequency denominator')
    out['pred_freq'] = frequency / total * out.pred_eff
    return out

def bound_frequencies(prediction):
    p = np.clip(np.asarray(prediction, float), 0, 100)
    return p / max(1.0, p.sum() / 100)

def distribution_tv_pp(truth, prediction):
    """Half-L1 including the no-window-A-to-G complement; range 0-100pp."""
    y, p = np.asarray(truth, float), bound_frequencies(prediction)
    if not np.isfinite(y).all() or (y < 0).any() or y.sum() > 100 + 1e-5:
        raise ValueError('Expected absolute outcome frequencies summing to <=100%')
    return float((np.abs(p - y).sum() + abs(p.sum() - y.sum())) / 2)

def paper_metrics(frame):
    truth = frame[['true_editing_efficiency', 'true_outcome_frequency']].to_numpy(float)
    pred = frame[['pred_eff', 'pred_freq']].to_numpy(float)
    result = joint_correlations(truth, pred)
    result['joint_mse'] = float(np.square(pred - truth).mean())
    result['efficiency_row_mse'] = float(np.square(pred[:, 0] - truth[:, 0]).mean())
    result['outcome_mse'] = float(np.square(pred[:, 1] - truth[:, 1]).mean())
    result['outcome_pearson'] = correlation(truth[:, 1], pred[:, 1])
    result['outcome_spearman'] = correlation(truth[:, 1], pred[:, 1], rank=True)
    result['outcome_mae_pp'] = float(np.abs(pred[:, 1] - truth[:, 1]).mean())
    efficiency = frame.groupby('condition_id').agg(
        truth=('true_editing_efficiency', 'first'), pred=('pred_eff', 'mean'))
    error = efficiency.pred - efficiency.truth
    result.update(efficiency_pearson=correlation(efficiency.truth, efficiency.pred),
                  efficiency_spearman=correlation(efficiency.truth, efficiency.pred, rank=True),
                  efficiency_mae_pp=float(error.abs().mean()),
                  efficiency_rmse_pp=float(np.sqrt(np.square(error).mean())),
                  n_nonfinite_prediction_rows=int((~np.isfinite(pred).all(axis=1)).sum()),
                  n_efficiency_outside_0_100=int((~efficiency.pred.between(0, 100)).sum()),
                  n_targets=frame.target_id.nunique(), n_conditions=len(efficiency), n_outcomes=len(frame),
                  normalization_status=frame.paper_normalization_status.iloc[0]
                      if 'paper_normalization_status' in frame else 'raw')
    return result

def custom_metrics(frame, bounded=True):
    out = frame.copy()
    if bounded:
        for _, g in out.groupby('condition_id'):
            out.loc[g.index, 'pred_freq'] = bound_frequencies(g.pred_freq)
    rows = []
    for cid, g in out.groupby('condition_id', sort=True):
        y, p = g.true_outcome_frequency.to_numpy(float), g.pred_freq.to_numpy(float)
        error = np.abs(p - y)
        rows.append(dict(condition_id=cid, target_id=g.target_id.iloc[0],
                         length_group_id=g.length_group_id.iloc[0], n_outcomes=len(g),
                         mae_pp=float(error.mean()), mse=float(np.square(error).mean()),
                         frequency_weighted_mae_pp=float(np.dot(y, error) / y.sum()) if y.sum() > 1e-12 else np.nan,
                         distribution_tv_pp=distribution_tv_pp(y, p) if bounded else np.nan,
                         true_window_editing_pp=float(y.sum()), pred_window_editing_pp=float(p.sum()),
                         window_editing_abs_error_pp=float(abs(p.sum() - y.sum()))))
    conditions = pd.DataFrame(rows)
    numeric = ['mae_pp', 'mse', 'frequency_weighted_mae_pp', 'distribution_tv_pp', 'window_editing_abs_error_pp']
    targets = conditions.groupby('target_id')[numeric].mean()
    mean = targets.mean()
    error = out.pred_freq - out.true_outcome_frequency
    result = dict(pooled_mae_pp=float(error.abs().mean()), macro_mae_pp=float(mean.mae_pp),
                  macro_frequency_weighted_mae_pp=float(mean.frequency_weighted_mae_pp),
                  macro_distribution_tv_pp=float(mean.distribution_tv_pp),
                  pooled_rmse_pp=float(np.sqrt(np.square(error).mean())),
                  macro_rmse_pp=float(np.sqrt(mean.mse)),
                  window_editing_mae_pp=float(mean.window_editing_abs_error_pp),
                  n_frequency_weighted_conditions=int(conditions.frequency_weighted_mae_pp.notna().sum()),
                  n_frequency_weighted_targets=int(targets.frequency_weighted_mae_pp.notna().sum()),
                  n_targets=len(targets), n_conditions=len(conditions), n_outcomes=len(out))
    return result, conditions, out

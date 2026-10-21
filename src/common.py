"""Shared experiment constants, target splits and small I/O helpers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from settings import CONFIG

MEMBERS = [f'model_{i}_{j}' for i in range(1, 6) for j in range(1, 6)]
EDITORS = ['ABE8e', 'D699']
CANDIDATES = ['original', 'length', 'single_delta_nearest', 'delta_nearest']
BITS = [f'outcome_pos{i}' for i in range(3, 11)]
DIST = [f'distance_pos{i}' for i in range(3, 11)]
KEY = ['target_id', 'length_group_id']
RAW = ['raw_pred_eff', 'raw_pred_freq']
SEED = int(CONFIG['training']['seed'])
EPOCHS = int(CONFIG['training']['epochs'])
PATIENCE = int(CONFIG['training']['patience'])
BATCH_SIZE = int(CONFIG['training']['batch_size'])
LEARNING_RATE = float(CONFIG['training']['learning_rate'])
BOTTLENECK = int(CONFIG['training']['bottleneck'])
PROTOCOL = f'joint_mse_seed{SEED}_position_logit_v1'


def split(targets, fold):
    targets = sorted(targets)
    test = targets[fold - 1]
    remaining = [t for t in targets if t != test]
    validation = remaining[(fold - 1) % len(remaining)]
    return [t for t in remaining if t != validation], validation, test


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False)


def read_input(work, editor):
    frame = pd.read_csv(Path(work) / 'inputs' / f'{editor}.tsv', sep='\t', dtype={'outcome_key': str})
    frame['outcome_key'] = frame[BITS].astype(int).astype(str).agg(''.join, axis=1)
    if len(frame) != 984 or frame.outcome_id.duplicated().any() or frame.condition_id.nunique() != 36:
        raise ValueError(f'{editor}: expected 984 unique outcomes and 36 conditions')
    return frame


def condition_features(frame):
    columns = ['condition_id', *KEY, 'target_30mer', 'CRISPRoff_score', 'CRISPRon_score',
               'sgRNA_length_numeric', 'editing_efficiency', *DIST]
    conditions = frame[columns].drop_duplicates('condition_id').reset_index(drop=True)
    reference = conditions[conditions.length_group_id.eq('L20')].set_index('target_id')[DIST]
    delta = conditions[DIST].to_numpy(float) - reference.reindex(conditions.target_id).to_numpy(float)
    return conditions, pd.DataFrame(delta, index=conditions.condition_id, columns=DIST)


def arrays(group, length_mean, length_sd, delta=None, delta_sd=None):
    """Use the original A,T,G,C channel order; divide delta by train SD without centering."""
    n, first = len(group), group.iloc[0]
    nt = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    sequence = np.eye(4, dtype=np.float32)[[nt[b] for b in first.target_30mer]][None, :, :]
    inputs = [np.repeat(sequence, n, axis=0), group[BITS].to_numpy(np.float32),
              np.full((n, 1), first.CRISPRoff_score, np.float32),
              np.full((n, 1), first.CRISPRon_score, np.float32),
              np.tile(np.array([0, 0, 0, 0, 1], np.float32), (n, 1)),
              np.full((n, 1), (float(first.sgRNA_length_numeric) - length_mean) / length_sd, np.float32)]
    if delta is not None:
        value = (delta.loc[first.condition_id].to_numpy(float) / delta_sd).astype(np.float32)
        inputs.append(np.repeat(value[None, :], n, axis=0))
    return inputs


def equal_weights(frame):
    conditions = frame.drop_duplicates('condition_id').groupby('target_id').size()
    n_sites = frame.groupby('condition_id').position.transform('size').to_numpy(float)
    return 1 / (frame.target_id.nunique() * frame.target_id.map(conditions).to_numpy(float) * n_sites)


def metric_tables(frame):
    scored = frame.assign(squared_error_pp2=(frame.pred_position_pp - frame.true_position_pp) ** 2,
                          absolute_error_pp=(frame.pred_position_pp - frame.true_position_pp).abs())
    conditions = scored.groupby(['model', 'fold', *KEY, 'condition_id'], as_index=False).agg(
        mse_pp2=('squared_error_pp2', 'mean'), mae_pp=('absolute_error_pp', 'mean'),
        n_A_positions=('position', 'size'))
    targets = conditions.groupby(['model', 'fold', 'target_id'], as_index=False).agg(
        mse_pp2=('mse_pp2', 'mean'), mae_pp=('mae_pp', 'mean'), n_conditions=('condition_id', 'size'))
    return conditions, targets

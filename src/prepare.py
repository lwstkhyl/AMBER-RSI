#!/usr/bin/env python3
"""Join this run's labels and clean nearest-C1′ distances once."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import DIST, KEY, save_table

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
a = p.parse_args()
distance = pd.read_csv(a.work / 'distances/shared_distance_clean.tsv', sep='\t')
matrix = distance[distance.position.between(3, 10)].pivot(index=KEY, columns='position', values='nearest_c1')
matrix = matrix.reindex(columns=range(3, 11))
matrix.columns = DIST
for editor in ['ABE8e', 'D699', 'TeABE']:
    labels = pd.read_csv(a.work / 'data' / f'{editor}_outcomes.tsv', sep='\t', dtype={'outcome_key': str})
    result = labels.merge(matrix.reset_index(), on=KEY, validate='many_to_one', sort=False)
    if len(result) != 984 or result.condition_id.nunique() != 36 or not np.isfinite(result[DIST]).all().all():
        raise ValueError(f'{editor}: incomplete labels or geometry')
    save_table(a.work / 'inputs' / f'{editor}.tsv', result)
positions = pd.read_csv(a.work / 'data/TeABE_positions.tsv', sep='\t')
positions = positions.merge(distance[[*KEY, 'position', 'nearest_c1']], on=[*KEY, 'position'], validate='one_to_one')
if len(positions) != 312 or not np.isfinite(positions[['nearest_c1', 'true_position_pp']]).all().all():
    raise ValueError('Expected 312 full20 TeABE A-position records')
save_table(a.work / 'data/TeABE_positions.tsv', positions.sort_values(KEY + ['position']))
print('Prepared 984 outcomes per editor and 312 TeABE positions; main models use clean nearest-C1′.')

#!/usr/bin/env python3
"""Pool the same five structures, retaining the established <2Å clash QC rule."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from extract_distances import FEATURES

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--parts', type=Path, required=True)
p.add_argument('--outdir', type=Path, required=True)
a = p.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(a.manifest, sep='\t')
raw = pd.concat([pd.read_csv(a.parts / f'{r.af3_id}_distances.tsv', sep='\t') for r in manifest.itertuples()], ignore_index=True)
if len(raw) != 3600 or raw.duplicated(['target_id', 'length_group_id', 'model', 'position']).any():
    raise ValueError('Require 36 jobs × 5 models × 20 positions')
if not np.isfinite(raw[FEATURES]).all().all() or (raw[FEATURES] < 0).any().any():
    raise ValueError('Invalid coordinate-derived distance')
model_keys = ['target_id', 'length_group_id', 'model']
flagged = raw.loc[raw.nucleotide_nearest_carbon < 2, model_keys].drop_duplicates()
flagged.to_csv(a.outdir / 'flagged_structures.tsv', sep='\t', index=False)
raw.to_csv(a.outdir / 'distance_per_model.tsv', sep='\t', index=False)
bad = set(flagged.itertuples(index=False, name=None))
keep = np.array([tuple(x) not in bad for x in raw[model_keys].itertuples(index=False, name=None)])
keys = ['target_id', 'length_group_id', 'position']
for name, frame in [('all', raw), ('clean', raw[keep])]:
    grouped = frame.groupby(keys, sort=True)[FEATURES]
    result = grouped.mean().join(grouped.std(ddof=1).add_suffix('_sd')).join(grouped.size().rename('n_models')).reset_index()
    result['af3_condition_id'] = result.target_id + '_' + result.length_group_id
    result.to_csv(a.outdir / f'shared_distance_{name}.tsv', sep='\t', index=False)
    if len(result) != 720 or result.n_models.min() < 4:
        raise ValueError('Incomplete distance coverage')
    print(f'{name}: {len(result)} rows; n_models={sorted(result.n_models.unique())}')

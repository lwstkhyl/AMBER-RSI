#!/usr/bin/env python3
"""Enumerate the canonical outcomes from raw sequences and freshly recomputed original scores."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--baseline-out', type=Path, required=True)
p.add_argument('--outdir', type=Path, required=True)
a = p.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
conditions = pd.read_csv(a.manifest, sep='\t').sort_values(['target_id', 'group_order'])
energy = pd.read_csv(a.baseline_out / 'CRISPRparams.tsv', sep='\t')
cas9 = pd.read_csv(a.baseline_out / 'crispron.csv')
if energy.groupby('guideSeq').CRISPRoff_score.nunique().max() != 1 or cas9.groupby('30mer').CRISPRon.nunique().max() != 1:
    raise ValueError('Conflicting original-model sequence scores')
energy = energy.drop_duplicates('guideSeq').set_index('guideSeq').CRISPRoff_score
cas9 = cas9.drop_duplicates('30mer').set_index('30mer').CRISPRon
conditions['CRISPRoff_score'] = conditions.target_30mer.str.slice(4, 27).map(energy)
conditions['CRISPRon_score'] = conditions.target_30mer.map(cas9)
if not np.isfinite(conditions[['CRISPRoff_score', 'CRISPRon_score']]).all().all():
    raise ValueError('Original sequence-score output is incomplete')
conditions.to_csv(a.outdir / 'condition_features.tsv', sep='\t', index=False)
rows = []
for r in conditions.itertuples(index=False):
    editable = [i for i in range(3, 11) if r.target_20mer[i - 1] == 'A']
    index = 0
    for n in range(1, len(editable) + 1):
        for edited in combinations(editable, n):
            bits = [int(i in edited) for i in range(3, 11)]
            outcome = list(r.target_30mer)
            for i in edited:
                outcome[4 + i - 1] = 'G'
            row = dict(target_id=r.target_id, length_group_id=r.length_group_id,
                target_20mer=r.target_20mer, target_30mer=r.target_30mer,
                CRISPRoff_score=r.CRISPRoff_score, CRISPRon_score=r.CRISPRon_score,
                sgRNA_length_numeric=r.sgRNA_length_numeric, outcome_key=''.join(map(str, bits)),
                outcome_sequence=''.join(outcome), outcome_index=index, n_editable_A_pos3_10=len(editable))
            row.update({f'outcome_pos{i}': bit for i, bit in zip(range(3, 11), bits)})
            rows.append(row)
            index += 1
table = pd.DataFrame(rows)
if len(table) != 984 or table.groupby(['target_id', 'length_group_id']).size().size != 36:
    raise ValueError('Outcome enumeration differs from the established 984-row experiment')
table.to_csv(a.outdir / 'outcome_template.tsv', sep='\t', index=False)
print(f'Enumerated {len(table)} nonempty outcomes; observed frequencies have not been used')

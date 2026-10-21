#!/usr/bin/env python3
"""Derive supplementary geometry and full20 observed profiles from this run's inputs."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def sample_positions(alleles, target):
    references = alleles.Reference_Sequence.str.upper().str.replace('-', '', regex=False)
    counts = alleles.groupby(references)['#Reads'].sum()
    candidates = [s for s in counts.index if target in s]
    reference = max(candidates, key=lambda s: (len(s), counts[s]))
    if reference.count(target) != 1:
        raise ValueError('Ambiguous full20 reference')
    start = reference.index(target)
    numerator = np.zeros(20)
    covered = 0.0
    coordinates = {}
    for row in alleles.to_dict('records'):
        ref, aligned = row['Reference_Sequence'].upper(), row['Aligned_Sequence'].upper()
        if ref not in coordinates:
            offset = reference.find(ref.replace('-', ''))
            if offset < 0:
                raise ValueError('Incompatible full20 reference')
            coordinates[ref] = {offset + j: col for j, col in enumerate(i for i, base in enumerate(ref) if base != '-')}
        mapping = coordinates[ref]
        if not all(start + i in mapping for i in range(20)):
            continue
        reads = float(row['#Reads'])
        covered += reads
        for i, base in enumerate(target):
            if base == 'A' and aligned[mapping[start + i]] == 'G':
                numerator[i] += reads
    if covered <= 0:
        raise ValueError('No reads cover the complete 20nt window')
    return 100 * numerator / covered, 100 * covered / float(alleles['#Reads'].sum())


def prepare(work, raw_zip):
    output = work / 'plot_data'
    output.mkdir(parents=True, exist_ok=True)
    geometry = pd.read_csv(work / 'distances/shared_distance_clean.tsv', sep='\t')
    reference = geometry[geometry.length_group_id.eq('L20')][['target_id', 'position', 'nearest_c1']]
    reference = reference.rename(columns={'nearest_c1': 'nearest_c1_L20'})
    geometry = geometry.merge(reference, on=['target_id', 'position'], validate='many_to_one')
    geometry['delta_nearest_c1'] = geometry.nearest_c1 - geometry.nearest_c1_L20
    geometry.to_csv(output / 'distance_per_target.tsv', sep='\t', index=False)
    manifest = pd.read_csv(work / 'raw/sample_manifest.tsv', sep='\t')
    rows = []
    with zipfile.ZipFile(raw_zip) as archive:
        for row in manifest[manifest.include.eq(1)].itertuples(index=False):
            alleles = pd.read_csv(io.BytesIO(archive.read(row.member)), sep='\t')
            values, coverage = sample_positions(alleles, row.target_20mer)
            for position, base in enumerate(row.target_20mer, 1):
                if base == 'A':
                    rows.append(dict(editor=row.editor, target_id=row.target_id,
                        length_group_id=row.length_group_id, sample_id=row.sample_id,
                        position=position, true_position_pp=float(values[position - 1]), full20_covered_percent=coverage))
    samples = pd.DataFrame(rows)
    samples.to_csv(output / 'full20_sample_positions.tsv', sep='\t', index=False)
    keys = ['editor', 'target_id', 'length_group_id', 'position']
    positions = samples.groupby(keys, as_index=False).true_position_pp.mean()
    positions.to_csv(output / 'full20_positions.tsv', sep='\t', index=False)
    teabe = pd.read_csv(work / 'data/TeABE_positions.tsv', sep='\t')
    expected = teabe.set_index(['target_id', 'length_group_id', 'position']).true_position_pp.sort_index()
    actual = positions[positions.editor.eq('TeABE')].set_index(['target_id', 'length_group_id', 'position']).true_position_pp.sort_index()
    if not actual.index.equals(expected.index) or not np.allclose(actual, expected, atol=1e-7, rtol=0):
        raise ValueError('Recomputed full20 TeABE marginals differ from this run\'s transfer labels')
    summary = positions.groupby(['editor', 'length_group_id', 'position']).true_position_pp.agg(['mean', 'std', 'size'])
    grid = pd.MultiIndex.from_product([['ABE8e', 'TeABE', 'D699'], ['L16_17', 'L18_19', 'L20'], range(1, 21)],
        names=['editor', 'length_group_id', 'position'])
    summary = summary.reindex(grid).fillna(0).rename(columns={'mean': 'observed_mean_pp', 'std': 'observed_sd_pp', 'size': 'n_targets_with_A'})
    summary['observed_sem_pp'] = summary.observed_sd_pp / np.sqrt(summary.n_targets_with_A.clip(lower=1))
    summary.reset_index().to_csv(output / 'observed_window_summary.tsv', sep='\t', index=False)
    record = dict(label_definition='A-to-G reads / full20-covered reads per sample; equal sample means, then mean across targets with A',
        geometry='same-run clean nearest-C1 distances; current length minus same-target L20',
        n_geometry_rows=len(geometry), n_position_rows=len(positions), n_summary_rows=len(summary),
        maximum_TeABE_label_difference_pp=float(np.max(np.abs(actual - expected))))
    (output / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--raw-zip', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.work, args.raw_zip)

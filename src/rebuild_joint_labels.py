#!/usr/bin/env python3
"""Rebuild efficiency and window A-to-G frequencies from the same sample set."""
import argparse
import hashlib
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--allele-zip', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--template', type=Path, required=True)
p.add_argument('--outdir', type=Path, required=True)
a = p.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(a.manifest, sep='\t')
template = pd.read_csv(a.template, sep='\t', dtype={'outcome_key': str})
bits = [f'outcome_pos{i}' for i in range(3, 11)]
template['outcome_key'] = template[bits].astype(int).astype(str).agg(''.join, axis=1)
static_columns = ['target_id', 'length_group_id', 'target_20mer', 'target_30mer',
                  'CRISPRoff_score', 'CRISPRon_score', 'sgRNA_length_numeric',
                  'outcome_key', 'outcome_sequence', 'outcome_index', 'n_editable_A_pos3_10'] + bits
static = template[static_columns].copy()
sample_rows, outcome_rows, position_rows = [], [], []
with zipfile.ZipFile(a.allele_zip) as archive:
    for r in manifest[manifest.include.eq(1)].itertuples(index=False):
        raw = archive.read(r.member)
        alleles = pd.read_csv(io.BytesIO(raw), sep='\t')
        total_reads = float(alleles['#Reads'].sum())
        if total_reads <= 0:
            raise ValueError(f'{r.sample_id}: no reads')
        weights = 100 * alleles['#Reads'] / total_reads
        reported_edited = alleles[['n_deleted', 'n_inserted', 'n_mutated']].sum(axis=1).gt(0)
        efficiency = float(weights[reported_edited].sum())
        references = alleles.Reference_Sequence.str.upper().str.replace('-', '', regex=False)
        ref_counts = alleles.groupby(references)['#Reads'].sum()
        candidates = [s for s in ref_counts.index if r.target_20mer in s]
        reference = max(candidates, key=lambda s: (len(s), ref_counts[s]))
        if reference.count(r.target_20mer) != 1:
            raise ValueError(f'{r.sample_id}: ambiguous protospacer')
        start = reference.index(r.target_20mer)
        frequencies, coordinates = {}, {}
        position_frequencies = np.zeros(20)
        full20_covered_percent = 0.0
        partial = 0.0
        for allele, weight in zip(alleles.to_dict('records'), weights):
            ref = allele['Reference_Sequence'].upper()
            aligned = allele['Aligned_Sequence'].upper()
            if len(ref) != len(aligned):
                raise ValueError(f'{r.sample_id}: alignment length mismatch')
            if ref not in coordinates:
                offset = reference.find(ref.replace('-', ''))
                if offset < 0:
                    raise ValueError(f'{r.sample_id}: incompatible reference')
                coordinates[ref] = {offset + j: col for j, col in enumerate(
                    i for i, b in enumerate(ref) if b != '-')}
            coordinate = coordinates[ref]
            if not all(start + i in coordinate for i in range(2, 10)):
                partial += weight
            # Match the existing labels: mask deletions/uncovered positions as WT.
            key = ''.join('1' if reference[start + i] == 'A' and start + i in coordinate
                          and aligned[coordinate[start + i]] == 'G' else '0' for i in range(2, 10))
            frequencies[key] = frequencies.get(key, 0.0) + float(weight)
            if r.editor == 'TeABE' and all(start + i in coordinate for i in range(20)):
                full20_covered_percent += float(weight)
                for i, base in enumerate(r.target_20mer):
                    if base == 'A' and start + i in coordinate and aligned[coordinate[start + i]] == 'G':
                        position_frequencies[i] += float(weight)
        window_editing = sum(v for k, v in frequencies.items() if '1' in k)
        sample_rows.append(dict(editor=r.editor, target_id=r.target_id,
                                length_group_id=r.length_group_id, sample_id=r.sample_id,
                                total_reads=total_reads, editing_efficiency=efficiency,
                                window_edited_percent=window_editing, partial_window_percent=partial,
                                source_md5=hashlib.md5(raw).hexdigest()))
        if r.editor == 'TeABE':
            if full20_covered_percent <= 0:
                raise ValueError(f'{r.sample_id}: no reads cover the complete 20nt window')
            # Preserve the established full20 label denominator; main 3-10 labels use all reads.
            position_frequencies *= 100 / full20_covered_percent
            for i, base in enumerate(r.target_20mer):
                if base == 'A':
                    position_rows.append(dict(target_id=r.target_id, length_group_id=r.length_group_id,
                        sample_id=r.sample_id, position=i + 1, true_position_pp=position_frequencies[i],
                        full20_covered_percent=full20_covered_percent))
        keys = static.loc[static.target_id.eq(r.target_id)
                          & static.length_group_id.eq(r.length_group_id), 'outcome_key']
        for key in keys:
            outcome_rows.append(dict(editor=r.editor, target_id=r.target_id,
                                     length_group_id=r.length_group_id, sample_id=r.sample_id,
                                     outcome_key=key, outcome_frequency=frequencies.get(key, 0.0)))
qc = pd.DataFrame(sample_rows)
replicates = pd.DataFrame(outcome_rows)
qc.to_csv(a.outdir / 'joint_sample_qc.tsv', sep='\t', index=False)
replicates.to_csv(a.outdir / 'joint_sample_outcomes.tsv', sep='\t', index=False)
keys = ['editor', 'target_id', 'length_group_id']
efficiency = qc.groupby(keys).agg(editing_efficiency=('editing_efficiency', 'mean'),
    editing_efficiency_sd=('editing_efficiency', 'std'), n_samples=('sample_id', 'size'),
    sample_ids=('sample_id', ','.join), window_edited_percent=('window_edited_percent', 'mean')).reset_index()
efficiency.to_csv(a.outdir / 'joint_condition_labels.tsv', sep='\t', index=False)
frequency = replicates.groupby(keys + ['outcome_key']).outcome_frequency.agg(['mean', 'std']).reset_index()
for editor in ['ABE8e', 'TeABE', 'D699']:
    f = frequency[frequency.editor.eq(editor)].drop(columns='editor')
    e = efficiency[efficiency.editor.eq(editor)].drop(columns='editor')
    result = static.merge(f, on=['target_id', 'length_group_id', 'outcome_key'], validate='one_to_one')
    result = result.merge(e, on=['target_id', 'length_group_id'], validate='many_to_one')
    result = result.rename(columns={'mean': 'outcome_frequency', 'std': 'outcome_frequency_sd'})
    if len(result) != len(static):
        raise ValueError(f'{editor}: incomplete conditions/outcomes')
    result['editor'] = editor
    result['condition_id'] = editor + '|' + result.target_id + '|' + result.length_group_id
    result['outcome_id'] = result.condition_id + '|' + result.outcome_key
    result['efficiency_definition'] = 'reported_any_mutation_or_indel_reads_over_total_reads'
    result['label_policy'] = 'equal_mean_of_the_same_included_samples'
    result.to_csv(a.outdir / f'{editor}_outcomes.tsv', sep='\t', index=False)
    print(f'{editor}: {result.condition_id.nunique()} conditions, {len(result)} outcomes', flush=True)
print('Efficiency includes reported substitutions and indels; window frequency includes A-to-G at positions 3-10.')
# The final transfer predicts marginals only; full20 outcome enumeration is unnecessary.
pd.DataFrame(position_rows).to_csv(a.outdir / 'TeABE_sample_positions.tsv', sep='\t', index=False)
positions = pd.DataFrame(position_rows).groupby(['target_id', 'length_group_id', 'position'], as_index=False).true_position_pp.mean()
metadata = manifest[['target_id', 'length_group_id', 'target_20mer', 'target_30mer', 'sgRNA_length_numeric']].drop_duplicates()
positions = positions.merge(metadata, on=['target_id', 'length_group_id'], validate='many_to_one')
positions['condition_id'] = 'TeABE|' + positions.target_id + '|' + positions.length_group_id
positions.to_csv(a.outdir / 'TeABE_positions.tsv', sep='\t', index=False)

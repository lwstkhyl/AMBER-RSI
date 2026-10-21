#!/usr/bin/env python3
"""Extract four cross-DNA distances from the original AF3 Server ZIPs, without running AF3."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd

FEATURES = ['paired_c1', 'nearest_c1', 'c1_nearest_carbon', 'nucleotide_nearest_carbon']

def atom_table(text):
    # AF3 exports one atom record per line. Refuse a different CIF layout instead of guessing.
    lines = text.splitlines()
    start = next(i for i, s in enumerate(lines) if s.startswith('_atom_site.'))
    columns = []
    while lines[start].startswith('_atom_site.'):
        columns.append(lines[start].split('.', 1)[1].strip())
        start += 1
    ix = {name: i for i, name in enumerate(columns)}
    chains = {}
    for line in lines[start:]:
        if not line.strip() or line.startswith('#'):
            break
        tokens = line.split()
        if len(tokens) != len(columns):
            raise ValueError('Unexpected AF3 atom-site row layout')
        row = {key: tokens[ix[key]] for key in ['label_asym_id', 'label_seq_id', 'label_comp_id',
            'label_atom_id', 'type_symbol', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'pdbx_PDB_model_num', 'label_alt_id']}
        if row['pdbx_PDB_model_num'] != '1' or row['label_alt_id'] not in ['.', '?', 'A']:
            raise ValueError('Unexpected multiple coordinate models or alternate conformers')
        residue = int(row['label_seq_id'])
        chain = chains.setdefault(row['label_asym_id'], {})
        current = chain.setdefault(residue, {'name': row['label_comp_id'], 'atoms': {}, 'carbons': []})
        atom = row['label_atom_id']
        if atom[0] in ['"', "'"] and atom[-1] == atom[0]:
            atom = atom[1:-1]
        xyz = np.array([float(row[k]) for k in ['Cartn_x', 'Cartn_y', 'Cartn_z']])
        if atom in current['atoms']:
            raise ValueError('Duplicate atom identity in AF3 coordinates')
        current['atoms'][atom] = xyz
        if row['type_symbol'] == 'C':
            current['carbons'].append((atom, xyz))
    return chains

def extract(archive_path, request_name, row, protein):
    with zipfile.ZipFile(archive_path) as archive:
        request = json.loads(archive.read(request_name))
        job = request[0] if isinstance(request, list) else request
        sequences = job['sequences']
        dnas = [s['dnaSequence']['sequence'].upper() for s in sequences if 'dnaSequence' in s]
        rna = next(s['rnaSequence']['sequence'].upper() for s in sequences if 'rnaSequence' in s)
        observed_protein = next(s['proteinChain']['sequence'] for s in sequences if 'proteinChain' in s)
        if sorted(dnas) != sorted([row.DNA1_af3, row.DNA2_af3]) or rna != row.full_sgRNA_rna_af3 or observed_protein != protein:
            raise ValueError(f'{row.af3_id}: AF3 input sequences differ from the reconstructed original job')
        parent = request_name.rsplit('/', 1)[0] + '/' if '/' in request_name else ''
        models = sorted(n for n in archive.namelist() if n.startswith(parent) and re.search(r'_model_[0-4]\.cif$', n))
        if len(models) != 5 or {int(re.search(r'_model_(\d)\.cif$', n)[1]) for n in models} != set(range(5)):
            raise ValueError(f'{row.af3_id}: require exactly the original five structures')
        distances, audits = [], []
        for name in models:
            cif_bytes = archive.read(name)
            chains = atom_table(cif_bytes.decode())
            dna_map = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}
            dna_chains = {}
            for chain, residues in chains.items():
                ordered = sorted(residues)
                if all(residues[i]['name'] in dna_map for i in ordered):
                    if ordered != list(range(1, len(ordered) + 1)):
                        raise ValueError('Noncontiguous DNA residue numbering')
                    dna_chains[chain] = ''.join(dna_map[residues[i]['name']] for i in ordered)
            own = [c for c, sequence in dna_chains.items() if sequence == row.DNA1_af3]
            other = [c for c, sequence in dna_chains.items() if sequence == row.DNA2_af3]
            if len(own) != 1 or len(other) != 1 or own == other:
                raise ValueError(f'{row.af3_id}: DNA chains cannot be assigned uniquely by sequence')
            own, other = own[0], other[0]
            opposite = chains[other]
            opposite_c1 = np.array([opposite[i]['atoms']["C1'"] for i in range(1, 61)])
            carbon_rows = [(i, atom, xyz) for i in range(1, 61) for atom, xyz in opposite[i]['carbons']]
            opposite_carbon = np.array([c[2] for c in carbon_rows])
            start = row.DNA1_af3.index(row.target_20mer)
            raw_spacer = row.raw_spacer_dna
            matched = max(k for k in range(1, 21) if raw_spacer.endswith(row.target_20mer[-k:]))
            for position in range(1, 21):
                residue = start + position
                paired = 61 - residue
                current = chains[own][residue]
                c1 = current['atoms']["C1'"]
                nearest_c1 = np.linalg.norm(opposite_c1 - c1, axis=1)
                c1_carbon = np.linalg.norm(opposite_carbon - c1, axis=1)
                own_carbon = np.array([c[1] for c in current['carbons']])
                carbon_distance = np.linalg.norm(own_carbon[:, None, :] - opposite_carbon[None, :, :], axis=2)
                c1_index = int(nearest_c1.argmin())
                carbon_index = int(c1_carbon.argmin())
                i, j = np.unravel_index(carbon_distance.argmin(), carbon_distance.shape)
                distances.append(dict(condition_id=row.af3_id, job_name=row.af3_id,
                    target_id=row.target_id, length_group_id=row.length_group_id,
                    source=str(archive_path), model=Path(name).name, position=position,
                    base=row.target_20mer[position - 1], dna_chain=own, dna_residue=residue,
                    other_dna_chain=other, paired_residue=paired, in_matched_spacer=int(position > 20 - matched),
                    nearest_c1_residue=c1_index + 1, nearest_carbon_residue=carbon_rows[carbon_index][0],
                    nearest_carbon_atom=carbon_rows[carbon_index][1], own_min_carbon_atom=current['carbons'][i][0],
                    min_carbon_other_residue=carbon_rows[j][0], min_carbon_other_atom=carbon_rows[j][1],
                    paired_c1=float(np.linalg.norm(c1 - opposite[paired]['atoms']["C1'"])),
                    nearest_c1=float(nearest_c1[c1_index]), c1_nearest_carbon=float(c1_carbon[carbon_index]),
                    nucleotide_nearest_carbon=float(carbon_distance[i, j])))
            audits.append(dict(af3_id=row.af3_id, target_id=row.target_id, length_group_id=row.length_group_id,
                model=Path(name).name, source=str(archive_path), dna_chain=own, other_dna_chain=other,
                protospacer20=row.target_20mer, protospacer_start_residue=start + 1,
                actual_AF3_matched_spacer_length=matched, modeling_length=row.target_matching_length,
                cif_sha256=hashlib.sha256(cif_bytes).hexdigest()))
    return pd.DataFrame(distances), pd.DataFrame(audits)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--af3-dir', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--index', type=int, help='Zero-based manifest row; omit to extract all 36 jobs')
    a = p.parse_args()
    manifest = pd.read_csv(a.manifest, sep='\t', keep_default_na=False).sort_values(['target_id', 'group_order']).reset_index(drop=True)
    if a.index is not None:
        manifest = manifest.iloc[[a.index]]
    norm = lambda x: re.sub(r'[^a-z0-9]', '', re.sub(r'^fold[_-]?', '', x.lower()))
    candidates = {}
    for path in sorted(a.af3_dir.rglob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            requests = [n for n in archive.namelist() if n.endswith('_job_request.json')]
            for name in requests:
                data = json.loads(archive.read(name))
                job = data[0] if isinstance(data, list) else data
                candidates.setdefault(norm(job['name']), []).append((path, name))
    protein = ''.join(s.strip() for s in (Path(__file__).parents[1] / '01_af3/cas9.fasta').read_text().splitlines() if not s.startswith('>'))
    a.outdir.mkdir(parents=True, exist_ok=True)
    for row in manifest.itertuples(index=False):
        choices = candidates.get(norm(row.af3_id), [])
        if len(choices) != 1:
            raise ValueError(f'{row.af3_id}: found {len(choices)} matching AF3 ZIPs; keep exactly one primary run')
        values, audit = extract(*choices[0], row, protein)
        values.to_csv(a.outdir / f'{row.af3_id}_distances.tsv', sep='\t', index=False)
        audit.to_csv(a.outdir / f'{row.af3_id}_audit.tsv', sep='\t', index=False)
        print(f'{row.af3_id}: 5 models × 20 positions × 4 distances')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Original AlphaFold Server job format, with the raw-archive manifest as input."""
import argparse
import json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--outdir', type=Path, required=True)
p.add_argument('--batch-size', type=int, default=6)
a = p.parse_args()
protein = ''.join(s.strip() for s in (Path(__file__).parent / 'cas9.fasta').read_text().splitlines() if not s.startswith('>'))
manifest = pd.read_csv(a.manifest, sep='\t').sort_values(['target_id', 'group_order'])
individual = a.outdir / 'individual_server_json'
batches = a.outdir / 'server_batches'
individual.mkdir(parents=True, exist_ok=True)
batches.mkdir(parents=True, exist_ok=True)
jobs = []
for r in manifest.itertuples(index=False):
    job = dict(name=r.af3_id, modelSeeds=[], sequences=[
        {'proteinChain': {'sequence': protein, 'count': 1}},
        {'rnaSequence': {'sequence': r.full_sgRNA_rna_af3, 'count': 1}},
        {'dnaSequence': {'sequence': r.DNA1_af3, 'count': 1}},
        {'dnaSequence': {'sequence': r.DNA2_af3, 'count': 1}}], dialect='alphafoldserver', version=1)
    (individual / f'{r.af3_id}.json').write_text(json.dumps([job], indent=2) + '\n')
    jobs.append(job)
for i in range(0, len(jobs), a.batch_size):
    (batches / f'af3_batch_{i // a.batch_size + 1:02d}.json').write_text(json.dumps(jobs[i:i + a.batch_size], indent=2) + '\n')
print(f'{len(jobs)} jobs, {(len(jobs) + a.batch_size - 1) // a.batch_size} batches: {batches}')

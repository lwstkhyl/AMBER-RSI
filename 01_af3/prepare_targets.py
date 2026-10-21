#!/usr/bin/env python3
"""Read all 12 targets directly from truncated sgRNA.zip; keep the original AF3 context."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--allele-zip', type=Path, required=True)
p.add_argument('--outdir', type=Path, required=True)
a = p.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
scaffold = ''.join(x.strip() for x in (here / 'scaffold.fasta').read_text().splitlines() if not x.startswith('>'))
editors = {'N-8E': 'ABE8e', 'TeABE(49-64)(400-1-1249-1264-MU148）': 'TeABE', 'D699-CBH-N-8E-D53R-2AG': 'D699'}
groups = {20: 'L20', 19: 'L18_19', 18: 'L18_19', 17: 'L16_17', 16: 'L16_17'}
group_order = {'L20': 0, 'L18_19': 1, 'L16_17': 2}
complement = str.maketrans('ACGT', 'TGCA')
with zipfile.ZipFile(a.allele_zip) as archive:
    # Older Python versions may display GBK ZIP filenames as CP437 even when a Unicode extra field exists.
    # Match decoded names, but read the original member name returned by this Python's ZipFile.
    books = []
    for name in archive.namelist():
        basename = name.replace('\\', '/').rsplit('/', 1)[-1]
        variants = [basename]
        try:
            raw_name = basename.encode('cp437')
        except UnicodeEncodeError:
            raw_name = None
        if raw_name is not None:
            for encoding in ['gb18030', 'utf-8']:
                try:
                    variants.append(raw_name.decode(encoding))
                except UnicodeDecodeError:
                    pass
        if any(value.casefold() == 'sgRNA编号.xlsx'.casefold() for value in variants):
            books.append(name)
    if not books:
        xlsx_names = [n for n in archive.namelist() if n.lower().endswith('.xlsx')]
        raise FileNotFoundError(f'{a.allele_zip}: sgRNA编号.xlsx文件找不到；ZIP内实际xlsx文件：{xlsx_names}')
    if len(books) != 1:
        raise ValueError(f'{a.allele_zip}: 找到多个sgRNA编号.xlsx，请保留唯一的样本表：{books}')
    book = books[0]
    print(f'Sample workbook ZIP member: {book}', flush=True)
    samples = pd.read_excel(io.BytesIO(archive.read(book)), header=1).iloc[:, :5]
    samples.columns = ['sample_id', 'editor_raw', 'guide_name', 'guide_sequence', 'length_label']
    samples = samples[samples.editor_raw.isin(editors)].copy()
    samples['editor'] = samples.editor_raw.map(editors)
    samples['nominal_length'] = samples.guide_name.str.extract(r'-(\d+)nt').astype(int)
    samples['length_group_id'] = samples.nominal_length.map(groups)
    samples = samples[samples.length_group_id.notna()].copy()
    members = {}
    for n in archive.namelist():
        if '/Alleles_frequency_table_around_sgRNA_' in n and n.endswith('.txt'):
            sample_id = n.split('/')[-2]
            if sample_id in members:
                raise ValueError(f'Multiple allele tables: {sample_id}')
            members[sample_id] = n
    samples['member'] = samples.sample_id.map(members)
    if samples.member.isna().any() or samples.sample_id.duplicated().any():
        raise ValueError('Missing allele tables or duplicate sample IDs')
    samples['target_id'] = samples.member.str.split('/').str[-3].replace({'ANGPTL3-A6': 'sgANGPTL3-A6'})
    samples['guide_target'] = samples.guide_name.str.extract(r'-(?:Sg|sg)?(ABE25|ABE8|ABS4|CS1P|HPE6|PT13|PT14|S12|S16|S18|S7|ANGPTL3-A6)-')
    samples['guide_target'] = samples.guide_target.replace({'ANGPTL3-A6': 'sgANGPTL3-A6'})
    if not samples.target_id.equals(samples.guide_target) or samples.target_id.nunique() != 12:
        raise ValueError('Require the 12 archive targets with matching guide annotations')
    targets, conditions = [], []
    for target, block in samples.groupby('target_id', sort=True):
        guides20 = block.loc[block.length_group_id.eq('L20'), 'guide_sequence'].str.upper().unique()
        if len(guides20) != 1:
            raise ValueError(f'{target}: conflicting 20nt guide sequences')
        target20 = guides20[0][-20:]
        sample = block[block.length_group_id.eq('L20')].iloc[0]
        nested_name = sample.member.rsplit('/', 1)[0] + '/Alleles_frequency_table.zip'
        with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
            table_name = next(n for n in nested.namelist() if n.endswith('.txt'))
            alleles = pd.read_csv(io.BytesIO(nested.read(table_name)), sep='\t')
        sequences = alleles.Reference_Sequence.str.upper().str.replace('-', '', regex=False)
        read_counts = alleles.groupby(sequences)['#Reads'].sum()
        candidates = [s for s in read_counts.index if target20 in s]
        full_reference = max(candidates, key=lambda s: (len(s), read_counts[s]))
        if full_reference.count(target20) != 1:
            raise ValueError(f'{target}: ambiguous full-length reference')
        start = full_reference.index(target20)
        dna1 = full_reference[start - 20:start + 40]
        if start < 20 or len(dna1) != 60 or dna1[20:40] != target20:
            raise ValueError(f'{target}: cannot reconstruct original 60nt AF3 DNA')
        dna2 = dna1.translate(complement)[::-1]
        target30 = full_reference[start - 4:start + 26]
        if len(target30) != 30 or target30[25:27] != 'GG':
            raise ValueError(f'{target}: invalid 30mer or NGG PAM')
        targets.append(dict(target_id=target, target_20mer=target20, target_30mer=target30,
                            PAM=target30[24:27], DNA1_af3=dna1, DNA2_af3=dna2,
                            full_reference_member=nested_name + '::' + table_name))
        if set(block.length_group_id) != set(group_order):
            raise ValueError(f'{target}: missing length group')
        for length_group, g in block.groupby('length_group_id', sort=False):
            if set(g.editor) != set(editors.values()) or g.guide_sequence.str.upper().nunique() != 1:
                raise ValueError(f'{target}/{length_group}: guides differ between editors')
            raw = str(g.guide_sequence.iloc[0]).strip()
            core = re.sub(r'^[acgt]+', '', raw).upper()
            matching = max(k for k in range(1, min(20, len(core)) + 1) if core.endswith(target20[-k:]))
            if matching != {'L20': 20, 'L18_19': 18, 'L16_17': 16}[length_group]:
                raise ValueError(f'{target}/{length_group}: unexpected target-complementary length {matching}')
            extra = raw[:-matching]
            conditions.append(dict(af3_id=target + '_' + length_group, target_id=target,
                length_group_id=length_group, group_order=group_order[length_group],
                raw_spacer_dna=raw.upper(), extra_5p_dna=extra.upper(),
                target_matching_spacer_dna=target20[-matching:], target_matching_length=matching,
                sgRNA_length_numeric=matching, full_sgRNA_rna_af3=raw.upper().replace('T', 'U') + scaffold,
                target_20mer=target20, target_30mer=target30, PAM=target30[24:27],
                DNA1_af3=dna1, DNA2_af3=dna2))
target_table = pd.DataFrame(targets)
manifest = pd.DataFrame(conditions).sort_values(['target_id', 'group_order']).reset_index(drop=True)
samples = samples.merge(manifest[['target_id', 'length_group_id', 'target_20mer', 'target_30mer', 'sgRNA_length_numeric']],
                        on=['target_id', 'length_group_id'], validate='many_to_one')
# Keep the already verified sample policy; these are explicit exclusions, not a label intersection.
excluded = samples.editor.eq('ABE8e') & samples.sample_id.isin(['6W63', '6U103'])
samples['include'] = (~excluded).astype(int)
samples['include_reason'] = 'included_in_existing_label_sample_set'
samples.loc[excluded, 'include_reason'] = 'excluded_to_preserve_verified_ABE8e_sample_set'
if len(manifest) != 36 or samples.groupby('editor').include.sum().to_dict() != {'ABE8e': 108, 'D699': 111, 'TeABE': 112}:
    raise ValueError('Source sample/condition counts differ from the established experiment')
target_table.to_csv(a.outdir / 'targets.tsv', sep='\t', index=False)
manifest.to_csv(a.outdir / 'af3_manifest.tsv', sep='\t', index=False)
samples.to_csv(a.outdir / 'sample_manifest.tsv', sep='\t', index=False)
with (a.outdir / 'targets_30mer.fa').open('w') as f:
    for r in target_table.itertuples(index=False):
        f.write(f'>{r.target_id}\n{r.target_30mer}\n')
qc = dict(targets=len(target_table), af3_jobs=len(manifest), samples=len(samples),
          included_samples=samples.groupby('editor').include.sum().to_dict(),
          allele_zip_sha256=hashlib.sha256(a.allele_zip.read_bytes()).hexdigest(),
          selection='all 12 targets, 3 length groups; no intersection with summary workbook')
(a.outdir / 'preparation.json').write_text(json.dumps(qc, indent=2) + '\n')
print(json.dumps(qc, indent=2))

#!/usr/bin/env python3
"""Value-level file comparisons shared by the reproduction report."""
import json
import numpy as np
import pandas as pd

IDENTIFIERS = {'outcome_key', 'outcome_id', 'condition_id', 'target_id', 'test_target',
               'validation_target', 'sample_id', 'sample_ids', 'source_md5', 'cif_sha256'}


def array_difference(a, b, atol):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, False, None, f'shape {a.shape} vs {b.shape}'
    if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
        exact = np.array_equal(a, b, equal_nan=True)
        close = np.allclose(a, b, rtol=0, atol=atol, equal_nan=True)
        finite = np.isfinite(a) & np.isfinite(b)
        maximum = float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))) if finite.any() else 0.0
        return exact, bool(close), maximum, '' if close else 'numeric values differ'
    same = np.array_equal(a, b)
    return same, same, None, '' if same else 'text or array dtype differs'


def file_fields(left, right, atol):
    if left.suffix == '.npz':
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                return [('keys', False, False, None, 'NPZ keys differ')]
            return [(key, *array_difference(a[key], b[key], atol)) for key in a.files]
    if left.suffix in ['.tsv', '.csv']:
        sep = ',' if left.suffix == '.csv' else '\t'
        a, b = [pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False) for p in [left, right]]
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            return [('table', False, False, None, 'table shape or columns differ')]
        fields = []
        for column in a:
            x, y = a[column].to_numpy(), b[column].to_numpy()
            if column not in IDENTIFIERS:
                try:
                    x, y = [pd.to_numeric(pd.Series(v).replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan}),
                                          errors='raise').to_numpy(float) for v in [x, y]]
                except (TypeError, ValueError):
                    pass
            fields.append((column, *array_difference(x, y, atol)))
        return fields
    if left.suffix == '.json':
        a, b = json.loads(left.read_text()), json.loads(right.read_text())
        if left.name in ['environment.json', 'worker_environment.json']:
            a = {k: v for k, v in a.items() if k not in ['host', 'cpu']}
            b = {k: v for k, v in b.items() if k not in ['host', 'cpu']}
        # Model parameters in JSON must also agree exactly, independently of an optional TSV/NPZ tolerance.
        same = a == b
        return [('json', same, same, None, '' if same else 'JSON settings or parameters differ')]
    same = left.read_bytes() == right.read_bytes()
    return [('content', same, same, None, '' if same else 'file content differs')]



#!/usr/bin/env python3
"""Record settings and environment once; require complete results before marking success."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
from common import (MEMBERS, EDITORS, PROTOCOL, SEED, EPOCHS, PATIENCE, BATCH_SIZE, LEARNING_RATE, BOTTLENECK,
                    save_json)

PACKAGE = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    return {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.rglob('*'))
            if p.is_file() and p.suffix in ['.py', '.sh', '.sbatch', '.fasta']}


def initialize(work):
    if (work / 'run_config.json').exists():
        raise ValueError(f'This run has already been initialized: {work}')
    configuration = dict(protocol=PROTOCOL, seed=SEED, epochs=EPOCHS, patience=PATIENCE,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, bottleneck=BOTTLENECK, checkpoint='validation_joint_mse',
        editors=EDITORS, members=MEMBERS, fold_seed=f'{SEED}+100*fold', bottleneck_seed=f'{SEED}+100*fold+1',
        raw_zip=os.environ['RAW_ZIP'], af3_dir=os.environ['AF3_DIR'], upstream_root=os.environ['UPSTREAM_ROOT'],
        pretrained_root=os.environ['PRETRAINED_ROOT'], code=code_hashes(),
        configuration_sha256=sha(work / 'config.snapshot.json'))
    save_json(work / 'run_config.json', configuration)


def runtime():
    import keras
    import numpy as np
    import pandas as pd
    import scipy
    import tensorflow as tf
    if tf.__version__ != '2.10.0' or keras.__version__ != '2.10.0':
        raise ValueError('Use the established TensorFlow/Keras 2.10.0 crispronbe environment')
    cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')), platform.processor())
    variables = ['PYTHONHASHSEED', 'TF_DETERMINISTIC_OPS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'CUDA_VISIBLE_DEVICES']
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, tensorflow=tf.__version__, keras=keras.__version__,
        variables={k: os.environ.get(k) for k in variables}, host=socket.gethostname(), cpu=cpu)


def environment(work):
    record = runtime()
    models = {str(p.relative_to(os.environ['PRETRAINED_ROOT'])): sha(p)
              for member in MEMBERS for p in sorted((Path(os.environ['PRETRAINED_ROOT']) / member).rglob('*'))
              if p.is_file()}
    if not all(any(name.startswith(member + '/') for name in models) for member in MEMBERS):
        raise ValueError('All 25 original backbones are required')
    save_json(work / 'environment.json', dict(**record, raw_zip_sha256=sha(os.environ['RAW_ZIP']), pretrained_files=models))


def finish(work):
    import pandas as pd
    configuration = json.loads((work / 'run_config.json').read_text())
    if configuration['code'] != code_hashes():
        raise ValueError('Code changed during this run; keep the package unchanged until both runs finish')
    if configuration['configuration_sha256'] != sha(work / 'config.snapshot.json'):
        raise ValueError('The configuration snapshot changed during this run')
    main = pd.read_csv(work / 'summary/main_four_models.tsv', sep='\t')
    folds = pd.read_csv(work / 'summary/fold_checkpoints.tsv', sep='\t')
    transfer = pd.read_csv(work / 'transfer/summary/test_position_predictions.tsv', sep='\t')
    if len(main) != 8 or len(folds) != 1800 or len(transfer) != 624:
        raise ValueError('Missing main-model folds or transfer predictions')
    counts = {name: len(list(work.glob(pattern))) for name, pattern in {
        'checkpoints': 'runs/*/*/*/checkpoints/*.npz', 'initial_weights': 'runs/*/*/*/initial_weights/*.npz',
        'branch_weights': 'runs/*/*/*/branch_weights/*.npz',
        'source_predictions': 'runs/*/*/delta_nearest/source_predictions/*.npz'}.items()}
    if counts != dict(checkpoints=1800, initial_weights=1800, branch_weights=1800, source_predictions=600):
        raise ValueError(f'Incomplete scientific artifacts: {counts}')
    folders = ['raw', 'af3_inputs', 'distance_parts', 'distances', 'data', 'inputs', 'runs', 'summary', 'transfer']
    artifacts = sorted(str(p.relative_to(work)) for name in folders for p in (work / name).rglob('*')
                       if p.is_file() and p.suffix in ['.tsv', '.npz', '.json', '.fa'])
    artifacts += ['original_model_output/' + name for name in ['CRISPRparams.tsv', 'crispron.csv', 'crispronABE_prediction.tsv']]
    save_json(work / 'completed.json', dict(protocol=PROTOCOL, complete=True, counts=counts, artifacts=artifacts))
    print('Complete:', work)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['init', 'environment', 'finish'])
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    {'init': initialize, 'environment': environment, 'finish': finish}[a.stage](a.work)

#!/usr/bin/env python3
"""Plan full-data refits and require all 50 members before publishing their manifest."""
import json
import math
from pathlib import Path
import shutil
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'all12_fixed_epochs_v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def addon_hashes():
    from state import sha
    return {p.name: sha(p) for p in sorted(ROOT.iterdir())
            if p.suffix in ['.py', '.sh', '.sbatch']}


def load_plan(run):
    plan = read_json(run / 'plan.json')
    sys.path.insert(0, str(Path(plan['analysis_root']) / 'src'))
    from state import code_hashes, sha
    if plan['protocol'] != PROTOCOL:
        raise ValueError('Not an all-12 deployment run')
    if plan['analysis_code'] != code_hashes() or plan['deployment_code'] != addon_hashes():
        raise ValueError('Code changed after the deployment plan was made')
    if any(sha(run / name) != value for name, value in plan['input_sha256'].items()):
        raise ValueError('Deployment inputs changed after planning')
    return plan


def prepare(analysis, cv, output, seed):
    from common import EDITORS, MEMBERS, EPOCHS, BATCH_SIZE, LEARNING_RATE, read_input, save_json, save_table
    from state import code_hashes, sha
    if output.exists():
        raise ValueError(f'Use a new deployment directory: {output}')
    if analysis == output or analysis in output.parents or cv == output:
        raise ValueError('Keep deployment output outside source code and separate from analysis outputs')
    config = read_json(cv / 'run_config.json')
    environment = read_json(cv / 'environment.json')
    if read_json(cv / 'completed.json').get('complete') is not True:
        raise ValueError('The source LOTO run must be complete')
    if config['code'] != code_hashes():
        raise ValueError('Analysis code does not match the completed LOTO run')
    folds = pd.read_csv(cv / 'summary/fold_checkpoints.tsv', sep='\t')
    tasks, targets = [], None
    for editor in EDITORS:
        frame = read_input(cv, editor)
        current_targets = sorted(frame.target_id.unique())
        if len(current_targets) != 12 or (targets is not None and targets != current_targets):
            raise ValueError('Expected the same 12 targets in both source editors')
        targets = current_targets
        for member in MEMBERS:
            row = dict(task=len(tasks), editor=editor, member=member)
            for candidate, name in [('length', 'length_epochs'), ('delta_nearest', 'delta_epochs')]:
                selected = folds[(folds.editor == editor) & (folds.backbone == member) &
                                 (folds.candidate == candidate)]
                if (len(selected) != 12 or set(selected.fold) != set(range(1, 13)) or
                        not selected.checkpoint_metric.eq('joint_mse').all() or
                        not selected.best_epoch.between(1, EPOCHS).all()):
                    raise ValueError(f'Incomplete CV epoch evidence: {editor}/{member}/{candidate}')
                row[name] = math.ceil(float(selected.best_epoch.median()))
            tasks.append(row)
    (output / 'inputs').mkdir(parents=True)
    for editor in EDITORS:
        shutil.copyfile(cv / 'inputs' / f'{editor}.tsv', output / 'inputs' / f'{editor}.tsv')
    save_table(output / 'epochs.tsv', pd.DataFrame(tasks))
    plan = dict(protocol=PROTOCOL, purpose='deployment refit; not held-out evaluation',
        analysis_root=str(analysis), cv_run=str(cv), pretrained_root=config['pretrained_root'],
        seed=seed, targets=targets, tasks=tasks, members=MEMBERS, editors=EDITORS,
        epoch_rule='ceil(median of the 12 LOTO best_epoch values), separately per editor/member/branch',
        learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, early_stopping=False, delta_centered=False,
        analysis_code=code_hashes(), deployment_code=addon_hashes(),
        source_folds_sha256=sha(cv / 'summary/fold_checkpoints.tsv'),
        source_environment=environment,
        input_sha256={f'inputs/{editor}.tsv': sha(output / 'inputs' / f'{editor}.tsv') for editor in EDITORS})
    save_json(output / 'plan.json', plan)
    (output / 'logs').mkdir()
    print(f'Planned 50 members on all 12 targets: {output}')


def finish(run):
    from common import save_json
    plan = load_plan(run)
    for task in plan['tasks']:
        folder = run / 'members' / task['editor'] / task['member']
        completed = read_json(folder / 'completed.json')
        if completed != dict(**task, seed=plan['seed'], n_targets=12, n_conditions=36, n_outcomes=984):
            raise ValueError(f'Missing or incorrect completed member: {folder}')
        for name in ['weights.npz', 'training_fit_predictions.npz', 'length_initial.npz',
                     'delta_initial.npz', 'length_history.tsv', 'delta_history.tsv', 'environment.json']:
            if not (folder / name).is_file():
                raise ValueError(f'Missing deployment artifact: {folder / name}')
    artifacts = sorted(str(p.relative_to(run)) for top in ['inputs', 'members']
                       for p in (run / top).rglob('*') if p.suffix in ['.npz', '.tsv', '.json'])
    artifacts += ['plan.json', 'epochs.tsv']
    save_json(run / 'completed.json', dict(protocol=PROTOCOL, complete=True, n_members=50,
        n_targets=12, purpose='deployment; training fits are not test scores', artifacts=artifacts))
    print(f'Complete deployment model: {run}')



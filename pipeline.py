#!/usr/bin/env python3
"""One entry point for AF3 inputs, LOTO analysis, PDF figures and deployment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from settings import load_config

DEPS = {
    'init': [], 'af3': ['init'], 'af3-check': ['af3'], 'environment': ['init'],
    'distance': ['af3-check'], 'distance-summary': ['distance'],
    'features': ['af3-check', 'environment'], 'prepare': ['features', 'distance-summary'],
    'length': ['prepare'], 'delta': ['length'], 'summary': ['delta'], 'transfer': ['delta'],
    'transfer-summary': ['transfer'], 'analysis-finish': ['summary', 'transfer-summary'],
    'compare-analysis': ['analysis-finish'], 'plot-data': ['prepare'],
    'plot': ['analysis-finish', 'plot-data'], 'deployment-plan': ['analysis-finish', 'compare-analysis'],
    'deployment': ['deployment-plan'], 'deployment-finish': ['deployment'],
    'deployment-verify': ['deployment-finish'], 'compare-deployment': ['deployment-finish'],
    'finish': ['af3-check', 'compare-analysis', 'plot', 'deployment-verify', 'compare-deployment']}
ARRAYS = {'distance': (0, 35), 'length': (0, 49), 'delta': (0, 99), 'transfer': (1, 12), 'deployment': (0, 49)}
TRAIN = ['init', 'af3', 'af3-check', 'environment', 'distance', 'distance-summary', 'features',
         'prepare', 'length', 'delta', 'summary', 'transfer', 'transfer-summary', 'analysis-finish', 'compare-analysis']
MODEL = ['deployment-plan', 'deployment', 'deployment-finish', 'deployment-verify', 'compare-deployment']
MODES = {'af3': TRAIN[:3], 'distance': ['init', 'af3', 'af3-check', 'distance', 'distance-summary'],
    'train': TRAIN, 'plot': ['plot-data', 'plot'], 'model': MODEL, 'verify': ['deployment-verify'],
    'compare': ['af3-check', 'compare-analysis', 'compare-deployment'],
    'all': TRAIN + ['plot-data', 'plot'] + MODEL + ['finish']}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def code_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ['.py', '.sh', '.sbatch', '.fasta']}


def marker(work, phase, index=None):
    suffix = f'.{index:02d}' if index is not None else ''
    return work / 'control' / f'{phase}{suffix}.done.json'


def finished(work, phase):
    if phase in ARRAYS:
        first, last = ARRAYS[phase]
        return all(marker(work, phase, i).exists() for i in range(first, last + 1))
    return marker(work, phase).exists()


def script(path, *arguments):
    previous = sys.argv
    sys.argv = [str(ROOT / path), *map(str, arguments)]
    try:
        runpy.run_path(str(ROOT / path), run_name='__main__')
    finally:
        sys.argv = previous


def perform(phase, work, config):
    paths = config['paths']
    deployment = work / 'deployment' / 'run01'
    if phase == 'init':
        from state import initialize
        if not (work / 'run_config.json').exists():
            initialize(work)
    elif phase == 'environment':
        from state import environment
        environment(work)
    elif phase == 'af3':
        script('01_af3/prepare_targets.py', '--allele-zip', paths['raw_zip'], '--outdir', work / 'raw')
        script('01_af3/write_af3_server_jobs.py', '--manifest', work / 'raw/af3_manifest.tsv',
               '--outdir', work / 'af3_inputs', '--batch-size', config['af3']['batch_size'])
    elif phase == 'af3-check':
        from reproduce import compare_af3
        compare_af3(work, config)
    elif phase == 'distance':
        script('src/extract_distances.py', '--manifest', work / 'raw/af3_manifest.tsv',
               '--af3-dir', paths['af3_results'], '--outdir', work / 'distance_parts',
               '--index', os.environ['SLURM_ARRAY_TASK_ID'])
    elif phase == 'distance-summary':
        script('src/collect_distances.py', '--manifest', work / 'raw/af3_manifest.tsv',
               '--parts', work / 'distance_parts', '--outdir', work / 'distances')
    elif phase == 'features':
        if not (work / 'original_model_output/crispronABE_prediction.tsv').exists():
            subprocess.run(['bash', 'bin/CRISPRonBE.sh', 'ABE', str(work / 'raw/targets_30mer.fa'),
                str(work / 'original_model_output'), '0-0-0-0-1'], cwd=paths['upstream_root'], check=True)
    elif phase == 'prepare':
        script('src/make_template.py', '--manifest', work / 'raw/af3_manifest.tsv',
               '--baseline-out', work / 'original_model_output', '--outdir', work / 'data')
        script('src/rebuild_joint_labels.py', '--allele-zip', paths['raw_zip'],
               '--manifest', work / 'raw/sample_manifest.tsv', '--template', work / 'data/outcome_template.tsv',
               '--outdir', work / 'data')
        script('src/prepare.py', '--work', work)
    elif phase in ['length', 'delta']:
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if phase == 'length':
            editor, member, candidate = ['ABE8e', 'D699'][index // 25], index % 25, 'length'
        else:
            editor, member = ['ABE8e', 'D699'][index // 50], (index % 50) // 2
            candidate = ['single_delta_nearest', 'delta_nearest'][index % 2]
        name = f'model_{member // 5 + 1}_{member % 5 + 1}'
        if not (work / 'runs' / editor / name / candidate / 'completed.json').exists():
            script('src/train.py', '--work', work, '--editor', editor, '--member', name, '--candidate', candidate)
    elif phase == 'summary':
        script('src/summarize.py', '--work', work)
    elif phase == 'transfer':
        fold = int(os.environ['SLURM_ARRAY_TASK_ID'])
        if not (work / 'transfer' / f'fold{fold:02d}' / 'completed.json').exists():
            script('src/transfer.py', '--work', work, '--fold', fold)
    elif phase == 'transfer-summary':
        script('src/transfer.py', '--work', work)
    elif phase == 'analysis-finish':
        from state import finish
        finish(work)
    elif phase == 'plot-data':
        script('plotting/prepare_plot_data.py', '--work', work, '--raw-zip', paths['raw_zip'])
    elif phase == 'plot':
        script('plotting/plot_figures.py', '--config', work / 'config.snapshot.json', '--work', work)
        from reproduce import figure_audit
        figure_audit(work)
    elif phase.startswith('deployment'):
        sys.path.insert(0, str(ROOT / 'deployment'))
        import manage
        if phase == 'deployment-plan':
            if not (deployment / 'plan.json').exists():
                manage.prepare(ROOT, work, deployment, config['training']['seed'])
        elif phase == 'deployment':
            index = int(os.environ['SLURM_ARRAY_TASK_ID'])
            plan = manage.load_plan(deployment)
            task = plan['tasks'][index]
            if not (deployment / 'members' / task['editor'] / task['member'] / 'completed.json').exists():
                from train_all import train
                train(deployment, index)
        elif phase == 'deployment-finish':
            manage.finish(deployment)
            from reproduce import deployment_manifest
            deployment_manifest(work)
        elif phase == 'deployment-verify':
            from verify_final_model import verify
            out = work / 'validation' / 'deployment'
            if out.exists():
                import shutil
                shutil.rmtree(out)  # Only this pipeline-owned, reproducible validation output.
            verify(ROOT / 'deployment', deployment, out)
    elif phase in ['compare-analysis', 'compare-deployment']:
        from reproduce import compare_analysis, compare_deployment
        if config['comparison']['enabled']:
            (compare_analysis if phase == 'compare-analysis' else compare_deployment)(work, config)
        else:
            write(work / 'comparison' / f'{phase}.json', {'status': 'disabled_by_configuration'})
    elif phase == 'finish':
        from reproduce import final_report
        final_report(work, config)


def worker(snapshot, work, phase):
    config = load_config(snapshot)
    metadata = read(work / 'pipeline.json')
    if metadata['code'] != code_hashes() or metadata['config'] != config:
        raise ValueError('Source code or the configuration snapshot changed during the run')
    index = int(os.environ['SLURM_ARRAY_TASK_ID']) if phase in ARRAYS else None
    output = marker(work, phase, index)
    if output.exists():
        print(f'Already complete: {phase}/{index}')
        return
    perform(phase, work, config)
    write(output, dict(phase=phase, task=index, complete=True))


def environment(config):
    runtime, paths = config['runtime'], config['paths']
    values = dict(PYTHONHASHSEED=config['training']['seed'], TF_DETERMINISTIC_OPS=1,
        CUDA_VISIBLE_DEVICES=runtime['cuda_visible_devices'], OMP_NUM_THREADS=runtime['omp_threads'],
        OPENBLAS_NUM_THREADS=runtime['blas_threads'], MKL_NUM_THREADS=runtime['blas_threads'],
        TF_NUM_INTRAOP_THREADS=runtime['tf_intra_threads'], TF_NUM_INTEROP_THREADS=runtime['tf_inter_threads'],
        RAW_ZIP=paths['raw_zip'], AF3_DIR=paths['af3_results'], UPSTREAM_ROOT=paths['upstream_root'],
        PRETRAINED_ROOT=paths['pretrained_root'])
    return values


def submit(phases, work, config):
    snapshot = work / 'config.snapshot.json'
    metadata = read(work / 'pipeline.json')
    jobs = metadata.setdefault('jobs', {})
    execution = config['execution']
    for phase in phases:
        if finished(work, phase):
            print(f'{phase}: complete; skipped')
            continue
        dependencies = []
        for parent in DEPS[phase]:
            if not finished(work, parent):
                if parent not in jobs:
                    raise ValueError(f'{phase} requires {parent}; run the earlier stage first')
                if parent not in phases and execution['backend'] == 'slurm':
                    active = subprocess.run(['squeue', '-h', '-j', jobs[parent]], capture_output=True, text=True, check=True)
                    if not active.stdout.strip():
                        raise ValueError(f'{phase} requires a completed {parent}; its recorded job is no longer active')
                dependencies.append(jobs[parent])
        previous = jobs.get(phase)
        if execution['backend'] == 'slurm' and previous:
            running = subprocess.run(['squeue', '-h', '-j', previous], capture_output=True, text=True, check=True)
            if running.stdout.strip():
                print(f'{phase}: existing active job {previous}; reused')
                continue
        command = ['bash', str(ROOT / 'hpc/job.sbatch'), str(ROOT), str(snapshot), str(work), phase,
                   execution['module'], execution['conda_env'], str(execution['activate_conda']).lower()]
        if execution['backend'] == 'local':
            indices = range(ARRAYS[phase][0], ARRAYS[phase][1] + 1) if phase in ARRAYS else [None]
            for index in indices:
                env = dict(os.environ, LDPRED_CONFIG=str(snapshot))
                if index is not None:
                    env['SLURM_ARRAY_TASK_ID'] = str(index)
                subprocess.run(command, env=env, check=True)
            continue
        options = ['sbatch', '--parsable', f'--job-name=ld_{phase}',
            f'--partition={execution["partition"]}', f'--account={execution["account"]}',
            '--nodes=1', '--ntasks=1', f'--cpus-per-task={execution["cpus_per_task"]}',
            f'--mem={execution["memory"]}', f'--output={work}/logs/{phase}_%A_%a.out',
            f'--error={work}/logs/{phase}_%A_%a.err']
        if execution['time']:
            options.append(f'--time={execution["time"]}')
        if dependencies:
            options += ['--dependency=afterok:' + ':'.join(dependencies), '--kill-on-invalid-dep=yes']
        if phase in ARRAYS:
            first, last = ARRAYS[phase]
            indices = [str(i) for i in range(first, last + 1) if not marker(work, phase, i).exists()]
            selection = f'{first}-{last}' if len(indices) == last - first + 1 else ','.join(indices)
            options.append('--array=' + selection + '%' + str(execution['concurrency'][phase]))
        # sbatch reads this script directly; do not put the shell executable before it.
        result = subprocess.run(options + command[1:], check=True, capture_output=True, text=True)
        identifier = result.stdout.strip().split(';')[0]
        if not identifier.isdigit():
            raise ValueError(f'Unexpected sbatch response: {result.stdout}')
        jobs[phase] = identifier
        write(work / 'pipeline.json', metadata)
        with (work / 'jobs.tsv').open('a') as stream:
            stream.write(f'{phase}\t{identifier}\t{":".join(dependencies)}\n')
        print(f'{phase}: submitted {identifier}', flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_env':
        snapshot = Path(sys.argv[2]).resolve()
        values = environment(load_config(snapshot))
        values['LDPRED_CONFIG'] = str(snapshot)
        for key, value in values.items():
            print(f'export {key}={shlex.quote(str(value))}')
        return
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        worker(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), sys.argv[4])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=[*MODES, 'status', 'predict', 'test'])
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--backend', choices=['slurm', 'local'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--editor', choices=['ABE8e', 'D699'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    work = (args.work or Path(config['paths']['work_dir'])).resolve()
    config['paths']['work_dir'] = str(work)
    if args.backend:
        config['execution']['backend'] = args.backend
    if args.stage == 'test':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'], check=True)
        return
    if args.stage == 'status':
        if not (work / 'pipeline.json').exists():
            print('Not started:', work)
            return
        jobs = read(work / 'pipeline.json').get('jobs', {})
        for phase in DEPS:
            print(f'{phase:22} {"complete" if finished(work, phase) else "pending":10} {jobs.get(phase, "")}')
        return
    if args.stage == 'predict':
        if not args.editor or not args.input or not args.out:
            parser.error('predict requires --editor, --input and --out')
        snapshot = work / 'config.snapshot.json'
        if not snapshot.is_file():
            parser.error('No completed deployment configuration; run model first')
        execution = load_config(snapshot)['execution']
        subprocess.run(['bash', str(ROOT / 'hpc/job.sbatch'), str(ROOT), str(snapshot), str(work), 'predict',
            execution['module'], execution['conda_env'], str(execution['activate_conda']).lower(),
            '--editor', args.editor, '--input', str(args.input.resolve()), '--out', str(args.out.resolve())], check=True)
        return
    phases = MODES[args.stage]
    if args.dry_run:
        for phase in phases:
            print(json.dumps(dict(phase=phase, dependencies=DEPS[phase], array=ARRAYS.get(phase),
                                  backend=config['execution']['backend']), ensure_ascii=False))
        print('work_dir:', work)
        return
    if work == ROOT or ROOT in work.parents:
        parser.error('Keep outputs outside the source-code directory')
    metadata_path = work / 'pipeline.json'
    if metadata_path.exists():
        metadata = read(metadata_path)
        if metadata['code'] != code_hashes() or metadata['config'] != config:
            parser.error('This output directory belongs to a different code/configuration; choose a fresh --work')
    else:
        if work.exists() and any(work.iterdir()):
            parser.error('Output directory is not empty; choose a fresh --work')
        (work / 'logs').mkdir(parents=True, exist_ok=True)
        (work / 'control').mkdir()
        write(work / 'config.snapshot.json', config)
        write(metadata_path, dict(config=config, code=code_hashes(), jobs={}))
    if args.stage in ['plot', 'verify', 'compare']:
        # Explicit audit/plot commands rerun; all/train continue completed stages.
        # A previous overall pass must not survive a newly failing audit.
        marker(work, 'finish').unlink(missing_ok=True)
        (work / 'reproduction_report.json').unlink(missing_ok=True)
        for phase in phases:
            marker(work, phase).unlink(missing_ok=True)
    submit(phases, work, config)


if __name__ == '__main__':
    main()

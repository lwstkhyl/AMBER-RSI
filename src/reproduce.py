"""Compare scientific values with historical runs; code/path metadata is recorded separately."""
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from common import save_json, save_table
from compare import file_fields
from state import sha


def chains(job):
    return [{kind: {'sequence': body['sequence'], 'count': body.get('count', 1)}}
            for entity in job.get('sequences', []) for kind, body in entity.items()]


def json_jobs(source, echo):
    source = Path(source)
    files = sorted(source.rglob('*')) if source.is_dir() else [source]
    jobs = {}
    for path in files:
        if not path.is_file():
            continue
        contents = []
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith('.json') and (not echo or name.endswith('_job_request.json')):
                        contents.append((str(path) + '::' + name, archive.read(name)))
        elif path.suffix.lower() == '.json' and (not echo or path.name.endswith('_job_request.json')):
            contents.append((str(path), path.read_bytes()))
        for origin, raw in contents:
            data = json.loads(raw)
            for job in data if isinstance(data, list) else [data]:
                if not isinstance(job, dict) or 'sequences' not in job or 'name' not in job:
                    continue
                key = re.sub(r'[^a-z0-9]', '', re.sub(r'^fold[_-]?', '', job['name'].lower()))
                if key in jobs and jobs[key][0] != job:
                    raise ValueError(f'Conflicting AF3 requests for {key}: {origin}')
                jobs[key] = job, origin
    return jobs


def compare_af3(work, config):
    submitted = config['paths']['af3_submitted_json']
    echo = not bool(submitted)
    source = submitted or config['paths']['af3_results']
    generated = json_jobs(work / 'af3_inputs/individual_server_json', False)
    historical = json_jobs(source, echo)
    rows = []
    for key, (job, _) in generated.items():
        old, origin = historical.get(key, ({}, ''))
        # Server echoes assign a model seed after submission. Preserve it in the
        # report, while comparing the submitted protein/RNA/DNA sequences/counts.
        same_sequences = chains(job) == chains(old)
        exact = job == old
        extras = [{kind: {k: v for k, v in body.items() if k not in ['sequence', 'count']}}
                  for entity in old.get('sequences', []) for kind, body in entity.items()]
        rows.append(dict(job=job['name'], reference=origin, found=bool(old),
            same_sequences_and_counts=same_sequences, same_complete_json=exact,
            generated_seeds=json.dumps(job.get('modelSeeds')), reference_seeds=json.dumps(old.get('modelSeeds')),
            reference_extra_chain_fields=json.dumps(extras, ensure_ascii=False),
            passed=same_sequences if echo else exact))
    table = pd.DataFrame(rows)
    report = dict(n_generated=len(generated), n_matched=int(table.found.sum()),
        all_passed=len(generated) == 36 and bool(table.passed.all()),
        comparison_scope='AF3 result job-request sequences/counts; server-assigned seeds reported separately' if echo else 'complete submitted JSON objects, ignoring whitespace/key order',
        full_submitted_json_comparison=not echo, reference=str(source))
    save_table(work / 'comparison/af3/jobs.tsv', table)
    save_json(work / 'comparison/af3/comparison.json', report)
    print(json.dumps(report, indent=2))
    if not report['all_passed']:
        raise ValueError('AF3 input comparison failed; inspect comparison/af3/jobs.tsv')


def compare_values(current, reference, output, names, atol):
    rows, differences = [], []
    for name in sorted(names):
        left, right = current / name, reference / name
        fields = file_fields(left, right, atol) if left.is_file() and right.is_file() else [('file', False, False, None, 'missing file')]
        maximum = max((f[3] for f in fields if f[3] is not None), default=0.0)
        rows.append(dict(file=name, identical_values=all(f[1] for f in fields),
                         within_atol=all(f[2] for f in fields), max_abs_difference=maximum))
        differences.extend(dict(file=name, field=f[0], identical_values=f[1], within_atol=f[2],
                                max_abs_difference=f[3], detail=f[4]) for f in fields if not f[1])
    table = pd.DataFrame(rows)
    report = dict(n_files=len(table), identical_values=bool(table.identical_values.all()),
        within_atol=bool(table.within_atol.all()), atol=atol, rtol=0,
        n_different_files=int((~table.identical_values).sum()),
        max_abs_numeric_difference=float(table.max_abs_difference.max()),
        current=str(current), reference=str(reference))
    save_table(output / 'files.tsv', table)
    save_table(output / 'differences.tsv', pd.DataFrame(differences, columns=[
        'file', 'field', 'identical_values', 'within_atol', 'max_abs_difference', 'detail']))
    save_json(output / 'comparison.json', report)
    return report


def compare_analysis(work, config):
    reports = []
    current_config = json.loads((work / 'run_config.json').read_text())
    for i, source in enumerate(config['paths']['reference_runs'], 1):
        reference = Path(source)
        marker = json.loads((reference / 'completed.json').read_text())
        if marker.get('complete') is not True or not marker.get('artifacts'):
            raise ValueError(f'Historical analysis is incomplete: {reference}')
        old_config = json.loads((reference / 'run_config.json').read_text())
        parameters = ['seed', 'epochs', 'patience', 'batch_size', 'learning_rate', 'bottleneck', 'checkpoint', 'editors', 'members']
        differences = {k: dict(current=current_config.get(k), reference=old_config.get(k))
                       for k in parameters if current_config.get(k) != old_config.get(k)}
        output = work / 'comparison' / f'analysis_reference{i:02d}'
        save_json(output / 'training_parameters.json', dict(identical=not differences, differences=differences,
            note='Code/configuration files differ by integration; they are archived, not required to be byte-identical.'))
        names = set(marker['artifacts']) | {'environment.json'}
        report = compare_values(work, reference, output, names, config['comparison']['numeric_atol'])
        report['training_parameters_identical'] = not differences
        save_json(output / 'comparison.json', report)
        reports.append(report)
    save_json(work / 'comparison/analysis.json', dict(references=reports,
        all_passed=bool(reports) and all(r['within_atol'] and r['training_parameters_identical'] for r in reports)))
    if not reports or any(not r['within_atol'] or not r['training_parameters_identical'] for r in reports):
        raise ValueError('Historical analysis comparison failed; inspect comparison/analysis_reference*/differences.tsv')


def compare_deployment(work, config):
    current = work / 'deployment/run01'
    reference = Path(config['paths']['reference_deployment'])
    marker = json.loads((reference / 'completed.json').read_text())
    if marker.get('complete') is not True or marker.get('n_members') != 50:
        raise ValueError('Historical deployment must contain all 50 completed members')
    # The new plan has different code paths and source fingerprints. Compare all
    # weights, seeds, scalers, epochs, histories, initial values and predictions.
    names = set(marker['artifacts']) - {'plan.json'}
    report = compare_values(current, reference, work / 'comparison/deployment', names,
                            config['comparison']['numeric_atol'])
    report['excluded_metadata'] = ['plan.json: new code/config paths and fingerprints; both originals remain archived']
    save_json(work / 'comparison/deployment/comparison.json', report)
    if not report['within_atol']:
        raise ValueError('Historical deployment comparison failed; inspect comparison/deployment/differences.tsv')


def deployment_manifest(work):
    folder = work / 'deployment/run01'
    plan = json.loads((folder / 'plan.json').read_text())
    models = []
    for task in plan['tasks']:
        weights = folder / 'members' / task['editor'] / task['member'] / 'weights.npz'
        models.append(dict(**task, weights=str(weights.relative_to(work)), weights_sha256=sha(weights)))
    save_json(work / 'model_manifest.json', dict(model='RLDpred-BE', n_members_per_editor=25,
        editors=plan['editors'], seed=plan['seed'], training_targets=plan['targets'], members=models,
        backbone_root=plan['pretrained_root'], backbone_files_sha256=plan['source_environment']['pretrained_files'],
        requires='25 original frozen SavedModels plus these 50 residual weight files and the saved input scalers',
        ensemble='mean both raw heads across 25 members before postprocessing',
        prediction_entry='bash run.sh predict --work THIS_RUN --editor ABE8e --input FEATURES.tsv --out OUTPUT.tsv'))


def figure_audit(work):
    summary = pd.read_csv(work / 'summary/total_results.tsv', sep='\t').set_index(['editor', 'candidate'])
    plotted = pd.read_csv(work / 'plot_data/plotted_values/main_b_metrics.tsv', sep='\t')
    expected = np.array([summary.loc[(r.editor, r.candidate), r.metric] for r in plotted.itertuples()])
    difference = float(np.max(np.abs(expected - plotted.value.to_numpy())))
    if len(plotted) != 32 or difference > 1e-10:
        raise ValueError('Figure b metrics do not match this run\'s analysis summary')
    record = json.loads((work / 'provenance/plot_run_all.json').read_text())
    paths = [Path(v['path']).resolve() for v in record['inputs'].values()]
    if any(work.resolve() not in p.parents for p in paths):
        raise ValueError('A figure is reading data from another run')
    files = sorted(p.name for p in (work / 'figures').iterdir() if p.is_file())
    if files != ['Fig1_RLDpred-BE.pdf', 'FigS1_RLDpred-BE.pdf']:
        raise ValueError('The figures directory must contain the two requested PDF files only')
    report = dict(n_main_metrics=32, max_metric_difference=difference, all_inputs_from_this_run=True,
        formats=['pdf'], pdf_files=files, layout_warnings=record['layout_warnings'])
    save_json(work / 'provenance/figure_audit.json', report)
    return report


def final_report(work, config):
    figures = figure_audit(work)
    af3 = json.loads((work / 'comparison/af3/comparison.json').read_text())
    validation = json.loads((work / 'validation/deployment/verification.json').read_text())
    historical = config['comparison']['enabled']
    analysis = json.loads((work / 'comparison/analysis.json').read_text()) if historical else None
    deployment = json.loads((work / 'comparison/deployment/comparison.json').read_text()) if historical else None
    report = dict(pipeline_complete=True, historical_comparison_performed=historical,
        historical_reproduction_verified=historical and analysis['all_passed'] and deployment['within_atol'],
        af3_input_comparison=af3, historical_analysis=analysis, historical_deployment=deployment,
        deployment_roundtrip=validation, figures=figures,
        interpretation='LOTO scores remain evaluation results; the all-12 model is for deployment.')
    save_json(work / 'reproduction_report.json', report)
    print(json.dumps(report, indent=2))

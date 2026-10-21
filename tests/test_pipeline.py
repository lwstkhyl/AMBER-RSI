"""Fast controller and comparison checks; no model training or synthetic paper scores."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import pipeline
from reproduce import compare_af3, compare_analysis


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ldpred_test_')
        self.work = Path(self.temp.name)
        self.config = pipeline.load_config(ROOT / 'config.json')

    def tearDown(self):
        self.temp.cleanup()

    def test_preserved_scientific_core(self):
        hashes = pipeline.read(ROOT / 'reference/scientific_core_sha256.json')
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_configuration_reaches_training_and_transfer(self):
        config = copy.deepcopy(self.config)
        config['training'] = dict(seed=17, epochs=6, patience=2, batch_size=4, learning_rate=.002, bottleneck=3)
        config['transfer']['penalties'] = [.02, .2]
        snapshot = self.work / 'config.json'
        pipeline.write(snapshot, config)
        source = 'import common,regression,json; print(json.dumps([common.SEED,common.EPOCHS,common.PATIENCE,common.BATCH_SIZE,common.LEARNING_RATE,common.BOTTLENECK,regression.PENALTIES]))'
        env = dict(os.environ, LDPRED_CONFIG=str(snapshot), PYTHONPATH=str(ROOT / 'src'))
        result = subprocess.run([sys.executable, '-c', source], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [17, 6, 2, 4, .002, 3, [.02, .2]])

    def test_all_slurm_dependencies_resources_and_reuse(self):
        work = self.work / 'output with spaces'
        pipeline.write(work / 'pipeline.json', dict(config=self.config, code=pipeline.code_hashes(), jobs={}))
        calls = []

        def fake_run(command, **kwargs):
            if command[0] == 'squeue':
                return SimpleNamespace(stdout='RUNNING\n')
            self.assertEqual(command[0], 'sbatch')
            calls.append(command)
            return SimpleNamespace(stdout=f'{8300000 + len(calls)};fixture\n')

        with patch.object(pipeline.subprocess, 'run', side_effect=fake_run):
            pipeline.submit(pipeline.MODES['all'], work, self.config)
            pipeline.submit(pipeline.MODES['all'], work, self.config)
        self.assertEqual(len(calls), 23)
        phases = pipeline.MODES['all']
        ids = {phase: str(8300001 + i) for i, phase in enumerate(phases)}
        execution = self.config['execution']
        expected_arrays = {phase: f'{first}-{last}%{execution["concurrency"][phase]}'
                           for phase, (first, last) in pipeline.ARRAYS.items()}
        for phase, command in zip(phases, calls):
            self.assertIn(f'--cpus-per-task={execution["cpus_per_task"]}', command)
            self.assertIn(f'--partition={execution["partition"]}', command)
            self.assertIn(f'--mem={execution["memory"]}', command)
            if pipeline.DEPS[phase]:
                dependency = '--dependency=afterok:' + ':'.join(ids[p] for p in pipeline.DEPS[phase])
                self.assertIn(dependency, command)
                self.assertIn('--kill-on-invalid-dep=yes', command)
            if phase in expected_arrays:
                self.assertIn('--array=' + expected_arrays[phase], command)
            position = command.index(str(ROOT / 'hpc/job.sbatch'))
            self.assertEqual(command[position + 3], str(work))
            self.assertEqual(command[position + 4], phase)

    def test_af3_exact_json_and_server_echo_modes(self):
        generated = self.work / 'af3_inputs/individual_server_json'
        reference = self.work / 'submitted'
        generated.mkdir(parents=True)
        reference.mkdir()
        for i in range(36):
            job = dict(name=f'target{i}_L20', modelSeeds=[], dialect='alphafoldserver', version=1,
                       sequences=[{'proteinChain': {'sequence': 'ACDE', 'count': 1}}])
            pipeline.write(generated / f'{i}.json', [job])
            pipeline.write(reference / f'{i}.json', [job])
        config = copy.deepcopy(self.config)
        config['paths']['af3_submitted_json'] = str(reference)
        compare_af3(self.work, config)
        self.assertTrue(pipeline.read(self.work / 'comparison/af3/comparison.json')['full_submitted_json_comparison'])
        echoed = self.work / 'echoed'
        echoed.mkdir()
        for p in reference.iterdir():
            job = pipeline.read(p)[0]
            job['modelSeeds'] = ['123']
            job['sequences'][0]['proteinChain']['useStructureTemplate'] = True
            pipeline.write(echoed / (p.stem + '_job_request.json'), [job])
        config['paths'].update(af3_submitted_json='', af3_results=str(echoed))
        compare_af3(self.work, config)
        report = pipeline.read(self.work / 'comparison/af3/comparison.json')
        self.assertTrue(report['all_passed'])
        self.assertFalse(report['full_submitted_json_comparison'])
        p = echoed / '0_job_request.json'
        changed = pipeline.read(p)
        changed[0]['sequences'][0]['proteinChain']['sequence'] = 'ACDF'
        pipeline.write(p, changed)
        with self.assertRaises(ValueError):
            compare_af3(self.work, config)

    def test_scientific_weights_are_compared_despite_changed_code_metadata(self):
        current, reference = self.work / 'new', self.work / 'old'
        parameters = dict(seed=42, epochs=2000, patience=100, batch_size=16, learning_rate=.0001,
            bottleneck=4, checkpoint='validation_joint_mse', editors=['ABE8e', 'D699'], members=['fixture'])
        for folder, label in [(current, 'new code'), (reference, 'old code')]:
            pipeline.write(folder / 'run_config.json', dict(**parameters, code={'metadata': label}))
            pipeline.write(folder / 'environment.json', dict(tensorflow='fixture', host=label))
            pipeline.write(folder / 'completed.json', dict(complete=True, artifacts=['weights.npz']))
            np.savez_compressed(folder / 'weights.npz', kernel=np.array([1., 2.]))
        config = copy.deepcopy(self.config)
        config['paths']['reference_runs'] = [str(reference)]
        compare_analysis(current, config)
        self.assertTrue(pipeline.read(current / 'comparison/analysis.json')['all_passed'])
        np.savez_compressed(current / 'weights.npz', kernel=np.array([1., 2.001]))
        with self.assertRaises(ValueError):
            compare_analysis(current, config)
        self.assertFalse(pipeline.read(current / 'comparison/analysis.json')['all_passed'])

    def test_rechecking_an_audit_invalidates_the_old_overall_pass(self):
        work = self.work / 'completed_run'
        config = copy.deepcopy(self.config)
        config['paths']['work_dir'] = str(work)
        pipeline.write(work / 'pipeline.json', dict(config=config, code=pipeline.code_hashes(), jobs={}))
        pipeline.write(pipeline.marker(work, 'finish'), dict(complete=True))
        pipeline.write(work / 'reproduction_report.json', dict(pipeline_complete=True))
        arguments = ['pipeline.py', 'verify', '--work', str(work), '--config', str(ROOT / 'config.json')]
        with patch.object(sys, 'argv', arguments), patch.object(pipeline, 'submit'):
            pipeline.main()
        self.assertFalse(pipeline.marker(work, 'finish').exists())
        self.assertFalse((work / 'reproduction_report.json').exists())


if __name__ == '__main__':
    unittest.main()

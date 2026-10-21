#!/usr/bin/env python3
"""Refit Length, then delta-nearest, on all 12 targets for one frozen backbone."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from manage import load_plan


def fit_fixed(model, inputs, truth, epochs, seed, folder, name, batch_size, learning_rate):
    import tensorflow as tf
    from common import save_table
    keras = tf.keras
    initial = {f'weight_{i}': v.numpy().copy() for i, v in enumerate(model.trainable_variables)}
    initial['variable_names'] = np.array([v.name for v in model.trainable_variables])
    initial['seed'] = np.array(seed)
    np.savez_compressed(folder / f'{name}_initial.npz', **initial)
    # Reuse the exact graph cut used in the LOTO trainer; only the branch changes.
    base = keras.Model(model.inputs[:5], model.get_layer('FrozenDense2Linear').output)
    train_model = keras.Model([model.get_layer('FrozenDense2Linear').output] + model.inputs[5:], model.output)
    cached = [base(inputs[:5], training=False).numpy()] + inputs[5:]
    direct = model([v[:16] for v in inputs], training=False).numpy()
    cached_prediction = train_model([v[:16] for v in cached], training=False).numpy()
    if np.max(np.abs(direct - cached_prediction)) > 1e-3:
        raise ValueError('Frozen-feature caching changes model predictions')
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 2
    dataset = tf.data.Dataset.from_tensor_slices((tuple(cached), truth))
    dataset = dataset.shuffle(len(truth), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).with_options(options).prefetch(1)
    train_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    history = train_model.fit(dataset, epochs=epochs, verbose=2)
    if not np.isfinite(history.history['loss']).all():
        raise ValueError('Nonfinite full-data training loss')
    save_table(folder / f'{name}_history.tsv', pd.DataFrame(dict(
        epoch=np.arange(1, epochs + 1), train_joint_mse=history.history['loss'])))


def train(run, index):
    plan = load_plan(run)
    import tensorflow as tf
    from common import RAW, read_input, condition_features, arrays, save_json
    from network import Backbone
    from state import runtime, sha
    task = plan['tasks'][index]
    seed = plan['seed']
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    environment = runtime()
    expected = plan['source_environment']
    for key in ['python', 'numpy', 'pandas', 'scipy', 'tensorflow', 'keras', 'variables']:
        value = expected[key]
        if key == 'variables':
            value = dict(value, PYTHONHASHSEED=str(seed))
        if environment[key] != value:
            raise ValueError(f'Deployment and LOTO environments differ: {key}')
    pretrained = Path(plan['pretrained_root'])
    for name, value in expected['pretrained_files'].items():
        if name.startswith(task['member'] + '/') and sha(pretrained / name) != value:
            raise ValueError(f'Original backbone changed: {name}')
    folder = run / 'members' / task['editor'] / task['member']
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / 'environment.json', environment)
    frame = read_input(run, task['editor'])
    conditions, delta = condition_features(frame)
    mean = float(conditions.sgRNA_length_numeric.mean())
    sd = float(conditions.sgRNA_length_numeric.std(ddof=0))
    sd = sd if sd >= 1e-8 else 1.0
    scale = delta.to_numpy(float).std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    groups = [frame[frame.condition_id.eq(cid)].sort_values('outcome_key') for cid in conditions.condition_id]
    ordered = pd.concat(groups, ignore_index=True)
    truth = ordered[['editing_efficiency', 'outcome_frequency']].to_numpy(np.float32)
    per_condition = [arrays(g, mean, sd, delta, scale) for g in groups]
    inputs = [np.concatenate([values[i] for values in per_condition]) for i in range(7)]
    backbone = Backbone(str(pretrained / task['member']))
    tf.keras.utils.set_random_seed(seed)
    length = backbone.build(seed)
    fit_fixed(length, inputs[:6], truth, task['length_epochs'], seed, folder, 'length',
              plan['batch_size'], plan['learning_rate'])
    length_weights = length.get_layer('LengthToDense2').get_weights()
    tf.keras.utils.set_random_seed(seed)
    combined = backbone.build(seed, length_weights)
    if np.max(np.abs(combined([v[:16] for v in inputs], training=False).numpy() -
                     length([v[:16] for v in inputs[:6]], training=False).numpy())) > 1e-3:
        raise ValueError('Zero delta projection must reproduce the refitted length branch')
    fit_fixed(combined, inputs, truth, task['delta_epochs'], seed, folder, 'delta',
              plan['batch_size'], plan['learning_rate'])
    parameters = dict(length_kernel=length_weights[0], length_bias=length_weights[1],
        bottleneck_kernel=combined.get_layer('DeltaDistanceBottleneck').get_weights()[0],
        projection_kernel=combined.get_layer('DeltaDistanceToDense2').get_weights()[0],
        length_mean=np.array([mean]), length_sd=np.array([sd]), delta_sd=scale,
        seed=np.array([seed]), length_epochs=np.array([task['length_epochs']]),
        delta_epochs=np.array([task['delta_epochs']]))
    if any(not np.isfinite(v).all() for v in parameters.values()):
        raise ValueError('Nonfinite deployment parameters')
    np.savez_compressed(folder / 'weights.npz', **parameters)
    raw = combined(inputs, training=False).numpy().astype(float)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite training-fit predictions')
    np.savez_compressed(folder / 'training_fit_predictions.npz',
                        outcome_id=ordered.outcome_id.to_numpy(str), raw=raw,
                        raw_columns=np.array(RAW))
    save_json(folder / 'completed.json', dict(**task, seed=seed,
        n_targets=12, n_conditions=36, n_outcomes=984))
    print(f"Full-data member complete: {task['editor']}/{task['member']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--task', type=int, choices=range(50), required=True)
    a = p.parse_args()
    train(a.run, a.task)

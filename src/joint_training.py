"""Original-style equal-row/equal-output MSE, Adam, batch16 and MSE checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from metrics import distribution_tv_pp

def efficiency_mse(y, p):
    return tf.square(p[:, 0] - y[:, 0])

def frequency_mse(y, p):
    return tf.square(p[:, 1] - y[:, 1])

def frequency_mae(y, p):
    return tf.abs(p[:, 1] - y[:, 1])

class BranchCheckpoint(keras.callbacks.Callback):
    """Save only changing weights: frozen backbone weights already exist on disk."""
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.best = np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        score = float(logs['val_loss'])
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation MSE')
        if score < self.best:
            self.best, self.best_epoch = score, epoch + 1
            self.best_weights = [v.numpy().copy() for v in self.model.trainable_variables]
            data = {f'weight_{i}': v for i, v in enumerate(self.best_weights)}
            data.update(variable_names=np.array([v.name for v in self.model.trainable_variables]),
                        epoch=np.array(self.best_epoch), validation_joint_mse=np.array(score))
            np.savez_compressed(self.path, **data)

def selected_validation_metrics(model, conditions):
    errors, maes, tvs = [], [], []
    for c in conditions:
        pred = model(c['inputs'], training=False).numpy()
        truth = np.column_stack([np.asarray(c['true_eff']), np.asarray(c['true_freq'])])
        errors.append(np.square(pred - truth))
        maes.append(float(np.abs(pred[:, 1] - truth[:, 1]).mean()))
        tvs.append(distribution_tv_pp(truth[:, 1], pred[:, 1]))
    error = np.concatenate(errors)
    return dict(best_validation_eff_mse=float(error[:, 0].mean()),
                best_validation_freq_mse=float(error[:, 1].mean()),
                best_validation_outcome_mae=float(np.mean(maes)),
                best_validation_distribution_tv_pp=float(np.mean(tvs)))

def train_joint(model, train_conditions, valid_conditions, fold_seed, args):
    fold = (fold_seed - args.seed) // 100
    initial = Path(args.outdir) / 'initial_weights'
    initial.mkdir(exist_ok=True)
    weights = {f'weight_{i}': v.numpy().copy() for i, v in enumerate(model.trainable_variables)}
    weights['variable_names'] = np.array([v.name for v in model.trainable_variables])
    weights['fold_seed'] = np.array(fold_seed)
    np.savez_compressed(initial / f'initial_fold{fold:02d}.npz', **weights)
    # Cutting the graph at the frozen Dense2 preactivation avoids recomputing
    # the unchanged CNN on every epoch. All downstream layer objects are shared.
    base = keras.Model(model.inputs[:5], model.get_layer('FrozenDense2Linear').output)
    train_model = keras.Model([model.get_layer('FrozenDense2Linear').output] + model.inputs[5:], model.output)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 2
    datasets = []
    cache_difference = 0.0
    for conditions, training in [(train_conditions, True), (valid_conditions, False)]:
        inputs = [np.concatenate([np.asarray(c['inputs'][i]) for c in conditions]) for i in range(len(model.inputs))]
        truth = np.concatenate([np.column_stack([np.asarray(c['true_eff']), np.asarray(c['true_freq'])]) for c in conditions])
        cached = [base(inputs[:5], training=False).numpy()] + inputs[5:]
        if not training:
            expected = model(inputs, training=False).numpy()
            actual = train_model(cached, training=False).numpy()
            cache_difference = float(np.max(np.abs(expected - actual)))
            if cache_difference > 1e-3:
                raise RuntimeError(f'Frozen-feature cache changes predictions: {cache_difference}')
            valid_inputs, valid_truth = cached, truth
        dataset = tf.data.Dataset.from_tensor_slices((tuple(cached), truth))
        if training:
            dataset = dataset.shuffle(len(truth), seed=fold_seed, reshuffle_each_iteration=True)
        datasets.append(dataset.batch(args.batch_size).with_options(options).prefetch(1))
    train_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss='mse',
                        metrics=[efficiency_mse, frequency_mse, frequency_mae])
    fold = (fold_seed - args.seed) // 100
    checkpoint = BranchCheckpoint(Path(args.outdir) / 'checkpoints' / f'best_fold{fold:02d}.npz')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=args.patience)
    history = train_model.fit(datasets[0], validation_data=datasets[1], epochs=args.epochs,
                              callbacks=[checkpoint, early_stop], verbose=2)
    # The original callbacks choose trained epochs; epoch0 is not a candidate.
    for variable, value in zip(train_model.trainable_variables, checkpoint.best_weights):
        variable.assign(value)
    restored_mse = float(np.square(train_model(valid_inputs, training=False).numpy() - valid_truth).mean())
    if not np.isclose(restored_mse, checkpoint.best, rtol=1e-5, atol=1e-3):
        raise RuntimeError('Restored checkpoint does not reproduce its validation MSE')
    frame = pd.DataFrame(history.history).rename(columns={
        'loss': 'train_mse', 'val_loss': 'val_mse',
        'efficiency_mse': 'train_eff_mse', 'val_efficiency_mse': 'val_eff_mse',
        'frequency_mse': 'train_freq_mse', 'val_frequency_mse': 'val_freq_mse',
        'frequency_mae': 'train_outcome_mae', 'val_frequency_mae': 'val_outcome_mae'})
    frame.insert(0, 'epoch', np.arange(1, len(frame) + 1))
    frame['checkpoint_metric'] = 'joint_mse'
    frame['max_cached_prediction_difference'] = cache_difference
    print(f'CHECKPOINT: fold={fold}, epoch={checkpoint.best_epoch}, validation_joint_MSE={checkpoint.best:.6f}', flush=True)
    return frame, checkpoint.best, checkpoint.best_epoch

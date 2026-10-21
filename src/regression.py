"""A six-coefficient fractional logit with continuous efficiency labels."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from common import equal_weights
from settings import CONFIG

PENALTIES = CONFIG['transfer']['penalties']
FEATURES = ['source_ABE8e_logit', 'source_D699_logit', 'nearest_c1', 'position_x', 'position_x2']

def feature_matrix(frame, with_distance):
    source = frame[['source_ABE8e_activity_pp', 'source_D699_activity_pp']].to_numpy(float) / 100
    if not np.isfinite(source).all() or (source < 0).any() or (source > 1 + 1e-8).any():
        raise ValueError('Source activities must be finite probabilities')
    x = (frame.position.to_numpy(float) - 10.5) / 9.5
    array = np.column_stack([logit(np.clip(source, 1e-4, 1 - 1e-4)), frame.nearest_c1, x, x ** 2])
    keep = np.arange(5) if with_distance else np.array([0, 1, 3, 4])
    return array[:, keep], [FEATURES[i] for i in keep]

def objective(beta, X, y, weights, penalty):
    z = X @ beta
    loss = float(np.dot(weights, np.logaddexp(0, z) - y * z) + penalty / 2 * np.dot(beta[1:], beta[1:]))
    gradient = X.T @ (weights * (expit(z) - y))
    gradient[1:] += penalty * beta[1:]
    return loss, gradient

def fit(train, validation, with_distance):
    raw, names = feature_matrix(train, with_distance)
    valid_raw, _ = feature_matrix(validation, with_distance)
    weights = equal_weights(train)
    mean = weights @ raw
    sd = np.sqrt(weights @ ((raw - mean) ** 2))
    sd[sd < 1e-8] = 1
    X = np.column_stack([np.ones(len(train)), (raw - mean) / sd])
    V = np.column_stack([np.ones(len(validation)), (valid_raw - mean) / sd])
    y = train.true_position_pp.to_numpy(float) / 100
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1 + 1e-8).any():
        raise ValueError('Continuous TeABE labels must be in [0,1]')
    initial = np.zeros(X.shape[1])
    initial[0] = logit(np.clip(np.dot(weights, y), 1e-6, 1 - 1e-6))
    best, history = None, []
    for penalty in PENALTIES:
        result = minimize(objective, initial, args=(X, y, weights, penalty), jac=True, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'Fractional logit failed to converge: {result.message}')
        errors = validation[['target_id', 'condition_id']].copy()
        errors['mse_pp2'] = (100 * expit(V @ result.x) - validation.true_position_pp.to_numpy(float)) ** 2
        score = float(errors.groupby(['target_id', 'condition_id']).mse_pp2.mean().groupby('target_id').mean().mean())
        history.append(dict(penalty=penalty, validation_mse_pp2=score, objective=float(result.fun),
                            n_iterations=int(result.nit), max_gradient=float(np.abs(result.jac).max())))
        if best is None or score < best['validation_mse_pp2']:
            beta = result.x
            best = dict(features=names, feature_mean=mean.tolist(), feature_sd=sd.tolist(),
                        coefficients_standardized=beta.tolist(), penalty=penalty, validation_mse_pp2=score,
                        intercept_raw=float(beta[0] - np.dot(beta[1:], mean / sd)),
                        coefficients_raw=(beta[1:] / sd).tolist(),
                        source_clip_probability=1e-4, position_x='(position - 10.5) / 9.5',
                        n_parameters=int(len(beta)), with_distance=with_distance,
                        train_targets=sorted(train.target_id.unique()), validation_targets=sorted(validation.target_id.unique()),
                        train_source_scaler_only=True, target_labels_are_continuous=True,
                        training_loss='balanced fractional cross-entropy + L2 on slopes; intercept unpenalized')
    return best, history

def predict(frame, model):
    raw, names = feature_matrix(frame, model['with_distance'])
    if names != model['features']:
        raise ValueError('Regression feature names differ')
    X = np.column_stack([np.ones(len(frame)), (raw - model['feature_mean']) / model['feature_sd']])
    return 100 * expit(X @ model['coefficients_standardized'])

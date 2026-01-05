"""Compute SHAP values for descriptor matrices."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    import shap
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "The 'shap' package is required for SHAP analysis. "
        "Install project dependencies via 'pip install -r requirements.txt' "
        "before enabling analysis['shap_validation']."
    ) from exc
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def compute_shap_analysis_payload(descriptors_list: Sequence[Sequence[float]],
                                  config: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> Dict[str, Any]:
    feature_matrix = _validate_feature_matrix(descriptors_list)
    feature_names = _resolve_feature_names(config, analysis, feature_matrix.shape[1])
    targets, target_meta = _extract_targets(config, feature_matrix.shape[0])
    task = _resolve_task(config, targets)
    scaled_matrix, scaler = _maybe_scale_features(feature_matrix, config)
    estimator = _instantiate_estimator(task, config)
    estimator.fit(scaled_matrix, targets)
    predictions, probabilities, metrics = _evaluate_model(estimator, scaled_matrix, targets, task, config)
    shap_values, expected_value, explainer_meta = _compute_shap_values(estimator,
                                                                      scaled_matrix,
                                                                      task,
                                                                      config)
    importance = np.mean(np.abs(shap_values), axis=0)
    ordered_indices = np.argsort(importance)[::-1]
    payload = {
        'feature_matrix': feature_matrix,
        'model_matrix': scaled_matrix,
        'feature_names': feature_names,
        'targets': targets,
        'task': task,
        'estimator': estimator,
        'estimator_name': type(estimator).__name__,
        'predictions': predictions,
        'probabilities': probabilities,
        'metrics': metrics,
        'shap_values': shap_values,
        'expected_value': expected_value,
        'importance': importance,
        'ordered_indices': ordered_indices,
        'explainer': explainer_meta,
        'scaler': scaler,
        'target_meta': target_meta,
        'sample_labels': _extract_sample_labels(analysis, feature_matrix.shape[0]),
    }
    return payload


def _validate_feature_matrix(data: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("SHAP analysis expects a 2D descriptor matrix.")
    if array.shape[0] < 2 or array.shape[1] == 0:
        raise ValueError("SHAP analysis requires at least two samples and one descriptor.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Descriptor matrix contains NaN or infinite values.")
    return array


def _resolve_feature_names(config: Dict[str, Any],
                           analysis: Dict[str, Any],
                           n_features: int) -> List[str]:
    candidate = config.get('feature_names') or analysis.get('descriptor_names')
    if isinstance(candidate, list) and len(candidate) == n_features:
        return [str(name) for name in candidate]
    return [f"descriptor_{idx + 1}" for idx in range(n_features)]


def _extract_targets(config: Dict[str, Any], n_samples: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    if 'targets' not in config:
        raise ValueError("analysis['shap_validation'] must provide a 'targets' list aligned with the descriptors.")
    raw = np.asarray(config['targets'])
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if raw.shape[0] != n_samples:
        raise ValueError("Targets length must match the number of samples in descriptors.")
    meta: Dict[str, Any] = {}
    if raw.dtype.kind in {'U', 'S', 'O'}:
        unique = list(dict.fromkeys(raw.tolist()))
        encoder = {label: idx for idx, label in enumerate(unique)}
        numeric = np.array([encoder[value] for value in raw], dtype=float)
        meta['encoder'] = encoder
        meta['original_labels'] = raw.tolist()
        return numeric, meta
    return raw.astype(float), meta


def _resolve_task(config: Dict[str, Any], targets: np.ndarray) -> str:
    candidate = str(config.get('task', '')).strip().lower()
    if candidate in {'regression', 'classification'}:
        return candidate
    unique = np.unique(targets)
    if unique.size <= 10 and np.allclose(unique, unique.astype(int)):
        return 'classification'
    return 'regression'


def _maybe_scale_features(matrix: np.ndarray,
                          config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    scaling = str(config.get('scaling', 'none')).lower()
    if scaling in {'none', 'off', 'false', ''}:
        return matrix.copy(), None
    if scaling != 'standard':
        raise ValueError("Only 'standard' scaling or 'none' is supported for SHAP analysis.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def _instantiate_estimator(task: str, config: Dict[str, Any]):
    model_type = str(config.get('model', 'random_forest')).lower()
    random_state = int(config.get('random_state', 42))
    n_estimators = int(config.get('n_estimators', 600))
    max_depth = config.get('max_depth')
    min_samples_leaf = int(config.get('min_samples_leaf', 1))
    if model_type == 'random_forest':
        if task == 'regression':
            return RandomForestRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf,
                                         random_state=random_state,
                                         n_jobs=int(config.get('n_jobs', -1)))
        return RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=random_state,
                                      n_jobs=int(config.get('n_jobs', -1)),
                                      class_weight=config.get('class_weight'))

    if model_type == 'extra_trees':
        if task == 'regression':
            return ExtraTreesRegressor(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=random_state,
                                       n_jobs=int(config.get('n_jobs', -1)))
        return ExtraTreesClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=random_state,
                                    n_jobs=int(config.get('n_jobs', -1)),
                                    class_weight=config.get('class_weight'))

    if model_type in {'hist_gradient_boosting', 'hgb'}:
        if task == 'regression':
            return HistGradientBoostingRegressor(max_depth=max_depth,
                                                 learning_rate=float(config.get('learning_rate', 0.05)),
                                                 max_iter=int(config.get('max_iter', 400)),
                                                 random_state=random_state)
        return HistGradientBoostingClassifier(max_depth=max_depth,
                                              learning_rate=float(config.get('learning_rate', 0.05)),
                                              max_iter=int(config.get('max_iter', 400)),
                                              random_state=random_state,
                                              class_weight=config.get('class_weight'))

    raise ValueError(f"Unsupported SHAP model '{model_type}'. Choose from random_forest, extra_trees, or hist_gradient_boosting.")


def _evaluate_model(estimator,
                    features: np.ndarray,
                    targets: np.ndarray,
                    task: str,
                    config: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    metrics: Dict[str, float] = {}
    probabilities: Optional[np.ndarray] = None
    if task == 'regression':
        predictions = estimator.predict(features)
        metrics['r2'] = float(r2_score(targets, predictions))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(targets, predictions)))
        metrics['mae'] = float(mean_absolute_error(targets, predictions))
        return predictions, probabilities, metrics

    class_index = _resolve_class_index(config, _infer_n_classes(estimator, targets))
    if hasattr(estimator, 'predict_proba'):
        probabilities = estimator.predict_proba(features)
        positive_scores = probabilities[:, class_index]
    elif hasattr(estimator, 'decision_function'):
        decision = estimator.decision_function(features)
        if decision.ndim == 1:
            positive_scores = _sigmoid(decision)
        else:
            positive_scores = _softmax(decision)[:, class_index]
    else:
        positive_scores = estimator.predict(features)

    predictions = estimator.predict(features)
    metrics['accuracy'] = float(accuracy_score(targets, predictions))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(targets, predictions))
    metrics['f1'] = float(f1_score(targets, predictions, average='weighted'))
    try:
        metrics['roc_auc'] = float(roc_auc_score(targets, positive_scores))
    except Exception:
        pass
    return predictions, probabilities, metrics


def _infer_n_classes(estimator, targets: np.ndarray) -> int:
    if hasattr(estimator, 'n_classes_'):
        return int(getattr(estimator, 'n_classes_'))
    return int(np.unique(targets).size)


def _resolve_class_index(config: Dict[str, Any], n_classes: int) -> int:
    index = int(config.get('class_index', 1 if n_classes > 1 else 0))
    return max(0, min(index, max(n_classes - 1, 0)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray) -> np.ndarray:
    exp = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _compute_shap_values(estimator,
                         features: np.ndarray,
                         task: str,
                         config: Dict[str, Any]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    mode = str(config.get('explainer', 'auto')).lower()
    metadata: Dict[str, Any] = {}
    background, background_size = _select_background(features, config)
    metadata['background_size'] = background_size

    if mode in {'auto', 'tree'}:
        try:
            perturbation = str(config.get('feature_dependence', 'interventional'))
            explainer = shap.TreeExplainer(estimator,
                                           data=background,
                                           feature_perturbation=perturbation)
            shap_raw = explainer.shap_values(features)
            shap_matrix = _to_matrix(shap_raw, task, config)
            expected = _select_expected_value(explainer.expected_value, task, config)
            metadata['mode'] = 'tree'
            metadata['perturbation'] = perturbation
            return shap_matrix, float(np.atleast_1d(expected)[0]), metadata
        except Exception as exc:
            print("TreeExplainer failed (", exc, ") - falling back to KernelExplainer.")
    def prediction_fn(batch):
        return estimator.predict(batch)

    explainer = shap.KernelExplainer(prediction_fn, background)
    nsamples = config.get('kernel_nsamples')
    shap_raw = explainer.shap_values(features, nsamples=nsamples)
    shap_matrix = _to_matrix(shap_raw, task, config)
    expected = _select_expected_value(explainer.expected_value, task, config)
    metadata['mode'] = 'kernel'
    metadata['kernel_nsamples'] = nsamples
    return shap_matrix, float(np.atleast_1d(expected)[0]), metadata


def _select_background(features: np.ndarray,
                       config: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    background_size = int(config.get('background_size', min(200, features.shape[0])))
    rng = np.random.default_rng(int(config.get('random_state', 42)))
    if background_size >= features.shape[0]:
        return features, features.shape[0]
    selection = rng.choice(features.shape[0], size=background_size, replace=False)
    return features[selection], background_size


def _to_matrix(values, task: str, config: Dict[str, Any]) -> np.ndarray:
    if isinstance(values, list):
        class_index = _resolve_class_index(config, len(values))
        matrix = values[class_index]
    else:
        matrix = values
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("Unexpected SHAP values shape. Expected 2D matrix.")
    return array


def _select_expected_value(value, task: str, config: Dict[str, Any]):
    if isinstance(value, list):
        index = _resolve_class_index(config, len(value))
        return value[index]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value.flatten()[0]
    return value


def _extract_sample_labels(analysis: Dict[str, Any], n_samples: int) -> List[str]:
    labels = analysis.get('molecules_label')
    if isinstance(labels, list) and len(labels) == n_samples:
        return [str(label) for label in labels]
    return [f"sample_{idx + 1}" for idx in range(n_samples)]

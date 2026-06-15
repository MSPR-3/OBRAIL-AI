import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN
from src.data import load_dataset, split_labelled_frame, validate_dataset
from src.evaluation import (
    compute_metrics,
    encode_target,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.models import ModelSpec, build_model_specs

LOGGER = logging.getLogger(__name__)

SCORING = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'roc_auc_ovr_weighted': make_scorer(roc_auc_score, response_method='predict_proba', multi_class='ovr', average='weighted'),
}

@dataclass(frozen=True)
class CandidateResult:
    name: str
    best_estimator: Pipeline
    cv_results: dict[str, float]
    validation_metrics: dict[str, float]
    best_params: dict[str, Any]
    search: Any

@dataclass(frozen=True)
class TrainingArtifacts:
    selected_model_name: str
    model_path: Path
    summary_path: Path
    candidates_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path
    model_bytes: bytes = b''
    test_metrics: dict[str, Any] | None = None
    n_rows_train: int = 0

def fit_search(spec: ModelSpec, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: np.ndarray, cv_splits: int = 5, random_state: int = RANDOM_STATE) -> Any:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search_cls = GridSearchCV if spec.search_kind == 'grid' else RandomizedSearchCV
    search_kwargs: dict[str, Any] = {'estimator': spec.estimator, 'param_grid' if spec.search_kind == 'grid' else 'param_distributions': spec.param_space, 'scoring': SCORING, 'refit': 'f1_macro', 'cv': cv, 'n_jobs': -1, 'verbose': 0, 'error_score': 'raise'}
    if spec.search_kind == 'random': search_kwargs.update({'n_iter': spec.n_iter, 'random_state': random_state})
    search = search_cls(**search_kwargs)
    search.fit(X_train, y_train, model__sample_weight=sample_weight)
    return search

def summarize_search(search: Any) -> dict[str, float]:
    return {
        'cv_accuracy_mean': float(search.cv_results_['mean_test_accuracy'][search.best_index_]),
        'cv_f1_macro_mean': float(search.cv_results_['mean_test_f1_macro'][search.best_index_]),
        'cv_roc_auc_mean': float(search.cv_results_['mean_test_roc_auc_ovr_weighted'][search.best_index_]),
    }

def retrain_final_model(best_result: CandidateResult, X_trainval: pd.DataFrame, y_trainval: np.ndarray) -> Pipeline:
    final_estimator = clone(best_result.best_estimator)
    final_estimator.fit(X_trainval, y_trainval, model__sample_weight=compute_sample_weight('balanced', y_trainval))
    return final_estimator

def run_training_pipeline(data_path: str | Path = 'data/obrail_features.csv', artifact_dir: str | Path = 'artifacts/member2', cv_splits: int = 5, df: pd.DataFrame | None = None) -> TrainingArtifacts:
    if df is None:
        df = load_dataset(data_path)
    validate_dataset(df)
    train_df, val_df, test_df = split_labelled_frame(df)

    label_encoder = encode_target(train_df[TARGET_COLUMN].astype(str))
    class_names = label_encoder.classes_.tolist()

    X_train, X_val, X_test = train_df[FEATURE_COLUMNS].copy(), val_df[FEATURE_COLUMNS].copy(), test_df[FEATURE_COLUMNS].copy()
    y_train = label_encoder.transform(train_df[TARGET_COLUMN].astype(str))
    y_val = label_encoder.transform(val_df[TARGET_COLUMN].astype(str))
    y_test = label_encoder.transform(test_df[TARGET_COLUMN].astype(str))

    sample_weight = compute_sample_weight('balanced', y_train)
    specs = build_model_specs(n_classes=len(class_names))

    candidate_results: list[CandidateResult] = []
    for spec in specs:
        LOGGER.info('Recherche d\'hyperparamètres: %s', spec.name)
        search = fit_search(spec, X_train, y_train, sample_weight, cv_splits)
        best_estimator = search.best_estimator_
        val_metrics = compute_metrics(y_val, best_estimator.predict(X_val), best_estimator.predict_proba(X_val), class_names)
        candidate_results.append(CandidateResult(spec.name, best_estimator, summarize_search(search), val_metrics, search.best_params_, search))
        LOGGER.info('%s | CV F1 macro=%.4f | VAL F1 macro=%.4f', spec.name, summarize_search(search)['cv_f1_macro_mean'], val_metrics['f1_macro'])

    selected = max(candidate_results, key=lambda item: (item.validation_metrics['f1_macro'], item.cv_results['cv_f1_macro_mean']))
    LOGGER.info('Modèle sélectionné: %s', selected.name)

    final_model = retrain_final_model(selected, pd.concat([X_train, X_val], axis=0), np.concatenate([y_train, y_val]))
    test_metrics = compute_metrics(y_test, final_model.predict(X_test), final_model.predict_proba(X_test), class_names)
    LOGGER.info('TEST | Accuracy=%.4f | F1 macro=%.4f | AUC=%.4f', test_metrics['accuracy'], test_metrics['f1_macro'], test_metrics['roc_auc_ovr_weighted'])

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    summary_payload = {'model_name': selected.name, 'feature_columns': FEATURE_COLUMNS, 'test_metrics': test_metrics}
    (artifact_dir / 'training_summary.json').write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding='utf-8')

    artifact_obj = {'model_name': selected.name, 'pipeline': final_model, 'label_encoder': label_encoder, 'class_names': class_names}
    buffer = io.BytesIO()
    joblib.dump(artifact_obj, buffer)
    model_bytes = buffer.getvalue()
    (artifact_dir / 'best_model.joblib').write_bytes(model_bytes)

    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, artifact_dir / 'confusion_matrix_test.png', 'Matrice de confusion')
    plot_feature_importance(final_model, artifact_dir / 'feature_importance.png')

    return TrainingArtifacts(
        selected.name,
        artifact_dir / 'best_model.joblib',
        artifact_dir / 'training_summary.json',
        artifact_dir / 'candidate_results.csv',
        artifact_dir / 'confusion_matrix_test.png',
        artifact_dir / 'feature_importance.png',
        model_bytes=model_bytes,
        test_metrics=test_metrics,
        n_rows_train=int(len(X_train) + len(X_val)),
    )
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')

RANDOM_STATE = 42
TARGET_COLUMN = 'classe_substitution'
SPLIT_COLUMN = 'split_classif'
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'
NON_LABELLED = 'non_etiquete'
CLASS_ORDER = ['non_pertinent', 'substitution_difficile', 'substitution_possible']

FEATURE_COLUMNS = [
    'duree_minutes',
    'heure_decimale',
    'is_nuit',
    'is_transfrontalier',
    'code_pays_dep',
    'code_pays_arr',
]
NUMERIC_COLUMNS = [
    'duree_minutes',
    'heure_decimale',
    'is_nuit',
    'is_transfrontalier',
]
CATEGORICAL_COLUMNS = ['code_pays_dep', 'code_pays_arr']
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN])

SCORING = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'roc_auc_ovr_weighted': make_scorer(
        roc_auc_score,
        response_method='predict_proba',
        multi_class='ovr',
        average='weighted',
    ),
}

LOGGER = logging.getLogger('member2_ml')


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    param_space: dict[str, Any]
    search_kind: str = 'random'
    n_iter: int = 20


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


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(message)s')


def resolve_dataset_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.exists():
        return path
    candidates = [Path('data/obrail_features.csv'), Path('../data/obrail_features.csv')]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Impossible de trouver le fichier de données: {data_path}')


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = resolve_dataset_path(data_path)
    df = pd.read_csv(path, low_memory=False)
    LOGGER.info('Dataset chargé: %s (%s lignes, %s colonnes)', path, len(df), len(df.columns))
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Colonnes manquantes dans le dataset: {sorted(missing)}')

    labelled = df[df[SPLIT_COLUMN].isin({TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT})]
    if labelled.empty:
        raise ValueError('Aucune ligne étiquetée trouvée pour la classification.')

    target_missing = labelled[TARGET_COLUMN].isna().mean() * 100
    if target_missing > 0:
        LOGGER.warning('La cible contient %.2f %% de valeurs manquantes dans le périmètre étiqueté.', target_missing)

    leakage_columns = ['distance_km', 'vitesse_kmh', 'id_trajet', 'split', 'split_classif', 'classe_substitution']
    present_leakage = [column for column in leakage_columns if column in df.columns]
    LOGGER.info('Colonnes de fuite/protocole présentes: %s', present_leakage)


def split_labelled_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = df[df[SPLIT_COLUMN].isin({TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT})].copy()
    train_df = labelled[labelled[SPLIT_COLUMN] == TRAIN_SPLIT].copy()
    val_df = labelled[labelled[SPLIT_COLUMN] == VAL_SPLIT].copy()
    test_df = labelled[labelled[SPLIT_COLUMN] == TEST_SPLIT].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Les splits train/val/test doivent tous être non vides.')
    if train_df[TARGET_COLUMN].isna().any() or val_df[TARGET_COLUMN].isna().any() or test_df[TARGET_COLUMN].isna().any():
        raise ValueError('La cible ne doit pas être manquante dans train/val/test.')

    return train_df, val_df, test_df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, NUMERIC_COLUMNS),
            ('categorical', categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def build_model_specs(n_classes: int) -> list[ModelSpec]:
    preprocessor = build_preprocessor()
    specs: list[ModelSpec] = [
        ModelSpec(
            name='logistic_regression',
            estimator=Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ]
            ),
            param_space={
                'model__C': [0.1, 0.3, 1.0, 3.0, 10.0],
                'model__solver': ['lbfgs'],
            },
            search_kind='grid',
        ),
        ModelSpec(
            name='random_forest',
            estimator=Pipeline(
                steps=[
                    ('preprocessor', preprocessor),
                    ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            param_space={
                'model__n_estimators': [300, 500, 800],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 0.5],
                'model__class_weight': [None],
            },
            search_kind='random',
            n_iter=20,
        ),
    ]

    if xgb is not None:
        specs.append(
            ModelSpec(
                name='xgboost',
                estimator=Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        (
                            'model',
                            xgb.XGBClassifier(
                                objective='multi:softprob',
                                num_class=n_classes,
                                eval_metric='mlogloss',
                                tree_method='hist',
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
                param_space={
                    'model__n_estimators': [200, 400, 600],
                    'model__max_depth': [3, 5, 7, 9],
                    'model__learning_rate': [0.01, 0.05, 0.1],
                    'model__subsample': [0.7, 0.85, 1.0],
                    'model__colsample_bytree': [0.7, 0.85, 1.0],
                    'model__min_child_weight': [1, 3, 5],
                    'model__reg_alpha': [0.0, 0.1, 1.0],
                    'model__reg_lambda': [1.0, 2.0, 5.0],
                },
                search_kind='random',
                n_iter=20,
            )
        )
    else:
        LOGGER.warning('XGBoost est absent de l\'environnement. Le modèle sera ignoré.')

    if lgb is not None:
        specs.append(
            ModelSpec(
                name='lightgbm',
                estimator=Pipeline(
                    steps=[
                        ('preprocessor', preprocessor),
                        (
                            'model',
                            lgb.LGBMClassifier(
                                objective='multiclass',
                                num_class=n_classes,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
                param_space={
                    'model__n_estimators': [200, 400, 600],
                    'model__num_leaves': [15, 31, 63],
                    'model__learning_rate': [0.01, 0.05, 0.1],
                    'model__subsample': [0.7, 0.85, 1.0],
                    'model__colsample_bytree': [0.7, 0.85, 1.0],
                    'model__min_child_samples': [10, 20, 40],
                    'model__reg_alpha': [0.0, 0.1, 1.0],
                    'model__reg_lambda': [0.0, 1.0, 5.0],
                },
                search_kind='random',
                n_iter=20,
            )
        )
    else:
        LOGGER.warning('LightGBM est absent de l\'environnement. Le modèle sera ignoré.')

    return specs


def encode_target(y: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(y.astype(str))
    return encoder


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc_ovr_weighted': float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).tolist(),
        'classification_report': classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def fit_search(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Any:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search_cls = GridSearchCV if spec.search_kind == 'grid' else RandomizedSearchCV
    search_kwargs: dict[str, Any] = {
        'estimator': spec.estimator,
        'param_grid' if spec.search_kind == 'grid' else 'param_distributions': spec.param_space,
        'scoring': SCORING,
        'refit': 'f1_macro',
        'cv': cv,
        'n_jobs': -1,
        'verbose': 0,
        'return_train_score': False,
        'error_score': 'raise',
    }
    if spec.search_kind == 'random':
        search_kwargs['n_iter'] = spec.n_iter
        search_kwargs['random_state'] = random_state

    search = search_cls(**search_kwargs)
    search.fit(X_train, y_train, model__sample_weight=sample_weight)
    return search


def summarize_search(search: Any) -> dict[str, float]:
    return {
        'cv_accuracy_mean': float(search.cv_results_['mean_test_accuracy'][search.best_index_]),
        'cv_accuracy_std': float(search.cv_results_['std_test_accuracy'][search.best_index_]),
        'cv_f1_macro_mean': float(search.cv_results_['mean_test_f1_macro'][search.best_index_]),
        'cv_f1_macro_std': float(search.cv_results_['std_test_f1_macro'][search.best_index_]),
        'cv_roc_auc_mean': float(search.cv_results_['mean_test_roc_auc_ovr_weighted'][search.best_index_]),
        'cv_roc_auc_std': float(search.cv_results_['std_test_roc_auc_ovr_weighted'][search.best_index_]),
    }


def get_feature_names(best_estimator: Pipeline) -> list[str]:
    preprocessor = best_estimator.named_steps['preprocessor']
    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]


def plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    values = np.asarray(matrix)
    im = ax.imshow(values, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title(title)

    threshold = values.max() / 2 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f'{values[i, j]}', ha='center', va='center', color='white' if values[i, j] > threshold else 'black')

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(best_estimator: Pipeline, output_path: Path, top_n: int = 20) -> None:
    model = best_estimator.named_steps['model']
    feature_names = get_feature_names(best_estimator)

    if hasattr(model, 'coef_'):
        importances = np.abs(np.asarray(model.coef_)).mean(axis=0)
    elif hasattr(model, 'feature_importances_'):
        importances = np.asarray(model.feature_importances_)
    else:
        LOGGER.warning('Le modèle ne fournit pas de feature importance exploitable.')
        return

    ranking = (
        pd.DataFrame({'feature': feature_names, 'importance': importances})
        .sort_values('importance', ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(ranking) + 2)))
    ax.barh(ranking['feature'], ranking['importance'], color='#356AE6')
    ax.set_title('Top features du modèle final')
    ax.set_xlabel('Importance')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def build_results_table(results: list[CandidateResult]) -> pd.DataFrame:
    rows = []
    for item in results:
        row = {'model': item.name}
        row.update(item.cv_results)
        row.update({f'val_{key}': value for key, value in item.validation_metrics.items() if key in {'accuracy', 'f1_macro', 'roc_auc_ovr_weighted'}})
        row['best_params'] = json.dumps(item.best_params, ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_result(results: list[CandidateResult]) -> CandidateResult:
    if not results:
        raise ValueError('Aucun modèle candidat disponible.')
    return max(results, key=lambda item: (item.validation_metrics['f1_macro'], item.cv_results['cv_f1_macro_mean']))


def retrain_final_model(best_result: CandidateResult, X_trainval: pd.DataFrame, y_trainval: np.ndarray) -> Pipeline:
    final_estimator = clone(best_result.best_estimator)
    sample_weight = compute_sample_weight('balanced', y_trainval)
    final_estimator.fit(X_trainval, y_trainval, model__sample_weight=sample_weight)
    return final_estimator


def run_training_pipeline(
    data_path: str | Path = 'data/obrail_features.csv',
    artifact_dir: str | Path = 'artifacts/member2',
    cv_splits: int = 5,
) -> TrainingArtifacts:
    df = load_dataset(data_path)
    validate_dataset(df)
    train_df, val_df, test_df = split_labelled_frame(df)

    label_encoder = encode_target(train_df[TARGET_COLUMN].astype(str))
    class_names = label_encoder.classes_.tolist()

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_val = val_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()

    y_train = label_encoder.transform(train_df[TARGET_COLUMN].astype(str))
    y_val = label_encoder.transform(val_df[TARGET_COLUMN].astype(str))
    y_test = label_encoder.transform(test_df[TARGET_COLUMN].astype(str))

    sample_weight = compute_sample_weight('balanced', y_train)
    specs = build_model_specs(n_classes=len(class_names))
    LOGGER.info('Modèles candidats disponibles: %s', [spec.name for spec in specs])

    candidate_results: list[CandidateResult] = []
    for spec in specs:
        LOGGER.info('Recherche d\'hyperparamètres: %s', spec.name)
        search = fit_search(spec, X_train, y_train, sample_weight=sample_weight, cv_splits=cv_splits)
        cv_summary = summarize_search(search)
        best_estimator = search.best_estimator_
        y_val_pred = best_estimator.predict(X_val)
        y_val_proba = best_estimator.predict_proba(X_val)
        validation_metrics = compute_metrics(y_val, y_val_pred, y_val_proba, class_names)
        candidate_results.append(
            CandidateResult(
                name=spec.name,
                best_estimator=best_estimator,
                cv_results=cv_summary,
                validation_metrics=validation_metrics,
                best_params=search.best_params_,
                search=search,
            )
        )
        LOGGER.info(
            '%s | CV F1 macro=%.4f | VAL F1 macro=%.4f',
            spec.name,
            cv_summary['cv_f1_macro_mean'],
            validation_metrics['f1_macro'],
        )

    results_table = build_results_table(candidate_results).sort_values(
        by=['val_f1_macro', 'cv_f1_macro_mean'], ascending=False
    )
    selected = select_best_result(candidate_results)
    LOGGER.info('Modèle sélectionné: %s', selected.name)

    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val])
    final_model = retrain_final_model(selected, X_trainval, y_trainval)

    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba, class_names)
    LOGGER.info('TEST | Accuracy=%.4f | F1 macro=%.4f | AUC=%.4f', test_metrics['accuracy'], test_metrics['f1_macro'], test_metrics['roc_auc_ovr_weighted'])

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / 'best_model.joblib'
    summary_path = artifact_dir / 'training_summary.json'
    candidates_path = artifact_dir / 'candidate_results.csv'
    confusion_matrix_path = artifact_dir / 'confusion_matrix_test.png'
    feature_importance_path = artifact_dir / 'feature_importance.png'

    artifact = {
        'model_name': selected.name,
        'pipeline': final_model,
        'label_encoder': label_encoder,
        'feature_columns': FEATURE_COLUMNS,
        'numeric_columns': NUMERIC_COLUMNS,
        'categorical_columns': CATEGORICAL_COLUMNS,
        'class_names': class_names,
        'selected_cv_results': selected.cv_results,
        'selected_best_params': selected.best_params,
        'validation_metrics': selected.validation_metrics,
        'test_metrics': test_metrics,
        'train_rows': int(len(train_df)),
        'val_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
    }
    summary_payload = {
        'model_name': selected.name,
        'feature_columns': FEATURE_COLUMNS,
        'numeric_columns': NUMERIC_COLUMNS,
        'categorical_columns': CATEGORICAL_COLUMNS,
        'class_names': class_names,
        'selected_cv_results': selected.cv_results,
        'selected_best_params': selected.best_params,
        'validation_metrics': selected.validation_metrics,
        'test_metrics': test_metrics,
        'train_rows': int(len(train_df)),
        'val_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
    }
    joblib.dump(artifact, model_path)
    results_table.to_csv(candidates_path, index=False, encoding='utf-8')
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding='utf-8')

    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, confusion_matrix_path, title='Matrice de confusion - test')
    plot_feature_importance(final_model, feature_importance_path)

    pred_frame = X_test.copy()
    pred_frame['y_true'] = label_encoder.inverse_transform(y_test)
    pred_frame['y_pred'] = label_encoder.inverse_transform(y_test_pred)
    pred_frame.to_csv(artifact_dir / 'test_predictions.csv', index=False, encoding='utf-8')

    LOGGER.info('Artifacts enregistrés dans %s', artifact_dir)
    LOGGER.info('Tableau comparatif enregistré: %s', candidates_path)

    return TrainingArtifacts(
        selected_model_name=selected.name,
        model_path=model_path,
        summary_path=summary_path,
        candidates_path=candidates_path,
        confusion_matrix_path=confusion_matrix_path,
        feature_importance_path=feature_importance_path,
    )


def load_model_artifact(model_path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError('Le fichier modèle ne contient pas un artefact attendu.')
    return artifact


def prepare_prediction_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    elif isinstance(payload, list):
        frame = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        frame = pd.DataFrame([payload])
    else:
        raise TypeError('Payload de prédiction non supporté.')

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f'Colonnes manquantes pour la prédiction: {missing}')
    return frame[FEATURE_COLUMNS].copy()


def predict_payload(model_path: str | Path, payload: Any) -> pd.DataFrame:
    artifact = load_model_artifact(model_path)
    pipeline: Pipeline = artifact['pipeline']
    label_encoder: LabelEncoder = artifact['label_encoder']
    class_names: list[str] = artifact['class_names']

    frame = prepare_prediction_frame(payload)
    proba = pipeline.predict_proba(frame)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)

    result = pd.DataFrame({'prediction': pred_labels})
    for index, class_name in enumerate(class_names):
        result[f'proba_{class_name}'] = proba[:, index]
    return result


def cli_train() -> None:
    parser = argparse.ArgumentParser(description='Entraînement M2 ObRail')
    parser.add_argument('--data', default='data/obrail_features.csv', help='Chemin vers obrail_features.csv')
    parser.add_argument('--artifact-dir', default='artifacts/member2', help='Répertoire de sortie des artefacts')
    parser.add_argument('--cv-splits', type=int, default=5, help='Nombre de folds de validation croisée')
    parser.add_argument('--log-level', default='INFO', help='Niveau de log')
    args = parser.parse_args()

    configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    artifacts = run_training_pipeline(args.data, args.artifact_dir, args.cv_splits)
    LOGGER.info('Modèle final: %s', artifacts.model_path)
    LOGGER.info('Synthèse: %s', artifacts.summary_path)


if __name__ == '__main__':
    cli_train()

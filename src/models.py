import logging
import warnings
from dataclasses import dataclass
from typing import Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features import build_preprocessor

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

warnings.filterwarnings('ignore', message='.*X does not have valid feature names.*')
LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    param_space: dict[str, Any]
    search_kind: str = 'random'
    n_iter: int = 20

def build_model_specs(n_classes: int) -> list[ModelSpec]:
    preprocessor = build_preprocessor()
    specs: list[ModelSpec] = [
        ModelSpec(
            name='logistic_regression',
            estimator=Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
            param_space={'model__C': [0.1, 0.3, 1.0, 3.0, 10.0], 'model__solver': ['lbfgs']},
            search_kind='grid',
        ),
        ModelSpec(
            name='random_forest',
            estimator=Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
            param_space={'model__n_estimators': [300, 500, 800], 'model__max_depth': [None, 10, 20, 30], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4]},
            search_kind='random',
            n_iter=20,
        ),
    ]
    if xgb is not None:
        specs.append(ModelSpec(
            name='xgboost',
            estimator=Pipeline([('preprocessor', preprocessor), ('model', xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss', tree_method='hist', random_state=RANDOM_STATE, n_jobs=-1))]),
            param_space={'model__n_estimators': [200, 400, 600], 'model__max_depth': [3, 5, 7, 9], 'model__learning_rate': [0.01, 0.05, 0.1]},
            search_kind='random',
            n_iter=20,
        ))
    else:
        LOGGER.warning('XGBoost absent.')

    if lgb is not None:
        specs.append(ModelSpec(
            name='lightgbm',
            estimator=Pipeline([('preprocessor', preprocessor), ('model', lgb.LGBMClassifier(objective='multiclass', num_class=n_classes, random_state=RANDOM_STATE, n_jobs=1, verbose=-1))]),
            param_space={'model__n_estimators': [200, 400, 600], 'model__num_leaves': [15, 31, 63], 'model__learning_rate': [0.01, 0.05, 0.1]},
            search_kind='random',
            n_iter=20,
        ))
    else:
        LOGGER.warning('LightGBM absent.')

    return specs

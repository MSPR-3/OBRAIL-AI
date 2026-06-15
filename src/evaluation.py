import logging
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)

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
        'classification_report': classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0, output_dict=True),
    }

def get_feature_names(best_estimator: Pipeline) -> list[str]:
    preprocessor = best_estimator.named_steps['preprocessor']
    return [str(name) for name in preprocessor.get_feature_names_out()]

def plot_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: Path, title: str) -> None:
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
    if hasattr(model, 'coef_'): importances = np.abs(np.asarray(model.coef_)).mean(axis=0)
    elif hasattr(model, 'feature_importances_'): importances = np.asarray(model.feature_importances_)
    else: return
    ranking = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False).head(top_n).iloc[::-1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(ranking) + 2)))
    ax.barh(ranking['feature'], ranking['importance'], color='#356AE6')
    ax.set_title('Top features du modèle final')
    ax.set_xlabel('Importance')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

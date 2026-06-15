﻿from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS

def load_artifact(model_path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError('Le fichier modèle ne contient pas un artefact attendu.')
    required_keys = {'pipeline', 'label_encoder', 'class_names'}
    missing = required_keys.difference(artifact)
    if missing:
        raise ValueError(f'Artefact incomplet: {sorted(missing)}')
    return artifact

def read_payload(payload_path: str | Path) -> pd.DataFrame:
    path = Path(payload_path)
    if not path.exists():
        raise FileNotFoundError(f'Payload introuvable: {path}')

    suffix = path.suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(path)
    elif suffix == '.json':
        content = json.loads(path.read_text(encoding='utf-8'))
        frame = pd.DataFrame(content if isinstance(content, list) else [content])
    else:
        raise ValueError('Le payload doit être un fichier CSV ou JSON.')

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f'Colonnes manquantes pour la prédiction: {missing}')
    return frame[FEATURE_COLUMNS].copy()

def predict_dataframe(model_path: str | Path, frame: pd.DataFrame) -> pd.DataFrame:
    artifact = load_artifact(model_path)
    pipeline = artifact['pipeline']
    label_encoder = artifact['label_encoder']
    class_names = artifact['class_names']

    proba = pipeline.predict_proba(frame[FEATURE_COLUMNS])
    pred_idx = np.argmax(proba, axis=1)
    prediction = label_encoder.inverse_transform(pred_idx)

    result = pd.DataFrame({'prediction': prediction})
    for index, class_name in enumerate(class_names):
        result[f'proba_{class_name}'] = proba[:, index]
    return result

def cli() -> None:
    parser = argparse.ArgumentParser(description='Prédiction ObRail M2')
    parser.add_argument('--model', default='artifacts/member2/best_model.joblib', help='Chemin vers le modèle joblib')
    parser.add_argument('--payload', required=True, help='Fichier CSV ou JSON contenant les variables d\'entrée')
    parser.add_argument('--output', default='', help='Fichier de sortie CSV optionnel')
    args = parser.parse_args()

    frame = read_payload(args.payload)
    result = predict_dataframe(args.model, frame)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding='utf-8')
    else:
        print(result.to_string(index=False))

if __name__ == '__main__':
    cli()

# Membre 2 — pipeline classique de classification

## Ce que fait le projet

Le module [member2_ml.py](member2_ml.py) entraîne plusieurs modèles classiques sur la cible `classe_substitution` livrée par M1, en respectant le périmètre anti-fuite défini dans `docs/01_business.md`.

## Variables utilisées

Les variables explicatives retenues sont:
- `duree_minutes`
- `emission_co2_kg`
- `heure_decimale`
- `is_nuit`
- `is_transfrontalier`
- `code_pays_dep`
- `code_pays_arr`

Les colonnes `distance_km`, `vitesse_kmh`, `split`, `split_classif`, `id_trajet` et `classe_substitution` ne sont jamais utilisées comme features.

## Modèles candidats

- Régression logistique: baseline interprétable.
- RandomForest: référence robuste pour données tabulaires.
- XGBoost: candidat fort pour la performance sur données structurées.
- LightGBM: alternative performante et rapide pour le tabulaire.

## Validation

L'entraînement utilise une validation croisée stratifiée sur le split `train`, puis une validation explicite sur `val`, et un dernier test sur `test`.

Métriques prioritaires:
- F1 macro
- matrice de confusion
- accuracy
- AUC-ROC multiclass

## Résultats générés

Les artefacts sont écrits dans `artifacts/member2/`:
- `best_model.joblib`
- `training_summary.json`
- `candidate_results.csv`
- `confusion_matrix_test.png`
- `feature_importance.png`
- `test_predictions.csv`

## Ré-entraînement

```powershell
python member2_ml.py --data data/obrail_features.csv --artifact-dir artifacts/member2 --cv-splits 5
```

## Prédiction

```powershell
python predict.py --model artifacts/member2/best_model.joblib --payload sample.csv --output predictions.csv
```

`sample.csv` doit contenir les mêmes colonnes de features que le module attend.

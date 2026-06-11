# Rapport Détaillé du Projet IA - MSPR ObRail

## 1. Contexte et Objectifs du Projet

Le projet **MSPR ObRail** (Bloc E6.2) a pour objectif d'intégrer une composante d'Intelligence Artificielle à une plateforme existante d'analyse du transport ferroviaire (composée de 52 314 services). 

L'enjeu métier principal s'inscrit dans la dynamique de **décarbonation et du report modal** (Green Deal). Le projet se divise en deux axes d'analyse :
1. **Axe Non-Supervisé (Clustering) :** Découverte automatique de familles de dessertes ferroviaires pour caractériser l'offre sans a priori.
2. **Axe Supervisé (Classification) :** Prédiction de la pertinence d'une substitution "Avion vers Train" sur un trajet donné, basée sur des caractéristiques d'exploitation (durée, horaires, transfrontalier, etc.) tout en évitant les fuites de données (data leakage) via l'exclusion de la distance.

---

## 2. Architecture et Arborescence du Répertoire

Le projet suit les standards de l'ingénierie logicielle et du Machine Learning (MLOps). Voici la structure logique :

```text
mspr-obrail-ia/
├── artifacts/          # (Généré) Modèles entraînés, métriques et graphiques
├── data/               # Données brutes et nettoyées
├── docs/               # Documentation métier et rapports
├── notebooks/          # Exploration de données (EDA) et scripts de préparation
├── README.md           # Point d'entrée global du projet
├── member2_ml_README.md# Point d'entrée spécifique au pipeline ML
├── member2_ml.py       # Script d'entraînement des modèles de classification
├── predict.py          # Script d'inférence (prédiction sur de nouvelles données)
└── sample.json         # Fichier de test pour l'inférence
```

---

## 3. Rôle Détaillé de Chaque Fichier et Dossier

### 3.1 Dossier `docs/` (Cadrage et Spécifications)
- **`01_business.md`** : C'est la pierre angulaire du projet. Rédigé par le Data Analyst (Membre 1), ce document justifie scientifiquement les choix d'IA. Il explique pourquoi la cible de substitution a été synthétisée, définit les règles de gestion (ex: < 300km = non pertinent), liste les variables à utiliser et surtout, pose les garde-fous pour éviter la "circularité" (exclure la distance et la vitesse de l'entraînement).
- **`02_rapport_projet.md`** : Ce présent document, résumant l'architecture globale.

### 3.2 Dossier `notebooks/` (Exploration et Préparation)
- **`01_eda.ipynb`** : (*Mentionné dans le README*) Notebook d'Analyse Exploratoire des Données. Il a permis de découvrir les valeurs manquantes et de comprendre la distribution du réseau.
- **`_build_prep.py`** : Script Python qui génère dynamiquement le notebook de préparation. Il documente le processus de nettoyage, la suppression de la variable redondante (le CO2 est écarté car c'est une transformation linéaire de la durée) et la standardisation.
- **`02_preparation.ipynb`** : Le résultat du script précédent. C'est ici que les données brutes sont transformées et divisées en `train` / `val` / `test` (70/15/15) pour figer un protocole d'évaluation reproductible.
- **`02b_cible_substitution.ipynb`** : (*Mentionné dans le README*) Construit la cible supervisée à 3 classes (`non_pertinent`, `substitution_difficile`, `substitution_possible`) et assure la stratification.

### 3.3 Dossier `data/` (Jeux de données)
- **`obrail_trajets.csv`** : L'extraction brute depuis la base de données PostgreSQL/PostGIS.
- **`obrail_features.csv`** : Le livrable "contrat" de l'étape de préparation. C'est un dataset propre, encodé et prêt à être consommé par les modèles d'IA.

### 3.4 Scripts à la racine (Pipeline ML et Inférence)
- **`member2_ml.py`** : Le cœur de l'Intelligence Artificielle supervisée. 
  - **Rôle :** Il entraîne et compare quatre algorithmes majeurs (Régression Logistique, Random Forest, XGBoost, LightGBM). 
  - **Mécanique :** Il utilise des `Pipeline` et `ColumnTransformer` scikit-learn pour garantir zéro fuite de données lors de l'imputation et du scaling. Il effectue une recherche d'hyperparamètres (Grid/Randomized Search) avec validation croisée, sélectionne le meilleur modèle en se basant sur le score F1-Macro, le ré-entraîne, et le sauvegarde.
- **`predict.py`** : L'interface d'inférence de l'IA.
  - **Rôle :** Il permet à un utilisateur (ou une API) de soumettre de nouvelles données (CSV ou JSON) au modèle entraîné pour obtenir des prédictions.
  - **Mécanique :** Il charge le fichier `best_model.joblib`, valide la présence des colonnes requises, applique les transformations mathématiques automatiquement via le pipeline sauvegardé, et restitue les classes prédites accompagnées de leurs probabilités associées (niveaux de confiance).
- **`sample.json`** : Fichier de test contenant deux exemples concrets (un trajet régional de jour et un long trajet transfrontalier de nuit) servant à valider le bon fonctionnement de `predict.py`.

### 3.5 Dossier `artifacts/member2/` (Résultats d'Entraînement)
Ce dossier est généré automatiquement par `member2_ml.py`. Il contient la "mémoire" du modèle :
- **`best_model.joblib`** : L'artefact binaire contenant le modèle gagnant et son pipeline de transformation de données.
- **`training_summary.json`** : Un résumé technique (métriques, hyperparamètres gagnants) de la phase d'apprentissage.
- **`candidate_results.csv`** : Le tableau comparatif prouvant quel algorithme a été le plus performant.
- **`confusion_matrix_test.png`** : Évaluation visuelle des erreurs de classification sur le jeu de test.
- **`feature_importance.png`** : Explicabilité du modèle, montrant quelles variables ont le plus influencé les décisions de l'algorithme (ex: `duree_minutes` ou `is_nuit`).

---

## 4. Le Workflow Opérationnel (Cycle de Vie)

1. **Acquisition & Analyse (M1) :** Les données SQL deviennent `obrail_trajets.csv` et sont explorées.
2. **Préparation (M1) :** `_build_prep.py` génère un pipeline de nettoyage produisant `obrail_features.csv` de manière reproductible.
3. **Apprentissage (M2) :** `member2_ml.py` charge ces features, entraîne plusieurs algorithmes, et sauvegarde le meilleur dans le dossier `artifacts/`.
4. **Déploiement / Utilisation :** `predict.py` lit `best_model.joblib` et est prêt à ingérer de nouveaux flux JSON/CSV pour la mise en production.
# MSPR ObRail — Solution IA (Bloc E6.2)

Modèle prédictif/analytique appliqué au référentiel de trajets ferroviaires **ObRail**,
construit lors de la MSPR Industrialisation (base PostgreSQL/PostGIS alimentée par ETL,
exposée via une API FastAPI).

## Périmètre — Membre 1 (Data Analyst / ML Lead)

Couvre les **étapes 1 à 3** du cycle projet :

| Étape | Intitulé | Livrable | État |
|------|----------|----------|------|
| 1 | Business | Spécifications fonctionnelles + justification de l'axe IA | `docs/01_business.md` |
| 2 | Data Acquisition / Analyse | Notebook EDA + tableau des variables retenues | `notebooks/01_eda.ipynb` |
| 3 | Data Préparation | Dataset nettoyé + features, livré à M2/M3 | `notebooks/02_preparation.ipynb` |

## Structure

```
mspr-obrail-ia/
├── data/
│   ├── obrail_trajets.csv        # dataset de travail (52 314 services, extrait de la BDD)
│   └── obrail_features.csv        # sortie de l'étape 3 (généré)
├── notebooks/
│   ├── 01_eda.ipynb              # étape 2 — analyse exploratoire
│   └── 02_preparation.ipynb      # étape 3 — préparation
├── docs/
│   └── 01_business.md            # étape 1 — cadrage métier
├── figures/                       # graphiques générés (reproductibles)
├── requirements.txt
└── README.md
```

## Installation (Windows / PowerShell)

> ⚠️ Le venv est volontairement créé **hors de OneDrive** : un dossier synchronisé
> verrouille les fichiers et corrompt l'environnement Python.

```powershell
python -m venv C:\Users\<user>\.venvs\mspr-obrail
C:\Users\<user>\.venvs\mspr-obrail\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Puis dans VS Code : **Select Kernel** → choisir l'interpréteur `mspr-obrail`, puis **Run All**.

## Données

- Source : dépôt `MSPR-3/OBRAIL-BDD` (`db/init_obrail_db.sql`), table `trajet` jointe à
  `gare` / `ligne` / `operateur`.
- `distance_km` recalculée en distance haversine à partir des coordonnées des gares.
- `type_train` (jour/nuit) dérivé de l'heure de départ.
- **Limites connues** (cf. EDA) : `distance_km` manquante ~55 %, jointure `operateur`
  cassée (~100 % nulle), gares manquantes ~47 %, quelques CO₂ négatifs / durées nulles.

## Réentraînement automatique depuis la base (`train_from_db.py`)

En production le modèle est réentraîné **directement depuis PostgreSQL**, sans
dump CSV ni notebook. Le script `train_from_db.py` :

1. **change-check** — ne réentraîne que si un nouvel import a eu lieu
   (`MAX(historique_import.date_import)` > date du dernier entraînement stockée
   dans `model_artifact`). `--force` ignore ce contrôle.
2. **extraction + features** — `src/db_extract.py` rejoue en code la chaîne des
   notebooks (`02_preparation` + `02b`) : jointure `trajet ⋈ gare`, distance
   haversine, `is_nuit` (heure ≥ 22 ou < 6), cible `classe_substitution` par
   bande de distance, split stratifié 70/15/15.
3. **entraînement** — `src/train.run_training_pipeline(df=...)` (mêmes modèles
   et métriques que l'entraînement CSV).
4. **stockage** — l'artefact joblib est écrit (octets) dans la table
   `model_artifact` de la base ; l'API le recharge à chaud (pas de volume
   partagé Railway).

```bash
export DATABASE_URL=postgresql://obrail_user:...@host:5432/obrail
python train_from_db.py            # réentraîne si nouvelles données
python train_from_db.py --force    # réentraîne dans tous les cas
```

Déploiement : service **cron** Railway (`Dockerfile` + `railway.json`,
`requirements-train.txt` allégé) dans l'environnement `obrail-test`.

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

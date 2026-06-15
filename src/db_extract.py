"""Extraction des trajets depuis la base ObRail et construction du jeu de features.

Reproduit en code la chaîne interactive des notebooks
``02_preparation.ipynb`` + ``02b_cible_substitution.ipynb`` afin que le modèle
puisse être réentraîné directement à partir de la base PostgreSQL, sans dump CSV
manuel.

Sortie : un DataFrame contenant exactement les colonnes attendues par
``src.data.validate_dataset`` / ``src.train.run_training_pipeline`` :
``FEATURE_COLUMNS`` + ``classe_substitution`` + ``split_classif``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLASS_ORDER,
    FEATURE_COLUMNS,
    NON_LABELLED,
    RANDOM_STATE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

LOGGER = logging.getLogger(__name__)

# Seuils de distance (km) -> bande de substitution avion->train.
# Justifiés dans docs/01_business.md §8 (règle ~3 h, loi 2023 vols intérieurs).
SEUIL_BAS, SEUIL_HAUT = 300, 500

# Requête d'extraction : trajet joint aux gares départ/arrivée pour récupérer le
# pays et les coordonnées (servant au calcul de distance haversine).
# ligne / operateur ne sont pas joints : ils n'alimentent aucune feature du
# modèle (la jointure operateur est ~100 % nulle, cf. EDA).
EXTRACT_SQL = """
    SELECT
        t.id_trajet,
        t.duree_minutes,
        t.emission_co2_kg,
        t.heure_depart,
        gd.code_pays AS code_pays_dep,
        ga.code_pays AS code_pays_arr,
        gd.latitude  AS lat_dep,
        gd.longitude AS lon_dep,
        ga.latitude  AS lat_arr,
        ga.longitude AS lon_arr
    FROM public.trajet t
    LEFT JOIN public.gare gd ON t.id_gare_depart  = gd.id_gare
    LEFT JOIN public.gare ga ON t.id_gare_arrivee = ga.id_gare
"""


def load_trajets_from_db(engine: Any) -> pd.DataFrame:
    """Charge les trajets bruts (+ pays/coordonnées des gares) depuis la base."""
    df = pd.read_sql(EXTRACT_SQL, engine)
    LOGGER.info("Trajets chargés depuis la base : %s lignes", len(df))
    return df


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Distance haversine (km) vectorisée ; NaN si une coordonnée manque."""
    r = 6371.0  # rayon terrestre moyen (km)
    lat1, lon1, lat2, lon2 = (
        pd.to_numeric(lat1, errors="coerce"),
        pd.to_numeric(lon1, errors="coerce"),
        pd.to_numeric(lat2, errors="coerce"),
        pd.to_numeric(lon2, errors="coerce"),
    )
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def _heure_decimale(serie: pd.Series) -> pd.Series:
    """Heure de départ -> décimale (hh + mm/60). Accepte time/str/datetime."""
    s = serie.astype(str).str.slice(0, 8)  # "HH:MM:SS"
    h = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
    return h.dt.hour + h.dt.minute / 60.0


def _assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Découpage stratifié 70/15/15 sur le sous-ensemble étiqueté (seed fixée).

    Repli sur un découpage non stratifié si une classe est trop rare pour être
    stratifiée (cas possible sur de petits lots de données fraîches).
    """
    df[SPLIT_COLUMN] = NON_LABELLED
    labelled = df[df[TARGET_COLUMN].notna()]
    if labelled.empty:
        raise ValueError(
            "Aucune ligne étiquetable : distance_km absente partout "
            "(coordonnées des gares manquantes ?)."
        )

    y = labelled[TARGET_COLUMN].astype(str)
    stratify_full: Any = y if y.value_counts().min() >= 2 else None
    try:
        train_idx, temp_idx = train_test_split(
            labelled.index, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify_full
        )
        y_temp = y.loc[temp_idx]
        stratify_temp: Any = y_temp if y_temp.value_counts().min() >= 2 else None
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=stratify_temp
        )
    except ValueError as exc:
        LOGGER.warning("Stratification impossible (%s) -> découpage aléatoire.", exc)
        train_idx, temp_idx = train_test_split(
            labelled.index, test_size=0.30, random_state=RANDOM_STATE
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=RANDOM_STATE
        )

    df.loc[train_idx, SPLIT_COLUMN] = TRAIN_SPLIT
    df.loc[val_idx, SPLIT_COLUMN] = VAL_SPLIT
    df.loc[test_idx, SPLIT_COLUMN] = TEST_SPLIT
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage + feature engineering + cible synthétique + split (cf. notebooks)."""
    df = raw.copy()

    # 1. Nettoyage des aberrants (cf. 02_preparation §2)
    df["duree_minutes"] = pd.to_numeric(df["duree_minutes"], errors="coerce")
    df["emission_co2_kg"] = pd.to_numeric(df["emission_co2_kg"], errors="coerce")
    n0 = len(df)
    df = df[(df["duree_minutes"] > 0) & (df["emission_co2_kg"] >= 0)].copy()
    LOGGER.info("Nettoyage aberrants : %s -> %s lignes", n0, len(df))

    # 2. Feature engineering (cf. 02_preparation §4)
    df["heure_decimale"] = _heure_decimale(df["heure_depart"])
    heure = df["heure_decimale"].fillna(-1).astype(int)
    df["is_nuit"] = ((heure >= 22) | (heure < 6)).astype(int)  # règle vérifiée sur les données

    df["distance_km"] = _haversine_km(
        df["lat_dep"], df["lon_dep"], df["lat_arr"], df["lon_arr"]
    )

    deux_pays = df["code_pays_dep"].notna() & df["code_pays_arr"].notna()
    df["is_transfrontalier"] = np.where(
        deux_pays, (df["code_pays_dep"] != df["code_pays_arr"]).astype(float), np.nan
    )

    # 3. Cible synthétique par bande de distance (cf. 02b §2)
    conditions = [
        df["distance_km"] < SEUIL_BAS,
        (df["distance_km"] >= SEUIL_BAS) & (df["distance_km"] < SEUIL_HAUT),
        df["distance_km"] >= SEUIL_HAUT,
    ]
    df[TARGET_COLUMN] = np.select(conditions, CLASS_ORDER, default=None)
    df.loc[df["distance_km"].isna(), TARGET_COLUMN] = np.nan

    n_lab = int(df[TARGET_COLUMN].notna().sum())
    LOGGER.info(
        "Lignes étiquetables (distance présente) : %s / %s (%.1f %%)",
        n_lab, len(df), (n_lab / len(df) * 100) if len(df) else 0,
    )

    # 4. Découpage stratifié (cf. 02b §4)
    df = _assign_split(df)

    cols_out = ["id_trajet", *FEATURE_COLUMNS, "distance_km", TARGET_COLUMN, SPLIT_COLUMN]
    return df[cols_out].copy()


def extract_features_from_db(engine: Any) -> pd.DataFrame:
    """Pipeline complet base -> DataFrame de features prêt pour l'entraînement."""
    return build_features(load_trajets_from_db(engine))

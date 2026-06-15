"""Réentraînement du modèle ObRail à partir de la base PostgreSQL.

Point d'entrée du service "trainer" (cron Railway). À chaque exécution :
  1. change-check : on ne réentraîne que si un nouvel import a eu lieu depuis le
     dernier entraînement (table ``historique_import`` vs ``model_artifact``) ;
  2. extraction des trajets + construction des features et de la cible ;
  3. entraînement (``run_training_pipeline``) ;
  4. stockage de l'artefact (octets joblib) dans la table ``model_artifact``,
     d'où l'API le recharge à chaud.

Usage :
    python train_from_db.py [--force] [--log-level INFO]

Variable d'environnement requise : ``DATABASE_URL``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

import sklearn
from sqlalchemy import create_engine

from src.db_extract import extract_features_from_db
from src.model_store import has_new_data, save_model
from src.train import run_training_pipeline

LOGGER = logging.getLogger("train_from_db")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def normalize_db_url(url: str) -> str:
    """Railway fournit ``postgres://`` ; SQLAlchemy attend ``postgresql+psycopg2://``."""
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description="Réentraînement ObRail depuis la base")
    parser.add_argument("--force", action="store_true", help="Réentraîner même sans nouvel import")
    parser.add_argument("--cv-splits", type=int, default=5, help="Folds de validation croisée")
    parser.add_argument("--log-level", default="INFO", help="Niveau de log")
    args = parser.parse_args()

    configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))

    raw_url = os.getenv("DATABASE_URL")
    if not raw_url:
        LOGGER.error("DATABASE_URL non défini.")
        return 2
    engine = create_engine(normalize_db_url(raw_url), pool_pre_ping=True)

    # 1. change-check
    retrain, watermark, last_trained = has_new_data(engine)
    LOGGER.info(
        "change-check : dernier import=%s | dernier entraînement=%s | réentraîner=%s",
        watermark, last_trained, retrain,
    )
    if not retrain and not args.force:
        LOGGER.info("Aucune nouvelle donnée : entraînement ignoré.")
        return 0

    # 2. extraction + features
    df = extract_features_from_db(engine)

    # 3. entraînement (artefacts disque dans un dossier temporaire éphémère)
    with tempfile.TemporaryDirectory(prefix="obrail_artifacts_") as tmp:
        artifacts = run_training_pipeline(df=df, artifact_dir=tmp, cv_splits=args.cv_splits)

    tm = artifacts.test_metrics or {}
    metrics_compact = {
        "accuracy": tm.get("accuracy"),
        "f1_macro": tm.get("f1_macro"),
        "roc_auc_ovr_weighted": tm.get("roc_auc_ovr_weighted"),
        "confusion_matrix": tm.get("confusion_matrix"),
    }
    LOGGER.info(
        "Modèle '%s' | accuracy=%.4f | f1_macro=%.4f",
        artifacts.selected_model_name,
        metrics_compact["accuracy"] or 0.0,
        metrics_compact["f1_macro"] or 0.0,
    )

    # 4. stockage en base
    id_model = save_model(
        engine,
        artifact_bytes=artifacts.model_bytes,
        model_name=artifacts.selected_model_name,
        metrics=metrics_compact,
        trained_on_import_date=watermark,
        n_rows_train=artifacts.n_rows_train,
        sklearn_version=sklearn.__version__,
    )
    LOGGER.info("Réentraînement terminé. Nouveau modèle actif : id_model=%s", id_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())

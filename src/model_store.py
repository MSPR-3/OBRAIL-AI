"""Stockage des artefacts de modèle dans PostgreSQL (table ``model_artifact``).

Permet au service d'entraînement (cron) et à l'API de s'échanger le modèle sans
volume partagé : le trainer écrit les octets joblib ici, l'API lit la dernière
ligne active. Contient aussi la logique de *change-check* (déclenchement du
réentraînement uniquement si de nouvelles données ont été importées).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import text

LOGGER = logging.getLogger(__name__)

# Statuts d'import considérés comme "données disponibles" pour le change-check.
IMPORT_OK_STATUTS = ("succès", "succes", "partiel")


def read_db_import_high_watermark(engine: Any) -> Optional[datetime]:
    """Date du dernier import réussi/partiel (signal "nouvelles données")."""
    sql = text(
        """
        SELECT MAX(date_import) AS hw
        FROM public.historique_import
        WHERE LOWER(statut) IN :statuts
        """
    ).bindparams(statuts=tuple(IMPORT_OK_STATUTS))
    with engine.connect() as conn:
        return conn.execute(sql).scalar()


def read_last_trained_import_date(engine: Any) -> Optional[datetime]:
    """Date d'import sur laquelle le dernier modèle a été entraîné (None si aucun)."""
    sql = text("SELECT MAX(trained_on_import_date) AS d FROM public.model_artifact")
    with engine.connect() as conn:
        return conn.execute(sql).scalar()


def has_new_data(engine: Any) -> tuple[bool, Optional[datetime], Optional[datetime]]:
    """Retourne (réentraîner ?, watermark base, date du dernier entraînement)."""
    hw = read_db_import_high_watermark(engine)
    last = read_last_trained_import_date(engine)
    if hw is None:
        return False, hw, last          # aucune donnée importée
    if last is None:
        return True, hw, last           # jamais entraîné
    return hw > last, hw, last


def save_model(
    engine: Any,
    artifact_bytes: bytes,
    model_name: str,
    metrics: dict[str, Any],
    trained_on_import_date: Optional[datetime],
    n_rows_train: int,
    sklearn_version: str,
) -> int:
    """Désactive les anciens modèles puis insère le nouveau (actif). Renvoie id_model."""
    with engine.begin() as conn:  # transaction
        conn.execute(text("UPDATE public.model_artifact SET is_active = FALSE WHERE is_active"))
        new_id = conn.execute(
            text(
                """
                INSERT INTO public.model_artifact
                    (model_name, sklearn_version, trained_on_import_date,
                     n_rows_train, metrics, artifact, is_active)
                VALUES
                    (:model_name, :skl, :trained_on, :n_rows,
                     CAST(:metrics AS jsonb), :artifact, TRUE)
                RETURNING id_model
                """
            ),
            {
                "model_name": model_name,
                "skl": sklearn_version,
                "trained_on": trained_on_import_date,
                "n_rows": n_rows_train,
                "metrics": json.dumps(metrics, ensure_ascii=False),
                "artifact": artifact_bytes,
            },
        ).scalar_one()
    LOGGER.info("Modèle '%s' enregistré en base (id_model=%s).", model_name, new_id)
    return int(new_id)

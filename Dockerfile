# Service "trainer" ObRail — réentraîne le modèle depuis la base et stocke
# l'artefact dans PostgreSQL. Conçu pour tourner en cron sur Railway.
FROM python:3.12-slim

# libgomp1 : requis par xgboost / lightgbm
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
COPY train_from_db.py train_model.py ./

# Exécution unique (le scheduler Railway relance le conteneur selon le cron).
CMD ["python", "train_from_db.py"]

import argparse
import logging
from src.train import run_training_pipeline

LOGGER = logging.getLogger('train_model')

def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(message)s')

def cli_train() -> None:
    parser = argparse.ArgumentParser(description='Entraînement M2 ObRail')
    parser.add_argument('--data', default='data/obrail_features.csv', help='Chemin vers obrail_features.csv')
    parser.add_argument('--artifact-dir', default='artifacts/member2', help='Répertoire de sortie des artefacts')
    parser.add_argument('--cv-splits', type=int, default=5, help='Nombre de folds de validation croisée')
    parser.add_argument('--log-level', default='INFO', help='Niveau de log')
    args = parser.parse_args()

    configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    artifacts = run_training_pipeline(args.data, args.artifact_dir, args.cv_splits)
    LOGGER.info('Modèle final: %s', artifacts.model_path)
    LOGGER.info('Synthèse: %s', artifacts.summary_path)

if __name__ == '__main__':
    cli_train()
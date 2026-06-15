import logging
from pathlib import Path
import pandas as pd

from src.config import REQUIRED_COLUMNS, SPLIT_COLUMN, TARGET_COLUMN, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

LOGGER = logging.getLogger(__name__)

def resolve_dataset_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.exists():
        return path
    candidates = [Path('data/obrail_features.csv'), Path('../data/obrail_features.csv')]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Impossible de trouver le fichier de données: {data_path}')

def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = resolve_dataset_path(data_path)
    df = pd.read_csv(path, low_memory=False)
    LOGGER.info('Dataset chargé: %s (%s lignes, %s colonnes)', path, len(df), len(df.columns))
    return df

def validate_dataset(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Colonnes manquantes dans le dataset: {sorted(missing)}')
    labelled = df[df[SPLIT_COLUMN].isin({TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT})]
    if labelled.empty:
        raise ValueError('Aucune ligne étiquetée trouvée pour la classification.')

def split_labelled_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = df[df[SPLIT_COLUMN].isin({TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT})].copy()
    train_df = labelled[labelled[SPLIT_COLUMN] == TRAIN_SPLIT].copy()
    val_df = labelled[labelled[SPLIT_COLUMN] == VAL_SPLIT].copy()
    test_df = labelled[labelled[SPLIT_COLUMN] == TEST_SPLIT].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Les splits train/val/test doivent tous être non vides.')
    return train_df, val_df, test_df

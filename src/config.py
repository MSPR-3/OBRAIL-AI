RANDOM_STATE = 42
TARGET_COLUMN = 'classe_substitution'
SPLIT_COLUMN = 'split_classif'
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'
NON_LABELLED = 'non_etiquete'
CLASS_ORDER = ['non_pertinent', 'substitution_difficile', 'substitution_possible']

FEATURE_COLUMNS = [
    'duree_minutes',
    'heure_decimale',
    'is_nuit',
    'is_transfrontalier',
    'code_pays_dep',
    'code_pays_arr',
]
NUMERIC_COLUMNS = [
    'duree_minutes',
    'heure_decimale',
    'is_nuit',
    'is_transfrontalier',
]
CATEGORICAL_COLUMNS = ['code_pays_dep', 'code_pays_arr']
REQUIRED_COLUMNS = set(FEATURE_COLUMNS + [TARGET_COLUMN, SPLIT_COLUMN])

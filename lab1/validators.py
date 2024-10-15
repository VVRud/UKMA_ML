import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def simple_validation(X, y):
    ''' Splits data into train and validation returns indexes.'''
    yield train_test_split(np.arange(len(X)), test_size=0.2)


def stratified_simple_validation(X, y):
    ''' Splits data into train and validation returns indexes.'''
    yield train_test_split(np.arange(len(X)), test_size=0.2, stratify=y)


def k_fold_validation(X, y):
    ''' Splits data into train and validation returns indexes.'''
    kf = KFold(n_splits=5, shuffle=True)
    yield from kf.split(X, y)


def stratified_k_fold_validation(X, y):
    ''' Splits data into train and validation returns indexes.'''
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    yield from kf.split(X, y)


VALIDATORS = {
    'simple': simple_validation,
    'stratified_simple': stratified_simple_validation,
    'k_fold': k_fold_validation,
    'stratified_k_fold': stratified_k_fold_validation
}

def create_validator(validator_name: str):
    validator = VALIDATORS.get(validator_name, None)
    if validator is None:
        raise ValueError(f"Validator {validator_name} not supported.")
    return validator

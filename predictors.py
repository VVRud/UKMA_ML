import numpy as np


def _soft_voting_values(models, X):
    ''' Predicts using soft voting ensemble.'''
    preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        preds[:, i] = model.predict_proba(X)[:, 1]
    return preds


def _hard_voting_values(models, X):
    ''' Predicts using hard voting ensemble.'''
    preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        preds[:, i] = model.predict(X)
    return preds


def voting_predictor(models, X, method, threshold=0.5):
    ''' Predicts using voting ensemble.'''
    if method == 'soft':
        preds = _soft_voting_values(models, X)
    elif method == 'hard':
        preds = _hard_voting_values(models, X)
    else:
        raise ValueError('method should be either `soft` or `hard`')
    return (np.mean(preds, axis=1) > threshold).astype(np.int32)

import numpy as np

from lab1.linear_regression import MyLinearRegression


class MyLogisticRegression(MyLinearRegression):
    def cost(self, target, pred):
        return -np.mean(target * np.log(pred + 1e-9) + (1 - target) * np.log(1 - pred + 1e-9))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _predict(self, x):
        return self.sigmoid(super()._predict(x))

    def predict_proba(self, x):
        values = self._predict(x)
        return np.concatenate([1 - values, values], axis=1)

    def predict(self, x):
        return (self._predict(x).ravel() > 0.5).astype(np.int32)

"""Baselines (neural models are added later by the Model Architect)."""
import numpy as np
from sklearn.linear_model import Ridge


class SeasonalNaive:
    """Tomorrow = today: repeat the last 24 h (needs input_len == output_len)."""
    name = "Seasonal naive (tomorrow = today)"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, -96:, 0]


class LastValue:
    """Repeat the last observed value for all 96 future steps."""
    name = "Last value"

    def __init__(self, output_len=96):
        self.output_len = output_len

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.repeat(X[:, -1, 0:1], self.output_len, axis=1)


class RidgeBaseline:
    """Linear model on the flattened input window."""
    name = "Ridge regression"

    def __init__(self, alpha=10.0, subsample=1):
        self.model = Ridge(alpha=alpha)
        self.subsample = subsample

    def fit(self, X, y):
        Xs, ys = X[::self.subsample], y[::self.subsample]
        self.model.fit(Xs.reshape(len(Xs), -1), ys)
        return self

    def predict(self, X):
        return self.model.predict(X.reshape(len(X), -1))

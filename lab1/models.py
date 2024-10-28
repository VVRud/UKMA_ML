from pathlib import Path

from sklearn.linear_model import LogisticRegression

from lab1.logistic_regression import MyLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


MODELS = {
    "logistic_regression": MyLogisticRegression(log_path=Path('logs') / 'part2', learning_rate=1e-2, num_iterations=500_000, verbose=True),
    "logistic_regression_sklearn": LogisticRegression(n_jobs=-1),
    "knn": KNeighborsClassifier(n_jobs=-1),
    "decision_tree": DecisionTreeClassifier(),
}


def create_model(model_name: str):
    model = MODELS.get(model_name, None)
    if model is None:
        raise ValueError(f"Model {model_name} not supported.")
    return model
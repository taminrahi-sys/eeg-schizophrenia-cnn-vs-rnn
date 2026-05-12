from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np



def cross_validate_model(model, X, y, n_splits=5):

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    accuracies = []

    for train_idx, test_idx in cv.split(X, y):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)
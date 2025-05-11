from random import randint
from time import time

import optuna
from cuml import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from covid import get_covid_19
from mimic import get_mimic_iv_31
from misc import prepare_data_split, objective


def main():
    name = 'MIMIC-IV 3.1'
    print(f'Loading {name} data...')

    data_load_start = time()
    data = get_mimic_iv_31() if name == 'MIMIC-IV 3.1' else get_covid_19()
    X, y = data
    data_load_end = time()

    print(f"Data loaded in {data_load_end - data_load_start:.2f} seconds.")

    # Initial split to simulate train/test
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = prepare_data_split(X, y)

    # Only optimize on the training set
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=2)

    print("Best parameters:", study.best_params)
    print("Best CV accuracy:", study.best_value)

    # Final model with best parameters
    best_model = RandomForestClassifier(**study.best_params, random_state=randint(0, 1000))
    best_model.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = best_model.predict(X_test)
    threshold = 0.3
    y_pred = (y_pred > threshold).astype(int)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()

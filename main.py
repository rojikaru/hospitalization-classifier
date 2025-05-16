from random import randint

import optuna
from cuml.dask.ensemble import RandomForestClassifier as DaskRF
from sklearn.metrics import classification_report, confusion_matrix

from mimic import get_mimic_iv_31
from misc import prepare_data_split, objective


def main():
    name = 'MIMIC-IV 3.1'
    print(f'Loading {name} data...')

    data = get_mimic_iv_31()
    data = data.compute()
    y = data['readmit_30d']
    X = data.drop('readmit_30d', axis=1)

    # Initial split to simulate train/test
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = prepare_data_split(
        X.to_pandas(),
        y.to_pandas(),
    )

    # Only optimize on the training set
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)

    print("Best parameters:", study.best_params)
    print("Best CV accuracy:", study.best_value)

    # Final model with best parameters
    best_model = DaskRF(**study.best_params, random_state=randint(0, 1000))
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

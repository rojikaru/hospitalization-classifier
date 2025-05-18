import gc
from random import randint

import matplotlib.pyplot as plt
import cudf
import optuna
import pandas as pd
from cuml import RandomForestClassifier as cuRF
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix

from mimic import get_mimic_iv_31
from misc import prepare_data_split, objective, ToCuDFTransformer


def main():
    name = 'MIMIC-IV 3.1'
    print(f'Loading {name} data...')

    X = get_mimic_iv_31()
    y = X.pop('readmit_30d')

    # Initial split to simulate train/test
    print(f'Data loaded. Preparing data for training...')
    X_train, X_test, y_train, y_test = prepare_data_split(X, y)

    all_cols = X.columns
    del X, y
    gc.collect()

    # Only optimize on the training set
    print("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)

    print("Best parameters:", study.best_params)
    print("Best CV accuracy:", study.best_value)

    # Final model with the best parameters
    best_model = cuRF(
        **study.best_params,
        random_state=randint(0, 1000),
        n_streams=1,
    )

    # Convert data to pandas explicitly for SMOTE compatibility
    X_train_pd = pd.DataFrame(
        X_train.compute().astype('float32')
    ).fillna(0)
    y_train_pd = y_train.compute().astype('int32')

    # Build pipeline with data conversion
    USE_SMOTE = False
    model_pipeline = Pipeline([
        ('smote', SMOTE(random_state=randint(0, 1000))) if USE_SMOTE else ('passthrough', 'passthrough'),
        ('to_cudf', ToCuDFTransformer()),
        ('classifier', best_model)
    ], verbose=True)

    # Train
    model_pipeline.fit(X_train_pd, y_train_pd)

    # Predict
    X_test_pd = pd.DataFrame(
        X_test.compute().astype('float32')
    ).fillna(0)
    y_proba = model_pipeline.predict_proba(cudf.DataFrame(X_test_pd)).to_numpy()[:, 1]
    y_pred = (y_proba > 0.2).astype(int)

    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # https://github.com/rapidsai/cuml/issues/3531
    result = permutation_importance(
        model_pipeline.named_steps['classifier'],
        X_test_pd.values,  # Requires CPU data
        y_test.compute(),
        n_repeats=5,
        random_state=42
    )

    plt.figure().set_figwidth(30)
    plt.figure().set_figheight(30)
    plt.barh(
        all_cols,
        result.importances_mean,
    )
    plt.title('Feature Importance')
    plt.show(savefig='feature_importance.png')


if __name__ == '__main__':
    main()

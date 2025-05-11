from time import time

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

from covid import get_covid_19
from mimic import get_mimic_iv_31
from misc import prepare_data_split, build_preprocessor, build_model_pipeline


def main():
    name = 'MIMIC-IV 3.1'
    print(f'Loading {name} data...')

    data_load_start = time()
    data = get_mimic_iv_31() if name == 'MIMIC-IV 3.1' else get_covid_19()
    X, y = data
    data_load_end = time()

    print(f"Data loaded in {data_load_end - data_load_start:.2f} seconds.")

    train_start = time()

    # 1. Initial split to simulate train/test
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = prepare_data_split(X, y)

    # 2. Build preprocessor and full pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model_pipeline = build_model_pipeline(preprocessor)

    # 3. Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(model_pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

    print("Cross-validation results:")
    for metric in scoring:
        print(f"{metric}: {scores[f'test_{metric}'].mean():.4f} ± {scores[f'test_{metric}'].std():.4f}")

    # 4. Fit final model on full training data
    model_pipeline.fit(X_train, y_train)

    train_end = time()
    print("Model training completed.\n"
          f"Training time: {train_end - train_start:.2f} seconds.")

    # 5. Evaluate on untouched test set
    y_pred = model_pipeline.predict(X_test)
    print("\nTest set evaluation:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    main()

import gc
from random import randint

import cudf
import numpy as np
from cuml import RandomForestClassifier, accuracy_score
from cuml.model_selection import StratifiedKFold
from cuml.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from numba.cuda import current_context
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ])


def prepare_data_split(
    features, labels, test_size=0.2, random_state=42, min_non_missing=1
):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    non_missing_counts = X_train.notnull().sum()
    cols_to_drop = non_missing_counts[non_missing_counts < min_non_missing].index.tolist()
    X_train = X_train.drop(columns=cols_to_drop)
    X_test  = X_test.drop(columns=cols_to_drop)

    numeric_cols = X_train.select_dtypes(include='number').columns
    categorical_cols = X_train.select_dtypes(include='object').columns

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def build_model_pipeline(preprocessor, model=None):
    if model is None:
        model = RandomForestClassifier(
            n_estimators=500,
            n_streams=1,
            random_state=randint(0, 1000)
        )

    return Pipeline([
        ('preprocessing', preprocessor),
        ('smote', SMOTE(random_state=randint(0, 1000))),
        # ('undersample', RandomUnderSampler(random_state=42)),
        ('classifier', model)
    ], verbose=True)

def objective(trial, X, y):
    # Suggest hyperparameters
    search_space = {
        'n_estimators': trial.suggest_int("n_estimators", 50, 100),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5)
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=randint(0, 1000))
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx.get()].fillna(0), X.iloc[val_idx.get()].fillna(0)
        y_train_cv, y_val_cv = y.iloc[train_idx.get()], y.iloc[val_idx.get()]

        model = RandomForestClassifier(
            n_estimators=search_space['n_estimators'],
            n_streams=1,
            max_depth=search_space['max_depth'],
            min_samples_leaf=search_space['min_samples_leaf'],
            random_state=randint(0, 1000),
        )

        X_train_cv = cudf.DataFrame.from_pandas(X_train_cv)
        y_train_cv = cudf.Series(y_train_cv)

        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_val_cv)
        acc = accuracy_score(y_val_cv.to_numpy(), preds)
        scores.append(acc)

        gc.collect()
        current_context().deallocations.clear()

    return np.mean(scores)

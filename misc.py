import gc
from random import randint

import cudf
import numpy as np
import pandas as pd
from cuml import RandomForestClassifier
from dask_ml.model_selection import KFold as StratifiedKFold
from dask_ml.preprocessing import DummyEncoder
from numba.cuda import current_context
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ToCuDFTransformer(BaseEstimator, TransformerMixin):
    """Convert pandas DataFrame to cuDF for GPU steps"""
    def fit(self, X, y=None):
        return self


    def transform(self, X):
        return cudf.DataFrame(X) if isinstance(X, pd.DataFrame) else X

def prepare_data_split(
    features, labels, test_size=0.2, random_state=42
):
    # 1) Extract column‐name lists for downstream preprocessing:
    categorical_cols = features.select_dtypes(include='object').columns.tolist()

    # 2) Dask-ML lazy pipelines:
    #    a) Categorize string columns
    print(f'Categorizing {len(categorical_cols)} categorical columns...')
    features = features.categorize(columns=categorical_cols)
    gc.collect()
    #    b) One-hot encode (🔧 Ensure numeric dtype (needed by cuML))
    print(f'One-hot encoding {len(categorical_cols)} categorical columns...')
    features = DummyEncoder().fit_transform(features).astype("float32")
    gc.collect()

    # 3) Materialize labels to numpy to get stratified indices
    print(f'Materializing labels to numpy array...')
    y_np = labels.compute()
    idx = np.arange(len(y_np))

    # 4) Stratified split of indices
    print(f'Stratifying {len(y_np)} labels into train/test split...')
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np
    )
    del y_np, idx
    gc.collect()

    # 5) Turn features/labels into Dask Arrays (lazy) and index
    print(f'Indexing {len(train_idx)} train and {len(test_idx)} test samples...')
    X_arr = features.to_dask_array(lengths=True)
    y_arr = labels.to_dask_array(lengths=True)

    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    # 6) Return Dask Arrays + plain‐list column names
    return X_train, X_test, y_train, y_test


def objective(trial, X, y):
    gc.collect()
    current_context().deallocations.clear()

    # Suggest hyperparameters
    search_space = {
        'n_estimators': trial.suggest_int("n_estimators", 50, 60),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 5)
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    scores = []

    for train_idx, val_idx in tqdm(cv.split(X, y)):
        X_train_cv = cudf.DataFrame.from_pandas(
            pd.DataFrame(X[train_idx].compute()).fillna(0)
        )
        y_train_cv = cudf.Series(y[train_idx].compute()).fillna(0)

        model = RandomForestClassifier(
            n_estimators=search_space['n_estimators'],
            n_streams=1,
            max_depth=search_space['max_depth'],
            min_samples_leaf=search_space['min_samples_leaf'],
            random_state=randint(0, 1000),
        )

        model.fit(X_train_cv, y_train_cv)

        # Convert validation data [Dask array → NumPy]
        X_val_cv = X[val_idx].compute()
        y_val_cv = y[val_idx].compute()

        preds = model.predict(X_val_cv)
        scores.append(f1_score(
            y_val_cv,
            preds,
            zero_division=0,
            average='weighted'
        ))

        del X_train_cv, X_val_cv, y_train_cv, y_val_cv
        gc.collect()
        current_context().deallocations.clear()

    return np.mean(scores)

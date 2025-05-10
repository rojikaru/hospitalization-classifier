import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def prepare_data_for_training(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        min_non_missing=1
):
    """
    Splits data, applies preprocessing, and resamples training data using SMOTE.

    Args:
        features (pd.DataFrame): The raw feature DataFrame.
        labels (pd.Series): The raw label Series.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        min_non_missing (int): Minimum number of non-null values to keep a column.

    Returns:
        tuple: (X_train_resampled, y_train_resampled, X_test_processed, y_test)
               Processed and resampled training data, processed test data, test labels.
    """
    print("\nPreparing data for training (Splitting, Preprocessing, Resampling)...")

    # 1. Split data into training and testing sets
    # Use stratify to maintain the same proportion of target classes in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # 2. Find columns to drop: those with fewer than min_non_missing non-null values
    non_missing_counts = X_train.notnull().sum()
    cols_to_drop = non_missing_counts[non_missing_counts < min_non_missing].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} empty/near-empty columns:")
        print(cols_to_drop)
        X_train = X_train.drop(columns=cols_to_drop)
        X_test  = X_test.drop(columns=cols_to_drop)

    print(f"X_train shape after split: {X_train.shape}")
    print(f"X_test shape after split: {X_test.shape}")
    print(f"y_train shape after split: {y_train.shape}")
    print(f"y_test shape after split: {y_test.shape}")
    print(f"Original train label distribution:\n{y_train.value_counts(normalize=True)}")

    # 3. Identify column types based on the training data
    # This is safer in case test set has different data characteristics
    numeric_cols = X_train.select_dtypes(include='number').columns
    categorical_cols = X_train.select_dtypes(include='object').columns

    # You might want to explicitly list categorical columns if select_dtypes is unreliable
    # For example: explicit_categorical_cols = ['ethnicity', 'marital_status', 'insurance']
    # Then use this list instead of categorical_cols

    print(f"Identified numeric columns: {list(numeric_cols)}")
    print(f"Identified categorical columns: {list(categorical_cols)}")


    # 4. Define preprocessing pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # Impute mean for numerical NaNs
        ('scaler', StandardScaler()) # Scale numerical features
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute mode for categorical NaNs
        # Add One-Hot Encoding for categorical features
        # handle_unknown='ignore' is good practice for test sets
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines in ColumnTransformer
    # 'remainder'='passthrough' keeps columns not explicitly listed (e.g., potentially boolean if any)
    # 'remainder'='drop' removes columns not explicitly listed
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ], remainder='passthrough')

    # 5. Apply preprocessing
    # Fit the preprocessor ONLY on the training data
    preprocessor.fit(X_train)

    # Transform both training and test data using the fitted preprocessor
    X_train_processed = preprocessor.transform(X_train)
    print(f"X_train_processed shape after preprocessing: {X_train_processed.shape}")
    X_test_processed = preprocessor.transform(X_test)
    print(f"X_test_processed shape after preprocessing: {X_test_processed.shape}")

    # 5. Handle class imbalance using SMOTE
    smote = SMOTE(random_state=random_state)
    # Apply SMOTE ONLY to the processed training data
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    print(f"X_train_resampled shape after SMOTE: {X_train_resampled.shape}")
    print(f"y_train_resampled shape after SMOTE: {y_train_resampled.shape}")
    print(f"Resampled train label distribution:\n{y_train_resampled.value_counts(normalize=True)}")

    # Return processed training data (resampled), processed test data, and test labels
    return X_train_resampled, y_train_resampled, X_test_processed, y_test

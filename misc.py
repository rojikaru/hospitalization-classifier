from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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
        model = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

    return Pipeline([
        ('preprocessing', preprocessor),
        ('smote', SMOTEENN(random_state=42)),
        ('classifier', model)
    ], verbose=True)

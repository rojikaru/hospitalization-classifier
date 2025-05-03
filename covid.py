import pandas as pd

def get_covid_19():
    # Load the dataset
    file_path = 'datasets/Covid Data.csv'
    data = pd.read_csv(file_path)

    # 1. Convert Boolean features (1/2) to 0/1
    boolean_columns = ['INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
                       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU']
    data[boolean_columns] = data[boolean_columns].replace({2: 0})

    # 2. Handle missing values (97 and 99)
    mask = ~data.isin([97, 99]).any(axis=1)
    data = data.loc[mask]

    # 3. Process `CLASIFFICATION_FINAL` column
    data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].replace({1: 1, 2: 1, 3: 1})
    data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].apply(lambda x: 0 if x >= 4 else x)

    # 4. Drop rows with missing or invalid values in `CLASIFFICATION_FINAL`
    data.dropna(subset=['CLASIFFICATION_FINAL'], inplace=True)

    # 5. Drop irrelevant columns
    data.drop(['DATE_DIED'], axis=1, errors='ignore', inplace=True)

    # 6. Encode categorical variables using one-hot encoding
    categorical_columns = ['USMER', 'MEDICAL_UNIT', 'PATIENT_TYPE']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # 7. Validate that the target variable ('CLASIFFICATION_FINAL') is binary
    if not set(data['CLASIFFICATION_FINAL'].unique()).issubset({0, 1}):
        raise ValueError("Target variable 'CLASIFFICATION_FINAL' contains invalid values. Ensure it is binary (0 or 1).")

    # 8. Select specific columns for features
    # columns_to_return = ["AGE", "INTUBED", "PNEUMONIA", "PREGNANT", "DIABETES", "COPD", "ASTHMA"]
    columns_to_return = [
        "SEX",           # Gender of the patient
        "AGE",           # Age of the patient
        "PREGNANT",      # Pregnancy status
        "DIABETES",      # Presence of diabetes
        "COPD",          # Presence of chronic obstructive pulmonary disease
        "ASTHMA",        # Presence of asthma
        "INMSUPR",       # Immunosuppressed status
        "HIPERTENSION",  # Presence of hypertension
        "CARDIOVASCULAR",# Presence of cardiovascular disease
        "OBESITY",       # Obesity status
        "RENAL_CHRONIC", # Presence of chronic renal disease
        "TOBACCO",       # Tobacco use status
        "INTUBED",       # Whether the patient was intubated
        "PNEUMONIA",     # Presence of pneumonia
    ]
    missing_columns = [col for col in columns_to_return if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following columns are not in the dataset: {missing_columns}")

    # 9. Ensure all feature columns and target column are numeric
    retX = data[columns_to_return].apply(pd.to_numeric, errors='coerce')
    retY = pd.to_numeric(data['ICU'], errors='coerce')

    # 10. Drop rows with any remaining invalid values
    retX.dropna(inplace=True)
    retY = retY[retX.index]

    return retX, retY

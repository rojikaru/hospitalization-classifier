import os
from hashlib import md5

import joblib
import pandas as pd


def get_mimic_iv_31(data_path='datasets', cache_path='datasets/cache'):
    """
    Prepares data from MIMIC-IV 3.1 for hospitalization risk prediction.

    Args:
        data_path (str): Path to the directory containing MIMIC-IV 3.1 hosp/ and icu/ subdirectories.
        cache_path (str): Path to the directory for caching processed data.

    Returns:
        tuple: A tuple containing:
            - features (pd.DataFrame): DataFrame of engineered features.
            - labels (pd.Series): Series of the target variable (30-day readmission).
    """
    os.makedirs(cache_path, exist_ok=True)

    # Create a unique cache filename based on data_path
    cache_file = os.path.join(
        cache_path,
        f'mimic_iv_31_{md5(data_path.encode()).hexdigest()}.joblib'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return joblib.load(cache_file)

    # --- Load Base Tables ---
    admissions = pd.read_csv(
        f'{data_path}/hosp/admissions.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type', 'admission_location',
                 'discharge_location', 'insurance', 'language', 'marital_status', 'race', 'hospital_expire_flag']
    )
    patients = pd.read_csv(
        f'{data_path}/hosp/patients.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod']
    )
    icustays = pd.read_csv(
        f'{data_path}/icu/icustays.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los']
    )

    # Merge base tables
    base = admissions.merge(patients, on='subject_id', how='left')
    # Keep only admissions with corresponding ICU stays for simplicity
    base = base.merge(icustays, on=['subject_id', 'hadm_id'], how='inner')

    # Convert time columns to datetime
    time_cols = ['admittime', 'dischtime', 'intime', 'outtime', 'dod']
    for col in time_cols:
        base[col] = pd.to_datetime(base[col], errors='coerce')

    # --- Target Variable: 30-day Readmission ---
    # Sort by subject_id and admittime to find the next admission
    admissions[['admittime', 'dischtime']] = admissions[['admittime', 'dischtime']].apply(
        pd.to_datetime, errors='coerce'
    )

    # Merge in patient demographics
    admission_base = (
        admissions
        .merge(
            patients[['subject_id', 'anchor_age', 'anchor_year', 'gender']],
            on='subject_id',
            how='left'
        )
    )

    # Sort and compute next admission + delta days
    admission_base.sort_values(['subject_id', 'admittime'], inplace=True)
    admission_base['next_admit'] = admission_base.groupby('subject_id')['admittime'].shift(-1)
    admission_base['time_to_next_admit'] = (
        admission_base['next_admit'] - admission_base['dischtime']
    ).dt.total_seconds() / (24 * 3600)

    # Flag 30-day readmits
    admission_base['readmit_30d'] = (
        admission_base['time_to_next_admit']
        .le(30)
        .fillna(False)
        .astype(int)
    )

    # Pull in first ICU stay info per admission
    first_icustay = icustays.groupby('hadm_id').first().reset_index()[['hadm_id', 'stay_id', 'los']]
    base = admission_base.merge(first_icustay, on='hadm_id', how='left')

    # --- Feature Engineering ---

    # Demographics
    # --- Feature Engineering ---
    # Demographics (now that admission_base has patient info)
    base['age'] = base['anchor_age'] + (
            base['admittime'].dt.year - base['anchor_year']
    )
    base['died_in_hospital'] = base['hospital_expire_flag'].astype(int)

    # Admission details (categorical features)
    categorical_cols = ['admission_type', 'admission_location', 'discharge_location', 'insurance', 'language',
                        'marital_status', 'race']
    base = pd.get_dummies(base, columns=categorical_cols, dummy_na=False)  # Handle NaNs later if needed

    # Diagnosis Features
    diagnoses_icd = pd.read_csv(
        f'{data_path}/hosp/diagnoses_icd.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version']
    )
    # Count number of unique diagnoses per admission
    diag_counts = (diagnoses_icd
                   .groupby('hadm_id')['icd_code']
                   .nunique()
                   .to_frame('num_unique_diagnoses')
                   .reset_index())
    base = base.merge(diag_counts, on='hadm_id', how='left')

    # Procedure Features
    procedures_icd = pd.read_csv(
        f'{data_path}/hosp/procedures_icd.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version']
    )
    # Count number of unique procedures per admission
    proc_counts = (procedures_icd
                   .groupby('hadm_id')['icd_code']
                   .nunique()
                   .to_frame('num_unique_procedures')
                   .reset_index())
    base = base.merge(proc_counts, on='hadm_id', how='left')

    # Lab Features (using user's initial approach with slight modification)
    target_labs = [
        'creatinine', 'glucose', 'wbc', 'hemoglobin', 'platelet', 'sodium',
        'potassium', 'albumin', 'amylase', 'bicarbonate', 'bilirubin',
        'uric acid', 'lactate', 'ph', 'pco2', 'po2', 'base excess', 'anion gap'
    ]  # Added common blood gas/chemistry labs
    labitems = pd.read_csv(f'{data_path}/hosp/d_labitems.csv.gz', compression='gzip')
    labels_lower = labitems['label'].fillna('').str.lower()
    mask = labels_lower.apply(lambda x: any(t in x for t in target_labs))
    target_itemids = labitems[mask]['itemid'].tolist()

    # Load and filter lab events in chunks
    lab_chunks = pd.read_csv(
        f'{data_path}/hosp/labevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
        compression='gzip',
        chunksize=10_000_000
    )
    lab_filtered_chunks = [
        chunk[chunk['itemid'].isin(target_itemids)]
        for chunk in lab_chunks
    ]
    labs_filtered = pd.concat(lab_filtered_chunks, ignore_index=True)

    # Aggregate lab results per admission
    # Merge with labitems to get labels for column names
    labs_filtered = labs_filtered.merge(labitems[['itemid', 'label']], on='itemid', how='left')
    # Aggregate mean and std, handling potential duplicate labels for itemids
    lab_features = (labs_filtered
                    .groupby(['hadm_id', 'label'])['valuenum']
                    .agg(['mean', 'std'])
                    .unstack())
    # Clean up column names - replace special characters
    lab_features.columns = [
        f"lab_{'_'.join(map(str, col))
        .strip()
        .replace(' ', '_')
        .replace('-', '_')
        .replace('(', '')
        .replace(')', '')}"
        for col in lab_features.columns
    ]
    lab_features.reset_index(inplace=True)

    base = base.merge(lab_features, on='hadm_id', how='left')

    # Vital Sign Features from Chartevents (aggregated per admission)
    # Note: This is a simplification. More robust approach would aggregate per ICU stay
    # and consider timing relative to admission/discharge.
    chartevents_itemids = {
        220045: 'HeartRate',
        220050: 'SBP', 220179: 'SBP', 220180: 'SBP',
        220051: 'DBP', 220181: 'DBP',
        220052: 'MAP', 220053: 'MAP',
        220210: 'RespRate', 224689: 'RespRate',
        223762: 'TempC',
        220277: 'SpO2',
        220739: 'GCS_Total', 223900: 'GCS_Motor', 223901: 'GCS_Verbal', 220734: 'GCS_Eye'
    }
    target_chartevents_itemids = list(chartevents_itemids.keys())

    # Load and filter chartevents in chunks
    chart_chunks = pd.read_csv(
        f'{data_path}/icu/chartevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
        compression='gzip',
        chunksize=10_000_000
    )
    chart_filtered_chunks = [
        chunk[chunk['itemid'].isin(target_chartevents_itemids)]
        for chunk in chart_chunks
    ]
    chart_filtered = pd.concat(chart_filtered_chunks, ignore_index=True)

    # Map itemids to meaningful names
    chart_filtered['label'] = chart_filtered['itemid'].map(chartevents_itemids)

    # Aggregate vital signs per admission (mean, min, max)
    vital_sign_features = (chart_filtered
                           .groupby(['hadm_id', 'label'])['valuenum']
                           .agg(['mean', 'min', 'max'])
                           .unstack())
    vital_sign_features.columns = [
        f"vs_{'_'.join(map(str, col)).strip()}"
        for col in vital_sign_features.columns
    ]
    vital_sign_features.reset_index(inplace=True)

    base = base.merge(vital_sign_features, on='hadm_id', how='left')

    # Medication Features (Count unique medications prescribed per admission)
    prescriptions = pd.read_csv(
        f'{data_path}/hosp/prescriptions.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'drug']
    )
    # Count unique drugs per admission
    drug_counts = (prescriptions
                   .groupby('hadm_id')['drug']
                   .nunique()
                   .to_frame('num_unique_drugs')
                   .reset_index())
    base = base.merge(drug_counts, on='hadm_id', how='left')

    # --- Final Data Preparation ---

    # Drop columns used for merging or target calculation
    cols_to_drop = [
        'subject_id', 'hadm_id', 'stay_id', 'admittime', 'dischtime', 'next_admit',
        'time_to_next_admit', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group',
        'dod', 'hospital_expire_flag'  # Keep died_in_hospital dummy instead
    ]
    base.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Separate features and labels
    labels = base['readmit_30d']
    features = base.drop(columns=['readmit_30d'])

    # Handle remaining NaNs (fill with 0 as a simple strategy)
    features.fillna(0, inplace=True)

    # Save to cache
    joblib.dump((features, labels), cache_file)

    return features, labels

import gc
import os
from collections import defaultdict
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

    # Create a unique cache filename based on data_path and function version (optional but good practice)
    # For simplicity, let's just base it on data_path for now.
    cache_file = os.path.join(
        cache_path,
        f'mimic_iv_31_{md5(data_path.encode()).hexdigest()}.joblib'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return joblib.load(cache_file)

    print("Processing data for hospitalization risk prediction...")

    # --- Load Base Tables ---
    # Keep all admissions, even those without ICU stays initially
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

    # Merge patient demographics to admissions
    base = admissions.merge(patients, on='subject_id', how='left')

    # Convert time columns to datetime (admissions and patients base)
    time_cols_base = ['admittime', 'dischtime', 'dod']
    for col in time_cols_base:
        base[col] = pd.to_datetime(base[col], errors='coerce')

    # Convert time columns in icustays
    time_cols_icu = ['intime', 'outtime']
    for col in time_cols_icu:
        icustays[col] = pd.to_datetime(icustays[col], errors='coerce')

    # --- Target Variable: 30-day Readmission ---
    # Sort by subject_id and admittime to find the next admission
    base.sort_values(['subject_id', 'admittime'], inplace=True)
    base['next_admit'] = base.groupby('subject_id')['admittime'].shift(-1)
    base['time_to_next_admit'] = (
                                         base['next_admit'] - base['dischtime']
                                 ).dt.total_seconds() / (24 * 3600)

    # Flag 30-day readmits
    base['readmit_30d'] = (
        base['time_to_next_admit']
        .le(30)
        .fillna(False)  # Readmission within 30 days is False if no next admission
        .astype(int)
    )

    # --- Feature Engineering ---

    # Demographics
    base['age'] = base['anchor_age'] + (
            base['admittime'].dt.year - base['anchor_year']
    )
    base['died_in_hospital'] = base['hospital_expire_flag'].astype(int)
    # Gender - Convert to numeric
    base['gender_M'] = (base['gender'] == 'M').astype(int)
    base.drop(columns=['gender'], inplace=True)  # Drop original gender

    # Admission details (categorical features)
    categorical_cols = ['admission_type', 'admission_location', 'discharge_location', 'insurance', 'language',
                        'marital_status', 'race']
    base = pd.get_dummies(base, columns=categorical_cols, dummy_na=False, prefix='adm')  # Add prefix

    # ICU Stay Features (Associate first ICU stay with admission if exists)
    first_icustay = icustays.groupby('hadm_id').first().reset_index()[['hadm_id', 'stay_id', 'los', 'intime']]
    base = base.merge(first_icustay, on='hadm_id', how='left')
    # Add binary feature indicating if the admission had an ICU stay
    base['has_icu_stay'] = (~base['stay_id'].isna()).astype(int)

    # Diagnosis Features - Add features for common diagnoses
    diagnoses_icd = pd.read_csv(
        f'{data_path}/hosp/diagnoses_icd.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'icd_code', 'seq_num']
    )
    # Count number of unique diagnoses per admission (keep this as it's still useful)
    diag_counts = (diagnoses_icd
                   .groupby('hadm_id')['icd_code']
                   .nunique()
                   .to_frame('num_unique_diagnoses')
                   .reset_index())
    base = base.merge(diag_counts, on='hadm_id', how='left')

    # Identify top N common diagnoses (Example: top 50 or 100, adjust as needed)
    # We'll use the first listed diagnosis (seq_num == 1) as it's often the primary reason for admission
    primary_diagnoses = diagnoses_icd[diagnoses_icd['seq_num'] == 1]
    common_diag_codes = primary_diagnoses['icd_code'].value_counts().nlargest(100).index.tolist() # Top 100 primary diagnoses

    print(f"Creating binary features for {len(common_diag_codes)} common primary diagnoses...")
    new_diag_features_list = []

    # Iterate through common codes and create a Series for each
    for code in common_diag_codes:
        col_name = f'diag_primary_{code.replace(".", "_")}_present'
        # Get the hadm_ids that have this primary diagnosis
        hadm_ids_with_code = primary_diagnoses[primary_diagnoses['icd_code'] == code]['hadm_id']
        # Create a boolean Series, indexed by hadm_id, indicating presence
        # Use base['hadm_id'].isin(...) but create the Series outside the base DataFrame
        is_present = base['hadm_id'].isin(hadm_ids_with_code).astype(int)
        is_present.name = col_name # Name the Series with the desired column name
        new_diag_features_list.append(is_present)

    # Concatenate all the new Series into a single DataFrame
    new_diag_features_df = pd.concat(new_diag_features_list, axis=1)
    base = base.merge(new_diag_features_df, left_index=True, right_index=True, how='left')

    # Procedure Features - Keep count (this part was fine)
    procedures_icd = pd.read_csv(
        f'{data_path}/hosp/procedures_icd.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'icd_code', 'seq_num']
    )
    # Count number of unique procedures per admission
    proc_counts = (procedures_icd
                   .groupby('hadm_id')['icd_code']
                   .nunique()
                   .to_frame('num_unique_procedures')
                   .reset_index())
    base = base.merge(proc_counts, on='hadm_id', how='left')

    # --- Temporal Lab Features ---
    target_labs = [
        'creatinine', 'glucose', 'wbc', 'hemoglobin', 'platelet', 'sodium',
        'potassium', 'albumin', 'amylase', 'bicarbonate', 'bilirubin',
        'uric acid', 'lactate', 'ph', 'pco2', 'po2', 'base excess', 'anion gap',
        'alt', 'ast', 'bilirubin, total', 'ck', 'ck-mb', 'inr(pt)', 'pt', 'ptt',
        'magnesium', 'calcium, total', 'phosphate', 'chloride', 'troponin',
    ]
    labitems = pd.read_csv(f'{data_path}/hosp/d_labitems.csv.gz', compression='gzip')
    labels_lower = labitems['label'].fillna('').str.lower()
    mask = labels_lower.str.extract(rf"({'|'.join(target_labs)})", expand=False).notna().any(axis=1)
    target_itemids_lab = labitems[mask]['itemid'].tolist()

    print(f"Processing labevents for {len(target_itemids_lab)} target lab itemids...")

    # Load and filter lab events with charttime
    agg_all = defaultdict(list)
    agg_24h = defaultdict(list)

    for chunk in pd.read_csv(
            f'{data_path}/hosp/labevents.csv.gz',
            usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
            compression='gzip',
            chunksize=1_000_000
    ):
        chunk = chunk[chunk['itemid'].isin(target_itemids_lab)].copy()
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')

        chunk = chunk.merge(base[['hadm_id', 'admittime']], on='hadm_id', how='left')
        chunk['time_from_admit'] = (chunk['charttime'] - chunk['admittime']).dt.total_seconds() / 3600.0

        chunk = chunk.merge(labitems[['itemid', 'label']], on='itemid', how='left')

        # Filter 24h for this chunk
        chunk_24h = chunk[(chunk['time_from_admit'] >= 0) & (chunk['time_from_admit'] <= 24)]

        # Group and aggregate in each chunk (this avoids storing all rows)
        for df, storage, suffix in [(chunk, agg_all, 'allstay'), (chunk_24h, agg_24h, '24h')]:
            grouped = df.groupby(['hadm_id', 'label'])['valuenum'].agg(
                ['mean', 'std', 'min', 'max', 'first', 'last'])
            for (hadm_id, label), row in grouped.iterrows():
                for stat in ['mean', 'std', 'min', 'max', 'first', 'last']:
                    col_name = f"lab_{label.lower().replace(' ', '_')}_{stat}_{suffix}"
                    storage[(hadm_id, col_name)].append(row[stat])

        gc.collect()

    # Collapse into filtered labs
    labs_filtered = pd.concat([agg_all.values(), agg_24h.values()], ignore_index=True)

    # Define time windows (e.g., first 24 hours)
    window_24h_mask = (labs_filtered['time_from_admit'] >= 0) & (labs_filtered['time_from_admit'] <= 24)

    # Aggregate temporal lab features
    lab_temporal_features_list = []

    # Aggregate over the entire stay
    agg_all_stay = labs_filtered.groupby(['hadm_id', 'label'])['valuenum'].agg(
        ['mean', 'std', 'min', 'max', 'first', 'last']).unstack()
    agg_all_stay.columns = [
        f"lab_{'_'.join(map(str, col)).strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}_allstay"
        for col in agg_all_stay.columns]
    lab_temporal_features_list.append(agg_all_stay)

    # Aggregate within the first 24 hours
    agg_24h = labs_filtered[window_24h_mask].groupby(['hadm_id', 'label'])['valuenum'].agg(
        ['mean', 'std', 'min', 'max', 'first', 'last']).unstack()
    agg_24h.columns = [
        f"lab_{'_'.join(map(str, col)).strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')}_24h"
        for col in agg_24h.columns]
    lab_temporal_features_list.append(agg_24h)

    # Combine all lab features
    lab_features = pd.concat(lab_temporal_features_list,
                             axis=1).reset_index()  # Use axis=1 for concatenating columns

    base = base.merge(lab_features, on='hadm_id', how='left')

    # --- Temporal Vital Sign Features ---
    chartevents_itemids = {
        220045: 'HeartRate', 220050: 'SBP', 220179: 'SBP', 220180: 'SBP',
        220051: 'DBP', 220181: 'DBP', 220052: 'MAP', 220053: 'MAP',
        220210: 'RespRate', 224689: 'RespRate', 223762: 'TempC', 220277: 'SpO2',
        220739: 'GCS_Total', 223900: 'GCS_Motor', 223901: 'GCS_Verbal', 220734: 'GCS_Eye',
        225664: 'MAP_ALERT', 220092: 'PCWP'  # Added a couple more common vitals
    }
    target_chartevents_itemids = list(chartevents_itemids.keys())

    print(f"Processing chartevents for {len(target_chartevents_itemids)} target chart itemids...")

    # Load and filter chartevents with charttime and hadm_id (ensure hadm_id is present)
    chart_chunks = pd.read_csv(
        f'{data_path}/icu/chartevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum'],
        # Include hadm_id and stay_id
        compression='gzip',
        chunksize=1_000_000
    )
    chart_filtered_chunks = [
        chunk[chunk['itemid'].isin(target_chartevents_itemids)].copy()
        for chunk in chart_chunks
    ]
    chart_filtered = pd.concat(chart_filtered_chunks, ignore_index=True)

    # Convert charttime to datetime
    chart_filtered['charttime'] = pd.to_datetime(chart_filtered['charttime'], errors='coerce')

    # Merge with first_icustay to get intime for time window calculations relative to ICU admission
    chart_filtered = chart_filtered.merge(
        first_icustay[['stay_id', 'intime']],
        on='stay_id',
        how='left'  # Keep all chart events
    )

    # Calculate time difference from ICU intime in hours
    chart_filtered['time_from_icu_intime'] = (chart_filtered['charttime'] - chart_filtered[
        'intime']).dt.total_seconds() / 3600.0

    # Map itemids to meaningful names BEFORE aggregation
    chart_filtered['label'] = chart_filtered['itemid'].map(chartevents_itemids)
    # Drop rows where label is NaN (itemid wasn't in our target list)
    chart_filtered.dropna(subset=['label'], inplace=True)

    # Define time windows relative to ICU intime (e.g., first 24 hours)
    # Ensure we only consider values recorded AFTER ICU intime or slightly before for convenience
    window_24h_icu_mask = (chart_filtered['time_from_icu_intime'] >= 0) & (
                chart_filtered['time_from_icu_intime'] <= 24)

    # Aggregate temporal vital sign features per HADM_ID (since features are linked to admission)
    # We need to aggregate per hadm_id, even though times are relative to stay_id's intime
    vital_sign_temporal_features_list = []

    # Aggregate over the entire ICU stay duration associated with the admission
    # Filter for valid times relative to ICU intime
    chart_valid_times = chart_filtered[chart_filtered['time_from_icu_intime'] >= 0]

    if not chart_valid_times.empty:
        agg_all_stay_vs = chart_valid_times.groupby(['hadm_id', 'label'])['valuenum'].agg(
            ['mean', 'std', 'min', 'max', 'first', 'last']).unstack()
        agg_all_stay_vs.columns = [f"vs_{'_'.join(map(str, col)).strip().replace(' ', '_')}_allstay" for col in
                                   agg_all_stay_vs.columns]
        vital_sign_temporal_features_list.append(agg_all_stay_vs)

        # Aggregate within the first 24 hours of ICU stay
        agg_24h_icu = chart_filtered[window_24h_icu_mask].groupby(['hadm_id', 'label'])['valuenum'].agg(
            ['mean', 'std', 'min', 'max', 'first', 'last']).unstack()
        agg_24h_icu.columns = [f"vs_{'_'.join(map(str, col)).strip().replace(' ', '_')}_24hicu" for col in
                               agg_24h_icu.columns]
        vital_sign_temporal_features_list.append(agg_24h_icu)

    # Combine all vital sign features
    if vital_sign_temporal_features_list:
        vital_sign_features = pd.concat(vital_sign_temporal_features_list, axis=1).reset_index()
        base = base.merge(vital_sign_features, on='hadm_id', how='left')
    else:
        print("No vital sign data found for aggregation windows.")
        # Create empty columns to avoid merge errors later if no data exists
        # This requires knowing the expected columns, which is tricky without data.
        # A safer approach is to check for existence before merge.
        pass  # The left merge handles cases where vital_sign_features is empty/missing hadm_ids

    # Medication Features (Count unique medications prescribed per admission - keep simple for now)
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
        'time_to_next_admit', 'anchor_age', 'anchor_year', 'anchor_year_group',
        'dod', 'hospital_expire_flag', 'intime'  # Drop intime after use
    ]
    base.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Separate features and labels
    labels = base['readmit_30d']
    features = base.drop(columns=['readmit_30d'])

    print(f"Generated {len(features.columns)} features.")

    # Handle remaining NaNs
    # Identify columns with NaNs after merging
    nan_cols = features.columns[features.isnull().any()].tolist()

    # Add missing indicator features for columns with NaNs (optional, but can be helpful)
    # Let's add indicators for some key aggregated features if they have NaNs
    aggregated_cols = [col for col in nan_cols if col.startswith('lab_') or col.startswith('vs_')]
    for col in aggregated_cols:
        features[f'{col}_ismissing'] = features[col].isna().astype(int)

    # Fill remaining NaNs (consider different strategies - 0 is simple, but mean/median might be better)
    # Using 0 for now for consistency with the original, but this is a point to experiment
    features.fillna(0, inplace=True)

    # Save to cache
    print(f"Saving processed data to {cache_file}")
    joblib.dump((features, labels), cache_file)

    print("Data processing complete.")

    return features, labels

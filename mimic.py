import gc
import os
from hashlib import md5
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


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
                 'discharge_location', 'insurance', 'language', 'marital_status', 'race', 'hospital_expire_flag'],
        parse_dates=['admittime', 'dischtime'],
    )
    patients = pd.read_csv(
        f'{data_path}/hosp/patients.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod'],
        parse_dates=['dod']
    )
    icustays = pd.read_csv(
        f'{data_path}/icu/icustays.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'],
        parse_dates=['intime', 'outtime']
    )

    # Merge patient demographics to admissions
    base = admissions.merge(patients, on='subject_id', how='left')

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
    common_diag_codes = (primary_diagnoses['icd_code']
                         .value_counts()
                         .nlargest(100)
                         .index
                         .tolist())  # Top 100 primary diagnoses

    print(f"Creating binary features for {len(common_diag_codes)} "
          f"common primary diagnoses...")
    new_diag_features_list = []

    # Iterate through common codes and create a Series for each
    for code in common_diag_codes:
        col_name = f'diag_primary_{code.replace(".", "_")}_present'
        # Get the hadm_ids that have this primary diagnosis
        hadm_ids_with_code = primary_diagnoses[primary_diagnoses['icd_code'] == code]['hadm_id']
        # Create a boolean Series, indexed by hadm_id, indicating presence
        # Use base['hadm_id'].isin(...) but create the Series outside the base DataFrame
        is_present = base['hadm_id'].isin(hadm_ids_with_code).astype(int)
        is_present.name = col_name  # Name the Series with the desired column name
        new_diag_features_list.append(is_present)

    # Concatenate all the new Series into a single DataFrame
    new_diag_features_df = pd.concat(new_diag_features_list, axis=1)
    # Ensure the index of new_diag_features_df aligns with base before merging on index
    new_diag_features_df.index = base.index
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
    reader = pd.read_csv(
        f'{data_path}/hosp/labevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        compression='gzip',
        chunksize=1_000_000,
        dtype={
            'subject_id': 'int32',
            'hadm_id': 'float32',  # float because some might be NaN
            'itemid': 'int32',
            'valuenum': 'float32'
        },
        parse_dates=['charttime']
    )

    lab_all_stay_chunks = []
    lab_24h_chunks = []

    for chunk in tqdm(reader):
        gc.collect()

        # Use .copy() to avoid SettingWithCopyWarning
        chunk = chunk[chunk['itemid'].isin(target_itemids_lab)].copy()

        if chunk.empty:
            continue

        # Merge with admissions to get admittime for time window calculations
        # Ensure hadm_id is int for merging
        chunk['hadm_id'] = chunk['hadm_id'].astype('Int64')  # Use nullable Int64

        # Get admittime from the base dataframe for the relevant hadm_ids in the chunk
        admittime_map = base.set_index('hadm_id')['admittime'].to_dict()
        chunk['admittime'] = chunk['hadm_id'].map(admittime_map)

        # Drop rows where admittime could not be found (shouldn't happen with correct data, but safe)
        chunk.dropna(subset=['admittime', 'charttime'], inplace=True)

        # Calculate time difference from admittime in hours
        chunk['time_from_admit'] = (chunk['charttime'] - chunk['admittime']).dt.total_seconds() / 3600.0

        # Filter for events after admission time
        chunk = chunk[chunk['time_from_admit'] >= 0].copy()  # Use .copy()

        if chunk.empty:
            continue

        # Merge with labitems to get labels
        chunk = chunk.merge(labitems[['itemid', 'label']], on='itemid', how='left')
        chunk.dropna(subset=['label'], inplace=True)  # Drop rows where label is missing

        # --- Aggregate for All Stay ---
        grouped_all_stay = chunk.groupby(['hadm_id', 'label'])['valuenum'].agg(
            ['mean', 'std', 'min', 'max', 'first', 'last'])
        lab_all_stay_chunks.append(grouped_all_stay)

        # --- Aggregate for First 24 Hours ---
        window_24h_mask = (chunk['time_from_admit'] >= 0) & (chunk['time_from_admit'] <= 24)
        chunk_24h = chunk[window_24h_mask].copy()  # Use .copy()

        if not chunk_24h.empty:
            grouped_24h = chunk_24h.groupby(['hadm_id', 'label'])['valuenum'].agg(
                ['mean', 'std', 'min', 'max', 'first', 'last'])
            lab_24h_chunks.append(grouped_24h)

    # Combine and unstack results from chunks
    lab_features_list = []

    if lab_all_stay_chunks:
        print("Combining 'all stay' lab features...")
        lab_all_stay = pd.concat(lab_all_stay_chunks).groupby(
            ['hadm_id', 'label']).mean()  # Use mean to aggregate across chunks
        lab_all_stay = lab_all_stay.unstack()
        lab_all_stay.columns = [f"lab_{label.lower().replace(' ', '_')}_{stat}_allstay"
                                for label, stat in lab_all_stay.columns]
        lab_features_list.append(lab_all_stay)

    if lab_24h_chunks:
        print("Combining '24h' lab features...")
        lab_24h = pd.concat(lab_24h_chunks).groupby(['hadm_id', 'label']).mean()  # Use mean to aggregate across chunks
        lab_24h = lab_24h.unstack()
        lab_24h.columns = [f"lab_{label.lower().replace(' ', '_')}_{stat}_24h"
                           for label, stat in lab_24h.columns]
        lab_features_list.append(lab_24h)

    if lab_features_list:
        # Concatenate all lab feature dataframes
        lab_features = pd.concat(lab_features_list, axis=1).reset_index()
        # Merge once to base
        base = base.merge(lab_features, on='hadm_id', how='left')
    else:
        print("No lab data found for aggregation.")
        # Create empty lab features if none were processed to avoid merge errors
        # This requires knowing potential column names, which is complex.
        # The left merge handles missing hadm_ids in lab_features if it's empty.

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
    reader = pd.read_csv(
        f'{data_path}/icu/chartevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum'],
        compression='gzip',
        chunksize=1_000_000,
        dtype={
            'subject_id': 'int32',
            'hadm_id': 'float32',  # float because some might be NaN
            'stay_id': 'float32',  # same reason as above
            'itemid': 'int32',
            'valuenum': 'float32'
        },
        parse_dates=['charttime']
    )
    chart_all_stay_chunks = []
    chart_24h_icu_chunks = []

    for chunk in tqdm(reader):
        gc.collect()

        filtered = chunk[chunk['itemid'].isin(target_chartevents_itemids)].copy()  # Use .copy()
        if filtered.empty:
            continue

        # Ensure hadm_id and stay_id are int for merging, handle NaNs
        filtered['hadm_id'] = filtered['hadm_id'].astype('Int64')
        filtered['stay_id'] = filtered['stay_id'].astype('Int64')
        filtered.dropna(subset=['hadm_id', 'stay_id'], inplace=True)  # Drop rows with missing hadm_id or stay_id

        # Merge with first_icustay to get intime for time window calculations relative to ICU admission
        # Get intime from first_icustay for the relevant stay_ids in the chunk
        intime_map = first_icustay.set_index('stay_id')['intime'].to_dict()
        filtered['intime'] = filtered['stay_id'].map(intime_map)

        # Drop rows where intime could not be found (event outside known ICU stays)
        filtered.dropna(subset=['intime', 'charttime'], inplace=True)

        # Calculate time difference from ICU intime in hours
        filtered['time_from_icu_intime'] = (filtered['charttime'] - filtered['intime']).dt.total_seconds() / 3600.0

        # Map itemids to meaningful names BEFORE aggregation
        filtered['label'] = filtered['itemid'].map(chartevents_itemids)
        # Drop rows where label is NaN (itemid wasn't in our target list)
        filtered.dropna(subset=['label', 'hadm_id'], inplace=True)  # Also ensure hadm_id is not NaN

        if filtered.empty:
            continue

        # --- Aggregate for All ICU Stay (relative to ICU intime) ---
        # Filter for valid times relative to ICU intime
        chart_valid_times = filtered[filtered['time_from_icu_intime'] >= 0].copy()  # Use .copy()

        if not chart_valid_times.empty:
            agg_all_stay_vs = chart_valid_times.groupby(['hadm_id', 'label'])['valuenum'].agg(
                ['mean', 'std', 'min', 'max', 'first', 'last'])
            chart_all_stay_chunks.append(agg_all_stay_vs)

        # --- Aggregate within the first 24 hours of ICU stay ---
        window_24h_icu_mask = (filtered['time_from_icu_intime'] >= 0) & (filtered['time_from_icu_intime'] <= 24)
        chunk_24h_icu = filtered[window_24h_icu_mask].copy()  # Use .copy()

        if not chunk_24h_icu.empty:
            agg_24h_icu = chunk_24h_icu.groupby(['hadm_id', 'label'])['valuenum'].agg(
                ['mean', 'std', 'min', 'max', 'first', 'last'])
            chart_24h_icu_chunks.append(agg_24h_icu)

    # Combine and unstack results from chunks
    vital_sign_features_list = []

    if chart_all_stay_chunks:
        print("Combining 'all stay' vital sign features...")
        vital_sign_all_stay = pd.concat(chart_all_stay_chunks).groupby(
            ['hadm_id', 'label']).mean()  # Aggregate across chunks
        vital_sign_all_stay = vital_sign_all_stay.unstack()
        vital_sign_all_stay.columns = [f"vs_{'_'.join(map(str, col)).strip().replace(' ', '_')}_allstay" for col in
                                       vital_sign_all_stay.columns]
        vital_sign_features_list.append(vital_sign_all_stay)

    if chart_24h_icu_chunks:
        print("Combining '24h ICU' vital sign features...")
        vital_sign_24h_icu = pd.concat(chart_24h_icu_chunks).groupby(
            ['hadm_id', 'label']).mean()  # Aggregate across chunks
        vital_sign_24h_icu = vital_sign_24h_icu.unstack()
        vital_sign_24h_icu.columns = [f"vs_{'_'.join(map(str, col)).strip().replace(' ', '_')}_24hicu" for col in
                                      vital_sign_24h_icu.columns]
        vital_sign_features_list.append(vital_sign_24h_icu)

    # Combine all vital sign features
    if vital_sign_features_list:
        vital_sign_features = pd.concat(vital_sign_features_list, axis=1).reset_index()
        base = base.merge(vital_sign_features, on='hadm_id', how='left')
    else:
        print("No vital sign data found for aggregation windows.")
        # The left merge handles cases where vital_sign_features is empty/missing hadm_ids

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

    # Columns to exclude from features in the final step
    cols_to_exclude_from_features = [
        'subject_id', 'hadm_id', 'stay_id', 'admittime', 'dischtime', 'next_admit',
        'time_to_next_admit', 'anchor_age', 'anchor_year', 'anchor_year_group',
        'dod', 'hospital_expire_flag', 'intime',  # Drop intime after use
        'readmit_30d'  # This is the label, so exclude from features
        # Any other temporary columns used during processing that shouldn't be features
    ]

    # Separate labels BEFORE creating the final features DataFrame
    labels = base['readmit_30d']

    # Create the initial features DataFrame by selecting desired columns from base
    # This creates a new, non-fragmented DataFrame
    feature_cols = [col for col in base.columns if col not in cols_to_exclude_from_features]
    features = base[feature_cols].copy()  # Use .copy() to ensure it's a distinct DataFrame

    print(f"Initial feature selection generated {len(features.columns)} features.")

    # Handle remaining NaNs and add missing indicators more efficiently

    # Identify columns with NaNs *before* filling
    nan_cols_before_filling = features.columns[features.isnull().any()].tolist()

    # Create missing indicator features in a separate DataFrame
    missing_indicator_features_list = []
    # Only add indicators for aggregated features if they have NaNs
    aggregated_cols_with_nan = [col for col in nan_cols_before_filling if
                                col.startswith('lab_') or col.startswith('vs_')]

    if aggregated_cols_with_nan:
        print(f"Creating missing indicators for {len(aggregated_cols_with_nan)} aggregated columns with NaNs...")
        for col in aggregated_cols_with_nan:
            # Create the indicator Series
            indicator_series = features[col].isna().astype(int)
            indicator_series.name = f'{col}_ismissing'  # Name the series
            missing_indicator_features_list.append(indicator_series)

    # Concatenate all missing indicator Series into a single DataFrame
    if missing_indicator_features_list:
        missing_indicators_df = pd.concat(missing_indicator_features_list, axis=1)
        # Concatenate the main features DataFrame and the missing indicator DataFrame
        features = pd.concat([features, missing_indicators_df], axis=1)
        print(f"Total features after adding indicators: {len(features.columns)}")

    # Fill remaining NaNs in the main features DataFrame (including newly added indicator columns if any had NaNs, though they shouldn't)
    print("Filling remaining NaNs with 0...")
    features.fillna(0, inplace=True)

    # --- Optimize Data Types (Downcasting) ---
    print("Optimizing data types to reduce memory usage...")
    initial_memory = features.memory_usage(deep=True).sum() / (1024 ** 3)  # in GB

    for col in features.columns:
        col_type = features[col].dtype

        # Downcast floats
        if col_type == 'float64':
            # Use ._check_values for older pandas versions if needed, but astype handles conversion
            # Check if column contains non-finite values (NaN, inf) that need to be handled
            if features[col].hasnans:
                # If NaNs are present, downcasting to float32 is usually safe
                features[col] = features[col].astype('float32')
            else:
                # If no NaNs, can try converting to integer first if possible
                # This step is optional but can save more memory if floats are actually integers
                temp_int = features[col].astype('int64', errors='ignore')
                if (temp_int == features[col]).all():
                    features[col] = temp_int.astype('int32', errors='ignore')  # Try int32
                    if features[col].dtype == 'int64':  # If int32 failed, stick to float32
                        features[col] = features[col].astype('float32')
                else:
                    features[col] = features[col].astype('float32')


        # Downcast integers
        elif col_type == 'int64':
            # Check if min/max fit in smaller integer types
            c_min = features[col].min()
            c_max = features[col].max()
            if c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                features[col] = features[col].astype('int32')
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                features[col] = features[col].astype('int16')
            elif c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                features[col] = features[col].astype('int8')

    final_memory = features.memory_usage(deep=True).sum() / (1024 ** 3)  # in GB
    print(f"Memory usage before downcasting: {initial_memory:.2f} GB")
    print(f"Memory usage after downcasting: {final_memory:.2f} GB")
    print(f"Memory saved: {initial_memory - final_memory:.2f} GB")
    print("Data type optimization complete.")

    # Save to cache
    print(f"Saving processed data to {cache_file}")
    # The index of features should be aligned with labels since features was derived from base
    joblib.dump((features, labels), cache_file)

    print("Data processing complete.")

    return features, labels

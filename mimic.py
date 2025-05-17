import gc
import os

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

CHUNK_SIZE = 10 ** 7  # Adjust based on memory constraints


def get_mimic_iv_31(data_path='datasets', cache_path='datasets/cache'):
    """
    Prepares data from MIMIC-IV 3.1 for hospitalization risk prediction.

    Args:
        data_path (str): Path to the directory containing MIMIC-IV 3.1 hosp/ and icu/ subdirectories.
        cache_path (str): Path to the directory for caching processed data.

    Returns:
        A Parquet file containing the processed data.
    """
    os.makedirs(cache_path, exist_ok=True)

    # Create a unique cache filename based on data_path and function version (optional but good practice)
    # For simplicity, let's just base it on data_path for now.
    cache_file = os.path.join(
        cache_path,
        'mimic_iv_31.parquet'
    )

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        return dd.read_parquet(cache_file)

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
    features = admissions.merge(patients, on='subject_id', how='left')

    # --- Target Variable: 30-day Readmission ---
    # Sort by subject_id and admittime to find the next admission
    features.sort_values(['subject_id', 'admittime'], inplace=True)
    features['next_admit'] = features.groupby('subject_id')['admittime'].shift(-1)
    features['time_to_next_admit'] = (
                                             features['next_admit'] - features['dischtime']
                                     ).dt.total_seconds() / (24 * 3600)

    # Flag 30-day readmits
    features['readmit_30d'] = (
        features['time_to_next_admit']
        .le(30)
        .fillna(False)  # Readmission within 30 days is False if no next admission
        .astype(int)
    )

    # --- Feature Engineering ---

    # Demographics
    features['age'] = features['anchor_age'] + (
            features['admittime'].dt.year - features['anchor_year']
    )
    features['died_in_hospital'] = features['hospital_expire_flag'].astype(int)
    # Gender - Convert to numeric
    features['gender_M'] = (features['gender'] == 'M').astype(int)
    features.drop(columns=['gender'], inplace=True)  # Drop original gender

    # Admission details (categorical features)
    categorical_cols = ['admission_type', 'admission_location', 'discharge_location', 'insurance', 'language',
                        'marital_status', 'race']
    features = pd.get_dummies(features, columns=categorical_cols, dummy_na=False, prefix='adm')  # Add prefix

    # ICU Stay Features (Associate first ICU stay with admission if exists)
    first_icustay = icustays.groupby('hadm_id').first().reset_index()[['hadm_id', 'stay_id', 'los', 'intime']]
    features = features.merge(first_icustay, on='hadm_id', how='left')
    # Add binary feature indicating if the admission had an ICU stay
    features['has_icu_stay'] = (~features['stay_id'].isna()).astype(int)

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
    features = features.merge(diag_counts, on='hadm_id', how='left')

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
        # Use features['hadm_id'].isin(...) but create the Series outside the base DataFrame
        is_present = features['hadm_id'].isin(hadm_ids_with_code).astype(int)
        is_present.name = col_name  # Name the Series with the desired column name
        new_diag_features_list.append(is_present)

    # Concatenate all the new Series into a single DataFrame
    new_diag_features_df = pd.concat(new_diag_features_list, axis=1)
    # Ensure the index of new_diag_features_df aligns with base before merging on index
    new_diag_features_df.index = features.index
    features = features.merge(new_diag_features_df, left_index=True, right_index=True, how='left')

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
    features = features.merge(proc_counts, on='hadm_id', how='left')

    # --- Temporal Lab Features ---
    # target_labs = [
    #     'creatinine', 'glucose', 'wbc', 'hemoglobin', 'platelet', 'sodium',
    #     'potassium', 'albumin', 'amylase', 'bicarbonate', 'bilirubin',
    #     'uric acid', 'lactate', 'ph', 'pco2', 'po2', 'base excess', 'anion gap',
    #     'alt', 'ast', 'bilirubin, total', 'ck', 'ck-mb', 'inr(pt)', 'pt', 'ptt',
    #     'magnesium', 'calcium, total', 'phosphate', 'chloride', 'troponin',
    # ]
    target_labs = [
        'creatinine',     # kidney function
        # 'potassium',      # electrolyte imbalance
        # 'sodium',         # common and dangerous if abnormal
        # 'wbc',            # infection/inflammation
        # 'hemoglobin',     # anemia/bleeding
        'lactate',        # sepsis/tissue hypoxia
        # 'glucose',        # critical hypo/hyperglycemia
        # 'ph',             # acid-base balance
        # 'inr(pt)',        # bleeding/clotting risk
        'troponin'        # cardiac damage
    ]

    labitems = pd.read_csv(f'{data_path}/hosp/d_labitems.csv.gz', compression='gzip')
    labels_lower = labitems['label'].fillna('').str.lower()
    mask = labels_lower.str.extract(rf"({'|'.join(target_labs)})", expand=False).notna().any()
    target_itemids_lab = labitems[mask]['itemid'].tolist()

    print(f"Processing labevents for {len(target_itemids_lab)} target lab itemids...")

    # Load and filter lab events with charttime
    reader = pd.read_csv(
        f'{data_path}/hosp/labevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
        compression='gzip',
        chunksize=CHUNK_SIZE,
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
        admittime_map = features.set_index('hadm_id')['admittime'].to_dict()
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
        # Merge once to features
        features = features.merge(lab_features, on='hadm_id', how='left')
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
        chunksize=CHUNK_SIZE,
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
        features = features.merge(vital_sign_features, on='hadm_id', how='left')
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
    features = features.merge(drug_counts, on='hadm_id', how='left')

    # --- Final Data Preparation ---

    # Columns to exclude from features in the final step
    cols_to_exclude_from_features = [
        'subject_id', 'hadm_id', 'stay_id', 'admittime', 'dischtime', 'next_admit',
        'time_to_next_admit', 'anchor_age', 'anchor_year', 'anchor_year_group',
        'dod', 'hospital_expire_flag', 'intime',  # Drop intime after use
        'readmit_30d'  # This is the label, so exclude from features
        # Any other temporary columns used during processing that shouldn't be features
    ]

    # Separate labels BEFORE dropping columns
    labels = features['readmit_30d']

    # Create the initial features DataFrame
    print(f"Initial feature selection generated {len(features.columns)} features.")
    features.drop(columns=cols_to_exclude_from_features, inplace=True, errors="ignore")

    # Handle remaining NaNs and add missing indicators more efficiently
    aggregated_cols = [c for c in features.columns
                       if c.startswith('lab_') or c.startswith('vs_')]
    cols_with_nan = [c for c in aggregated_cols if features[c].isna().any()]

    # Detect duplicate columns
    duplicates = features.columns[features.columns.duplicated()].tolist()
    if duplicates:
        print(f"{len(duplicates)} duplicate columns detected and will be dropped: {duplicates}")
        features = features.loc[:, ~features.columns.duplicated()]
    del duplicates
    gc.collect()

    # fill nans with 0 for numerical features
    features.fillna({c: 0 for c in cols_with_nan}, inplace=True)
    features.astype({ c: 'float32' for c in cols_with_nan }, copy=False)

    print('Final feature selection generated '
          f'{len(features.columns)} features after dropping NaNs and duplicates.\n'
          f'Features: {features.columns.tolist()}\n'
          f'Labels: {labels.name}\n'
          f'Starting to cache the data...\n')

    # Combine features + label, write parquet, and return Dask-CuDF
    df = pd.concat([features, labels.rename('readmit_30d')], axis=1)
    df.to_parquet(
        path=cache_file,
        compression='snappy',
        index=False,
        row_group_size=500,
    )

    print(f"Data processing complete. Cached data saved.")
    return dd.read_parquet(cache_file)

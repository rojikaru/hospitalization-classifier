from sqlalchemy import create_engine
import pandas as pd


DATABASE_URL = 'postgresql://mimic-iv_owner:npg_aN0dqO4mVBpi@ep-round-sun-a28cdod8-pooler.eu-central-1.aws.neon.tech/mimic-iv?sslmode=require'


def get_mimic_iv_22_demo():
    engine = create_engine(DATABASE_URL)

    # Load data from the database
    admissions = pd.read_sql("SELECT * FROM mimic_hosp.admissions", engine)
    patients = pd.read_sql("SELECT * FROM mimic_hosp.patients", engine)
    transfers = pd.read_sql("SELECT * FROM mimic_hosp.transfers", engine)

    # Filter ICU admissions and merge with hospital admissions
    icu_transfers = transfers[transfers['careunit'].str.contains('ICU', na=False)]
    merged = pd.merge(admissions, icu_transfers, on=["hadm_id", "subject_id"])

    # Calculate time to ICU and create the target variable
    merged['time_to_icu'] = pd.to_datetime(merged['intime']) - pd.to_datetime(merged['admittime'])
    merged['icu_within_24h'] = (merged['time_to_icu'].dt.total_seconds() <= 86400).astype(int)

    # Merge with patient data to include age
    patients['anchor_age'] = 2022 - pd.to_datetime(patients['dod']).dt.year
    data = pd.merge(merged, patients[['subject_id', 'anchor_age']], on='subject_id')

    # Drop rows with missing values in key columns
    data = data.dropna(subset=['anchor_age', 'icu_within_24h'])

    # Select specific columns for features
    retX = data[["anchor_age"]].apply(pd.to_numeric, errors='coerce')
    retY = pd.to_numeric(data['icu_within_24h'], errors='coerce')

    # Drop rows with any remaining invalid values
    retX.dropna(inplace=True)
    retY = retY[retX.index]

    return retX, retY


def get_mimic_iv_31():
    # Load relevant datasets
    admissions = pd.read_csv(
        'datasets/hosp/admissions.csv.gz',
        compression='gzip',
        usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime']
    )
    patients = pd.read_csv(
        'datasets/hosp/patients.csv.gz',
        compression='gzip',
        # usecols=['subject_id', 'dod', 'dod_hosp', 'dod_ssn']
    )
    icustays = pd.read_csv(
        'datasets/icu/icustays.csv.gz',
        compression='gzip',
        # usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'first_careunit']
    )

    # Merge for base table
    base = admissions.merge(patients, on='subject_id').merge(icustays, on=['subject_id', 'hadm_id'])

    # Define Hospitalization Risk Target
    base['admittime'] = pd.to_datetime(base['admittime'], errors='coerce')
    base['dischtime'] = pd.to_datetime(base['dischtime'], errors='coerce')
    base.sort_values(['subject_id', 'admittime'], inplace=True)
    base['next_admit'] = pd.to_datetime(
        base.groupby('subject_id')['admittime'].shift(-1),
        errors='coerce'
    )
    base['readmit_30d'] = (
            (base['next_admit'] - base['dischtime']) <= pd.Timedelta(days=30)
    ).astype(int)

    # Extract Features

    # Pick relevant lab tests (example IDs for illustration)
    # Target lab names as substrings
    target_labs = [
        'creatinine',
        'glucose',
        'wbc',
        'hemoglobin',
        'platelet',
        'sodium',
        'potassium',
        'albumin',
        'amylase',
        'bicarbonate',
        'bilirubin',
        'immunoelectrophoresis',
        'immunoglobulin g',
        'uric acid',
    ]

    # Load labitems
    labitems = pd.read_csv('datasets/hosp/d_labitems.csv.gz', compression='gzip')

    # Filter: case-insensitive substring match
    labels_lower = labitems['label'].fillna('').str.lower()
    mask = labels_lower.apply(lambda x: any(t in x for t in target_labs))
    target_itemids = labitems[mask]['itemid'].tolist()

    # Optional: preview the matching labels
    matching_labels = labitems[mask][['itemid', 'label']]
    print("Matching lab items:")
    print(matching_labels)

    # Load lab events
    lab_chunks = pd.read_csv(
        'datasets/hosp/labevents.csv.gz',
        usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
        compression='gzip',
        chunksize=10_000_000
    )
    lab_filtered_chunks = [
        chunk[chunk['itemid'].isin(target_itemids)]
        for chunk in lab_chunks
    ]

    # Filter lab events
    labs_filtered = pd.concat(lab_filtered_chunks, ignore_index=True)
    labs_filtered = labs_filtered.merge(labitems[['itemid', 'label']], on='itemid')

    # Aggregate lab results per patient stay
    lab_features = labs_filtered.groupby(['subject_id', 'hadm_id', 'label'])['valuenum'].agg(['mean', 'std']).unstack()
    lab_features.columns = ['_'.join(col).strip() for col in lab_features.columns.values]
    lab_features.reset_index(inplace=True)

    # Merge features
    data = base.merge(lab_features, on=['subject_id', 'hadm_id'], how='left')

    features = data.drop(columns=[
        'readmit_30d',
        'subject_id',
        'hadm_id',
        'stay_id',
        'admittime',
        'dischtime',
        'next_admit',
    ])
    labels = data['readmit_30d']
    return features, labels

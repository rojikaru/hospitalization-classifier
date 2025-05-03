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
    columns_to_return = ["anchor_age", ]
    retX = data[["anchor_age"]].apply(pd.to_numeric, errors='coerce')
    retY = pd.to_numeric(data['icu_within_24h'], errors='coerce')

    # Drop rows with any remaining invalid values
    retX.dropna(inplace=True)
    retY = retY[retX.index]

    return retX, retY

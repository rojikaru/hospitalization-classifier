from sqlalchemy import create_engine
import pandas as pd

DATABASE_URL='postgresql://mimic-iv_owner:npg_aN0dqO4mVBpi@ep-round-sun-a28cdod8-pooler.eu-central-1.aws.neon.tech/mimic-iv?sslmode=require'


def get_mimic_iv_22_demo():
    engine = create_engine(DATABASE_URL)

    # Example: Load admissions
    admissions = pd.read_sql("SELECT * FROM mimic_hosp.admissions", engine)
    patients = pd.read_sql("SELECT * FROM mimic_hosp.patients", engine)
    transfers = pd.read_sql("SELECT * FROM mimic_hosp.transfers", engine)

    # Merge ICU admissions with hospital admissions
    icu_transfers = transfers[transfers['careunit'].str.contains('ICU', na=False)]

    # Join on admission ID
    merged = pd.merge(admissions, icu_transfers, on=["hadm_id", "subject_id"])

    # Filter ICU admissions within 24h of hospital admission
    merged['time_to_icu'] = pd.to_datetime(merged['intime']) - pd.to_datetime(merged['admittime'])
    merged['icu_within_24h'] = merged['time_to_icu'].dt.total_seconds() <= 86400  # 24h in seconds

    labels = merged[['subject_id', 'hadm_id', 'icu_within_24h']].drop_duplicates()

    # Add age
    patients['anchor_age'] = 2022 - pd.to_datetime(patients['dod']).dt.year
    data = pd.merge(labels, patients[['subject_id', 'anchor_age']], on='subject_id')

    # Filter out patients with no age information
    data = data[data['anchor_age'].notnull()]

    return data

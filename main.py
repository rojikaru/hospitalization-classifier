from time import time

from sqlalchemy import create_engine
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATABASE_URL='postgresql://mimic-iv_owner:npg_aN0dqO4mVBpi@ep-round-sun-a28cdod8-pooler.eu-central-1.aws.neon.tech/mimic-iv?sslmode=require'


def main():
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

    # Assume `X` are your features and `y` is 'icu_within_24h'
    X = data[['anchor_age']]
    y = data['icu_within_24h']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()

    print(f"Time elapsed: {end_time - start_time} seconds")

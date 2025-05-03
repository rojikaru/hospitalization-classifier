from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from mimic import get_mimic_iv_22_demo
from covid import get_covid_19


def train_on_data(data, name='MIMIC-IV 2.2 Demo'):
    print(f'Training on {name}...')
    start_time = time()

    # Assume `X` are your features and `y` is 'icu_within_24h'
    X, y = data
    print(f"Data to process: {X.shape[0]} rows, {X.shape[1]} columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    end_time = time()

    print(classification_report(y_test, y_pred))
    print(f"Time elapsed: {end_time - start_time} seconds")


def main():
    # mimic_demo = get_mimic_iv_22_demo()
    # train_on_data(mimic_demo, name='MIMIC-IV 2.2 Demo')
    covid_data = get_covid_19()
    train_on_data(covid_data, name='COVID-19 Data')


if __name__ == '__main__':
    main()

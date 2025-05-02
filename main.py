from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from demo import get_mimic_iv_22_demo


def train_on_data(data, name='MIMIC-IV 2.2 Demo'):
    print(f'Training on {name}...')
    start_time = time()

    # Assume `X` are your features and `y` is 'icu_within_24h'
    X = data[['anchor_age']]
    y = data['icu_within_24h']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    end_time = time()

    print(classification_report(y_test, y_pred))
    print(f"Time elapsed: {end_time - start_time} seconds")


def main():
    mimic_demo = get_mimic_iv_22_demo()
    train_on_data(mimic_demo, name='MIMIC-IV 2.2 Demo')


if __name__ == '__main__':
    main()

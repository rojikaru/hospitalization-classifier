from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from covid import get_covid_19
from mimic import get_mimic_iv_31
from misc import prepare_data_for_training


def main():
    name = 'MIMIC-IV 3.1'
    # name = 'COVID-19'
    print(f'Loading {name} data...')

    data_load_start = time()
    match name:
        case 'MIMIC-IV 3.1':
            data = get_mimic_iv_31()
        case 'COVID-19':
            data = get_covid_19()
        case _:
            raise ValueError(f"Unknown dataset: {name}")
    data_load_end = time()

    print(f"Data loaded in {data_load_end - data_load_start} seconds")

    print(f'Training on {name}...')
    start_time = time()

    X_train, y_train, X_test, y_test = prepare_data_for_training(
        data[0], data[1], test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    end_time = time()

    print(classification_report(y_test, y_pred))
    print(f"Time elapsed: {end_time - start_time} seconds")


if __name__ == '__main__':
    main()

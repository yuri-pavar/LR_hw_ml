import pandas as pd
import numpy as np
import re
import pickle

PICKLE_PATH = 'model_data.pkl'
noname_cars = ['Maruti', 'Tata', 'Force', 'MG', 'Ambassador', 'Ashok', 'Mahindra']


def correct_mileage(x):
    if not isinstance(x, str):
        return np.nan
    else:
        # пробуем преобразовать в float
        try:
            res = float(re.sub(r'[^\d\.]', '', x))
        except:
            res = np.nan
        # перевод единиц
        if 'km/kg' in x:
            res = res * 0.73

        return res


def get_pickle_data(path):
    with open(path, 'rb') as file:
        model_data = pickle.load(file)
    return model_data


def preprocessing(input_df):
    df = input_df.copy()
    # читаем сохраненные данные модели
    model_data = get_pickle_data(PICKLE_PATH)
    # обработка признаков
    df.mileage = df.mileage.apply(correct_mileage)
    df.engine = df.engine.apply(lambda x: np.nan if pd.isnull(x) or re.match(r'.*\d+\.?\d+?.*', x) is None
                                          else float(re.sub(r'[^\d\.]', '', x)))
    df.max_power = df.max_power.apply(lambda x: np.nan if pd.isnull(x) or re.match(r'.*\d+\.?\d+?.*', x) is None
                                                else float(re.sub(r'[^\d\.]', '', x)))
    # удаляем таргет и torque
    df.drop(['torque', 'selling_price'], axis=1, inplace=True, errors='ignore')
    # заполняем пропуски медианами из треина
    for col in ['mileage', 'engine', 'max_power', 'seats']:
        df.loc[df[col].isna(), col] = model_data[col + '_median']
    # приводим к целым числам
    df.engine = df.engine.astype(int)
    df.seats = df.seats.astype(int)
    # логарифмируем признак
    df.km_driven = np.log(df.km_driven.astype(float) + 1)
    # новый признак
    df['prem'] = df.name.str.split().str[0].apply(lambda x: 'no_prem' if x in noname_cars else 'prem')
    # удаляем признак name
    df.drop(['name'], axis=1, inplace=True, errors='ignore')
    # переопределяем признак owner
    df.owner = df.owner.apply(lambda x: x if x in ['First Owner', 'Second Owner'] else 'Above 2 Owner')
    # заменяем год на его квадрат
    df.year = df.year ** 2

    return df


def get_prediction(input_df):
    # читаем сохраненные данные модели
    model_data = get_pickle_data(PICKLE_PATH)
    # предобработка данных
    prep_df = preprocessing(input_df)
    # модель и scaler из pickle
    model, column_transformer = model_data['model'], model_data['scaler']
    # преобразование признаков
    X_new_transf = pd.DataFrame(column_transformer.transform(prep_df),
                                columns=column_transformer.get_feature_names_out())
    # логарифмированное предсказание
    pred_raw = model.predict(X_new_transf)
    # предсказание
    pred = np.exp(pred_raw) - 1

    return pred


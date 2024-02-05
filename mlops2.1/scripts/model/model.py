import pandas as pd
import pickle
import yaml
import sys
import os
from catboost import CatBoostRegressor

def load_params(param_file):
    # Загрузка параметров обучения из YAML файла
    with open(param_file, 'r') as file:
        params = yaml.safe_load(file)['train']
    return params

def load_data(data_file):
    # Загрузка данных для обучения
    data = pd.read_csv(data_file)
    X = data.iloc[:, [1, 2, 3]]
    y = data.iloc[:, 0]
    return X, y

def train_model(X, y, params):
    # Обучение модели
    model = CatBoostRegressor(iterations=params['iterations'], random_seed=params['random_seed'], loss_function='RMSE')
    model.fit(X, y)
    return model

def save_model(model, model_path):
    # Сохранение обученной модели
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Ошибка аргументов. Использовать:\n")
        sys.stderr.write("\tpython dt.py data-file model \n")
        sys.exit(1)

    data_file = sys.argv[1]
    model_output = os.path.join("models", sys.argv[2])
    os.makedirs("models", exist_ok=True)

    params = load_params("params.yaml")
    X, y = load_data(data_file)
    model = train_model(X, y, params)
    save_model(model, model_output)

if __name__ == "__main__":
    main()

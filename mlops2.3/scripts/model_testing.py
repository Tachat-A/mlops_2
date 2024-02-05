import os
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score
import joblib

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps3/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('model_testing')

# Загрузка тестовых данных
X_test = np.loadtxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_test.csv', delimiter=',')
y_test = np.loadtxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_test.csv', delimiter=',')

# Загрузка обученной модели из файла .pkl
model_filename = '/Users/AntroNataExtyl/Desktop/MineMlOps3/model/model.pkl'
model = joblib.load(model_filename)

with mlflow.start_run():
    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка производительности модели
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('accuracy', accuracy)

    # Логирование результатов тестирования
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_test.csv', 'test_data')
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_test.csv', 'test_data')

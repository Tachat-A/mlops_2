import os
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
import joblib

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps3/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('model_training')

# Загрузка обучающих данных
X_train = np.loadtxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_train.csv', delimiter=',')
y_train = np.loadtxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_train.csv', delimiter=',')

with mlflow.start_run():
    # Создание и обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Логирование параметров модели
    mlflow.log_param('model', 'LogisticRegression')

    # Сохранение модели в формате .pkl
    model_filename = '/Users/AntroNataExtyl/Desktop/MineMlOps3/model/model.pkl'
    joblib.dump(model, model_filename)

    # Логирование модели в MLflow
    mlflow.sklearn.log_model(model, 'model')

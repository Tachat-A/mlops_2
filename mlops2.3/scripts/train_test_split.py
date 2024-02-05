import os
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps3/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('data_splitting')

# Загрузка обработанных данных и меток кластеров
processed_comments_file = '/Users/AntroNataExtyl/Desktop/MineMlOps3/data/processed_comments.csv'
cluster_labels_file = '/Users/AntroNataExtyl/Desktop/MineMlOps3/data/cluster_labels.csv'

X_matrix = np.loadtxt(processed_comments_file, delimiter=',')
labels = np.loadtxt(cluster_labels_file, delimiter=',')

with mlflow.start_run():
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_matrix, labels, test_size=0.2, random_state=42)

    # Логирование параметров
    mlflow.log_param('test_size', 0.2)
    
    # Сохранение разделенных данных
    np.savetxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_train.csv', X_train, delimiter=',')
    np.savetxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_test.csv', X_test, delimiter=',')
    np.savetxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_train.csv', y_train, delimiter=',')
    np.savetxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_test.csv', y_test, delimiter=',')
    
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_train.csv', 'split_data')
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/X_test.csv', 'split_data')
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_train.csv', 'split_data')
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/y_test.csv', 'split_data')

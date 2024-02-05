import os
import re
import numpy as np
from pymystem3 import Mystem
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def remove_emoji(string):
    """
    Удаление эмоджи из текста
    """
    emoji_pattern = re.compile('['u'\U0001F600-\U0001F64F'  # emoticons
                               u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                               u'\U0001F680-\U0001F6FF'  # transport & map symbols
                               u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                               u'\U00002702-\U000027B0'
                               u'\U000024C2-\U0001F251'
                               u'\U0001f926-\U0001f937'
                               u'\U00010000-\U0010ffff'
                               u'\u200d'
                               u'\u2640-\u2642'
                               u'\u2600-\u2B55'
                               u'\u23cf'
                               u'\u23e9'
                               u'\u231a'
                               u'\u3030'
                               u'\ufe0f'
                               ']+', flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_links(string):
    """
    Удаление ссылок
    """
    string = re.sub(r'http\S+', '', string)  # remove http links
    string = re.sub(r'bit.ly/\S+', '', string)  # rempve bitly links
    string = re.sub(r'www\S+', '', string)  # rempve bitly links
    string = string.strip('[link]')  # remove [links]
    return string


def preprocessing(string, stopwords, stem):
    """
    Простой препроцессинг текста, очистка, лематизация, удаление коротких слов
    """
    string = remove_emoji(string)
    string = remove_links(string)

    # удаление символов "\r\n"
    str_pattern = re.compile('\r\n')
    string = str_pattern.sub(r'', string)

    # очистка текста от символов
    string = re.sub('(((?![а-яА-Я ]).)+)', ' ', string)
    # лематизация
    string = ' '.join([
        re.sub('\\n', '', ' '.join(stem.lemmatize(s))).strip()
        for s in string.split()
    ])
    # удаление слов, короче 3 символов
    string = ' '.join([s for s in string.split() if len(s) > 3])
    # удаление стоп-слова
    string = ' '.join([s for s in string.split() if s not in stopwords])
    return string


def get_clean_text(data, stopwords):
    """
    Получение текста в преобразованной после очистки
    матричном виде, а также модель векторизации
    """
    # Простой препроцессинг текста
    stem = Mystem()
    comments = [preprocessing(x, stopwords, stem) for x in data]
    # Удаление комментов, которые имеют меньше, чем 5 слов
    comments = [y for y in comments if len(y.split()) > 5]
    # common_texts = [i.split(' ') for i in comments]
    return comments


def vectorize_text(data, tfidf):
    """
    Получение матрицы кол-ва слов в комменариях
    Очистка от пустых строк
    """
    # Векторизация
    X_matrix = tfidf.transform(data).toarray()
    # удаление строк в матрице с пустыми значениями
    mask = (np.nan_to_num(X_matrix) != 0).any(axis=1)
    return X_matrix[mask]

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps3/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('text_preprocessing_clustering')

# Загрузка комментариев из файла
comments_file = '/Users/AntroNataExtyl/Desktop/MineMlOps3/data/comments.txt'
with open(comments_file, 'r') as f:
    comments = f.readlines()
comments = [comment.strip() for comment in comments]

stopwords = ['и', 'в', 'на', 'с']  # список стоп-слов

with mlflow.start_run():
    # Препроцессинг комментариев
    cleaned_comments = get_clean_text(comments, stopwords)

    # Векторизация
    tfidf = TfidfVectorizer(min_df=5, max_df=0.95)
    tfidf.fit(cleaned_comments)
    X_matrix = vectorize_text(cleaned_comments, tfidf)

    # Кластеризация с помощью K-средних
    num_clusters = 5  # число кластеров
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_matrix)

    # Получение меток кластеров
    labels = kmeans.labels_

    # Логирование параметров и модели
    mlflow.log_param('model', 'KMeans')
    mlflow.log_param('num_clusters', num_clusters)
    mlflow.sklearn.log_model(kmeans, 'model')

    # Сохранение результатов
    np.savetxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/processed_comments.csv', X_matrix, delimiter=',')
    np.savetxt('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/cluster_labels.csv', labels, delimiter=',')
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/processed_comments.csv', 'processed_data')
    mlflow.log_artifact('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/cluster_labels.csv', 'cluster_data')

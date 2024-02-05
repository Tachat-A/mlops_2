import json
import os
import requests
from pyyoutube import Api
import mlflow
from mlflow.tracking import MlflowClient

def get_data(YOUTUBE_API_KEY, videoId, maxResults, nextPageToken):
    """
    Получение информации со страницы с видео по video id
    """
    YOUTUBE_URI = 'https://www.googleapis.com/youtube/v3/commentThreads?key={KEY}&textFormat=plainText&' + \
                  'part=snippet&videoId={videoId}&maxResults={maxResults}&pageToken={nextPageToken}'
    format_youtube_uri = YOUTUBE_URI.format(KEY=YOUTUBE_API_KEY,
                                            videoId=videoId,
                                            maxResults=maxResults,
                                            nextPageToken=nextPageToken)
    content = requests.get(format_youtube_uri).text
    data = json.loads(content)
    return data


def get_text_of_comment(data):
    """
    Получение комментариев из полученных данных под одним видео
    """
    comms = set()
    for item in data['items']:
        comm = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comms.add(comm)
    return comms

def get_all_comments(YOUTUBE_API_KEY, query, count_video=10, limit=30, maxResults=10, nextPageToken=''):
    api = Api(api_key=YOUTUBE_API_KEY)
    video_by_keywords = api.search_by_keywords(q=query,
                                               search_type=["video"],
                                               count=count_video,
                                               limit=limit)
    videoIds = [x.id.videoId for x in video_by_keywords.items]

    comments_all = []
    for id_video in videoIds:
        try:
            data = get_data(YOUTUBE_API_KEY,
                            id_video,
                            maxResults=maxResults,
                            nextPageToken=nextPageToken)
            comment = list(get_text_of_comment(data))
            comments_all.append(comment)
        except:
            continue
    comments = sum(comments_all, [])
    return comments

# Инициализация MLflow
os.environ['MLFLOW_TRACKING_URI'] = '/Users/AntroNataExtyl/Desktop/MineMlOps3/mlflow'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('get_comments')

# Установить свой ключ API-авторизации
YOUTUBE_API_KEY = 'AIzaSyA0QY_fT02Xlf_ce7jnPhjjmPzlfhWIJ7w'
query = 'машинное обучение'

with mlflow.start_run():
    mlflow.log_param('Query', query)
    mlflow.log_param('Count Video', 10)
    mlflow.log_param('Max Results', 10)

    comments = get_all_comments(YOUTUBE_API_KEY, query)

    # Запись комментариев в файл
    with open('/Users/AntroNataExtyl/Desktop/MineMlOps3/data/comments.txt', 'w') as f:
        for comment in comments:
            f.write(comment + '\n')

    mlflow.log_artifact(local_path='/Users/AntroNataExtyl/Desktop/MineMlOps3/data/comments.txt',
                        artifact_path='comments')

    mlflow.end_run()

#!/usr/bin/python3

import pandas as pd

# Получение данных
file_id = '1JZvbXisuHJdSzqZHk6v_LM4zCWLbiEoC'
url = f'https://drive.google.com/uc?id={file_id}'
train = pd.read_csv(url)

train.to_csv("data/raw/train.csv", index=False)

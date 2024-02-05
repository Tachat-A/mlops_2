import pandas as pd
import pickle
import json
import sys
import os

def load_dataset(file_path):
    # Загрузка датасета
    data = pd.read_csv(file_path)
    return data.iloc[:, 1:], data.iloc[:, 0]

def load_model(model_path):
    # Загрузка модели
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model

def evaluate_model(model, X, y):
    # Оценка модели
    return model.score(X, y)

def save_evaluation_score(score, output_dir):
    # Сохранение результата оценки
    score_file_path = os.path.join(output_dir, "score.json")
    with open(score_file_path, "w") as score_file:
        json.dump({"score": score}, score_file)

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Ошибка в аргументах. Использовать:\n")
        sys.stderr.write("\tpython evaluate.py data-file model\n")
        sys.exit(1)

    data_file, model_file = sys.argv[1], sys.argv[2]
    evaluation_directory = "evaluate"
    os.makedirs(evaluation_directory, exist_ok=True)

    X, y = load_dataset(data_file)
    model = load_model(model_file)
    evaluation_score = evaluate_model(model, X, y)
    save_evaluation_score(evaluation_score, evaluation_directory)

if __name__ == "__main__":
    main()

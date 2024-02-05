import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

def split_dataset(file_path, test_size=0.3, random_state=42):
    # Загрузка и подготовка данных
    data = pd.read_csv(file_path)
    target_column = "Price(euro)"
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Разбиение на обучающую и тестовую выборку
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_datasets(X_train, X_test, y_train, y_test, output_dir):
    # Сохранение датасетов
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Ошибка в аргументах. Использовать:\n")
        sys.stderr.write("\tpython3 split_test_train.py data-file\n")
        sys.exit(1)

    input_file = sys.argv[1]
    output_directory = os.path.join("data", "stage4")
    os.makedirs(output_directory, exist_ok=True)

    X_train, X_test, y_train, y_test = split_dataset(input_file)
    save_datasets(X_train, X_test, y_train, y_test, output_directory)

if __name__ == "__main__":
    main()

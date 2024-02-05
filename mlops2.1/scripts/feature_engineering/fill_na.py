import pandas as pd
import os
import sys

def clean_dataset(file_path):
    # Загрузка датасета
    data = pd.read_csv(file_path)

    # Удаление дубликатов
    data = data.drop_duplicates().reset_index(drop=True)

    # Фильтрация данных по заданным критериям
    filters = [
        (data['Year'] < 2021) & (data['Distance'] < 1100),
        (data['Distance'] > 1e6),
        (data['Engine_capacity(cm3)'] < 200),
        (data['Engine_capacity(cm3)'] > 5000),
        (data['Price(euro)'] < 101),
        (data['Price(euro)'] > 1e5),
        (data['Year'] < 1971)
    ]
    for filter_condition in filters:
        data = data[~filter_condition]

    # Сброс индексов после фильтрации
    data.reset_index(drop=True, inplace=True)

    return data

def fill_missing_values(data):
    # Определение числовых и категориальных колонок
    numeric_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(exclude='number').columns

    # Заполнение пропусков
    data[numeric_cols] = data[numeric_cols].fillna(0)
    data[categorical_cols] = data[categorical_cols].fillna("")

    return data

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Ошибка в аргументах. Использовать:\n")
        sys.stderr.write("\tpython3 fill_na.py data-file\n")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = os.path.join("data", "stage1")
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_data = clean_dataset(input_file)
    final_data = fill_missing_values(cleaned_data)
    final_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)

if __name__ == "__main__":
    main()

import pandas as pd
import os
import sys

def preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Удаление определенных категориальных столбцов
    categorical_columns_to_remove = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    data = remove_categorical_columns(data, categorical_columns_to_remove)

    return data

def remove_categorical_columns(data, columns):
    return data.drop(columns=columns, axis=1)

def main():
    if len(sys.argv) != 2:
        error_message = "Ошибка в аргументах. Использовать:\n"
        error_message += "\tpython3 change_text_to_numeric.py data-file\n"
        sys.stderr.write(error_message)
        sys.exit(1)

    input_file = sys.argv[1]
    output_directory = os.path.join("data", "stage3")
    os.makedirs(output_directory, exist_ok=True)

    processed_data = preprocess_data(input_file)
    processed_data.to_csv(os.path.join(output_directory, "train.csv"), index=False)

if __name__ == "__main__":
    main()

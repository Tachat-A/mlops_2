import pandas as pd
import os
import sys

def process_dataset(file_path):
    # Чтение данных
    data = pd.read_csv(file_path)

    # Расчет новых признаков
    current_year = 2022
    data['Vehicle_Age'] = current_year - data['Year']
    data['Mileage_per_Year'] = data['Distance'] / data['Vehicle_Age']

    # Фильтрация данных
    outlier_indices = data[(data['Mileage_per_Year'] > 50e3) | (data['Mileage_per_Year'] < 100)].index
    data_cleaned = data.drop(outlier_indices).reset_index(drop=True)

    return data_cleaned

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Ошибка в аргументах. Использовать:\n")
        sys.stderr.write("\tpython3 get_features.py data-file\n")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = os.path.join("data", "stage2")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_data = process_dataset(input_file)
    processed_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)

if __name__ == "__main__":
    main()

from datetime import datetime as dt, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'nataantro',
    'start_date': dt(2022, 12, 1),
    'retries': 1,
    'retry_delays': timedelta(minutes=1),
    'depends_on_past': False
}

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='MlOps2_3_pipeline',
    schedule_interval=None,
    max_active_runs=1,
    tags=['mlops', 'ml_pipeline']
) as dag:
    get_data = BashOperator(
        task_id='get_data',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps3/scripts/get_data.py'
    )

    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps3/scripts/preprocess_data.py'
    )

    train_test_split = BashOperator(
        task_id='train_test_split',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps3/scripts/train_test_split.py'
    )
    
    model_training = BashOperator(
        task_id='model_training',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps3/scripts/model_training.py'
    )
    
    model_testing = BashOperator(
        task_id='model_testing',
        bash_command='python3 /Users/AntroNataExtyl/Desktop/MineMlOps3/scripts/model_testing.py'
    )

    # Установка порядка выполнения задач
    get_data >> preprocess_data >> train_test_split >> model_training >> model_testing

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

default_args ={'start_date': datetime(2021,1,1),
                'parallelism': 10}

with DAG(dag_id='taxi-price-pipeline2',
         schedule='0 0 10 * *', # 매월 10일 실행
         default_args=default_args,
         dagrun_timeout = timedelta(minutes=60),
         tags=['spark'],
         catchup=False,
         max_active_tasks=12) as dag:
    
    extract = SparkSubmitOperator(
        application = '/home/ubuntu/working/taxi_analysis2/spark_pipeline/extract_data.py',
        task_id = 'extract',
        conn_id = 'spark_local'
    )

    preprocess = SparkSubmitOperator(
        application='/home/ubuntu/working/taxi_analysis2/spark_pipeline/preprocess.py',
        task_id = 'preprocess',
        conn_id = 'spark_local')

    tune_hyperparameter = SparkSubmitOperator(
      application="/home/ubuntu/working/taxi_analysis2/spark_pipeline/tune_param.py", 
      task_id="tune_hyperparameter", conn_id="spark_local")

    train_model = SparkSubmitOperator(
      application="/home/ubuntu/working/taxi_analysis2/spark_pipeline/train_model.py", 
      task_id="train_model", conn_id="spark_local")
    
    extract >> preprocess >> tune_hyperparameter >> train_model

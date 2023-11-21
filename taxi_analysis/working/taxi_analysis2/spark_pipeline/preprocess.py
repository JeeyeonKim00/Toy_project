from pyspark.sql import SparkSession
from datetime import datetime
from dateutil.relativedelta import relativedelta

MAX_MEMORY = '10g'

spark = SparkSession.builder.appName('taxi_fare_prediction')\
                    .config('spark.executor.memory', MAX_MEMORY)\
                    .config('spark.driver.memory', MAX_MEMORY)\
                    .getOrCreate()


# taxi 파케이 불러오기
two_month_ago = datetime.now().date() - relativedelta(months=2)
taxi_filepath = f'/home/ubuntu/working/taxi_analysis2/data/raw_data/tripdata_{two_month_ago.year}-{two_month_ago.strftime("%m")}.parquet'
taxi_df = spark.read.parquet(f"file:///{taxi_filepath}")
taxi_df.createOrReplaceTempView('trips')

# 데이터 전처리
preprocess_query = f"""
            SELECT 
                passenger_count,
                PULocationID as pickup_location_id,
                DOLocationID as dropoff_location_id,
                trip_distance,
                HOUR(tpep_pickup_datetime) as pickup_time,
                DATE_FORMAT(TO_DATE(tpep_pickup_datetime), 'EEEE') AS day_of_week,
                total_amount
            FROM trips
            WHERE (total_amount BETWEEN 0 AND 5000)
            AND (trip_distance BETWEEN 0 AND 500)
            AND passenger_count < 4
    """

data_df = spark.sql(preprocess_query)

# train, test 데이터 쪼개기
train_df, test_df = data_df.randomSplit([0.8, 0.2], seed=5)

# 최종 데이터 저장하기
data_dir = f'/home/ubuntu/working/taxi_analysis2/data/train_test'

train_df.write.mode('overwrite').format('parquet').save(f'{data_dir}/train/train_{two_month_ago.year}{two_month_ago.strftime("%m")}')
test_df.write.mode('overwrite').format('parquet').save(f'{data_dir}/test/test_{two_month_ago.year}{two_month_ago.strftime("%m")}')
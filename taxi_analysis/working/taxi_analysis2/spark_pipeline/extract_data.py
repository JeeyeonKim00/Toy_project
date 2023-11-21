import os
import requests
from pyspark.sql import SparkSession
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Spark 세션 초기화
spark = SparkSession.builder.master('local').appName('parquet_download').getOrCreate()

# 현재 날짜에서 두 달 전 날짜 기반으로 다운로드 URL 정의
current_date = datetime.now().date()
two_month_ago = current_date - relativedelta(months=2)
parquet_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{two_month_ago.year}-{two_month_ago.strftime('%m')}.parquet"
download_path = "/home/ubuntu/working/taxi_analysis2/data/raw_data"

# 다운로드할 파일 경로 설정
downloaded_file_path = os.path.join(download_path, f"tripdata_{two_month_ago.year}-{two_month_ago.strftime('%m')}.parquet")

# 파일 다운로드
response = requests.get(parquet_url)
with open(downloaded_file_path, 'wb') as f:
    f.write(response.content)
print(f"Parquet 파일이 성공적으로 다운로드되었습니다. 경로: {downloaded_file_path}")

# 다운로드한 Parquet 파일을 Spark DataFrame으로 읽어들이기
spark_df = spark.read.parquet(downloaded_file_path)

spark_df.show(5)
# 세션 종료
spark.stop()
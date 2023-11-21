from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

MAX_MEMORY = '10g'
spark = SparkSession.builder.appName('taxi_fare_prediction')\
                    .config('spark.executor.memory', MAX_MEMORY)\
                    .config('spark.driver.memory', MAX_MEMORY)\
                    .getOrCreate()

# train, test 데이터 불러오기
two_month_ago = datetime.now().date() - relativedelta(months=2)
data_dir = '/home/ubuntu/working/taxi_analysis2/data/train_test'
train_df = spark.read.parquet(f'file:///{data_dir}/train/train_{two_month_ago.year}{two_month_ago.strftime("%m")}')
test_df = spark.read.parquet(f'file:///{data_dir}/test/test_{two_month_ago.year}{two_month_ago.strftime("%m")}')

hyper_df = pd.read_csv(f'{data_dir}/param/hyper_param_{two_month_ago.year}{two_month_ago.strftime("%m")}.csv')
best_alpha = float(hyper_df.iloc[0]['alpha'])
best_reg_param = float(hyper_df.iloc[0]['reg_param'])


# pipeliine 생성
## 전처리 pipeline stage 생성
stages = []

cat_features = ['pickup_location_id','dropoff_location_id','day_of_week']
num_features = ['passenger_count','trip_distance','pickup_time']

for col in cat_features:
    cat_indexer = StringIndexer(inputCol= col, outputCol=col+'_idx').setHandleInvalid('keep')
    oh_encoder = OneHotEncoder(inputCols= [cat_indexer.getOutputCol()], outputCols=[col+'_oh'])
    stages += [cat_indexer, oh_encoder]

for col in num_features:
    num_vec = VectorAssembler(inputCols= [col], outputCol= col+'_vec')
    num_scaler = StandardScaler(inputCol= num_vec.getOutputCol(), outputCol= col+'_scaled')
    stages += [num_vec, num_scaler]

all_features = [cat+'_oh' for cat in cat_features] + [num+'_scaled' for num in num_features]
vec_assembler = VectorAssembler(inputCols= all_features, outputCol='feature_vector')    
stages += [vec_assembler]

# pipeline training
final_pipeline = Pipeline(stages = stages)
fitted_transformer = final_pipeline.fit(train_df)

vec_train_df = fitted_transformer.transform(train_df)
vec_test_df = fitted_transformer.transform(test_df)

# model 생성
lr = LinearRegression(maxIter = 30,
                      solver = 'normal',
                      labelCol = 'total_amount',
                      featuresCol='feature_vector',
                      elasticNetParam=best_alpha,
                      regParam=best_reg_param)

final_model = lr.fit(vec_train_df)
predictions = final_model.transform(vec_test_df)
# predictions.cache()
predictions.select(['trip_distance','day_of_week','total_amount','prediction']).show()

# 평가
print(f'RMSE: {final_model.summary.rootMeanSquaredError:.4f}')
print(f'R2: {final_model.summary.r2:.4f}')

model_dir = '/home/ubuntu/working/taxi_analysis2/data/lr_model'
final_model.write().overwrite().save(f'{model_dir}/lr_model_{two_month_ago.year}{two_month_ago.strftime("%m")}')
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


MAX_MEMORY = '20g'
spark = SparkSession.builder.appName('taxi_fare_prediction')\
                    .config('spark.executor.memory', MAX_MEMORY)\
                    .config('spark.driver.memory', MAX_MEMORY)\
                    .getOrCreate()


# train, test 데이터 불러오기
two_month_ago = datetime.now().date() - relativedelta(months=2)
data_dir = '/home/ubuntu/working/taxi_analysis2/data/train_test'

train_df = spark.read.parquet(f'file:///{data_dir}/train/train_{two_month_ago.year}{two_month_ago.strftime("%m")}')
test_df = spark.read.parquet(f'file:///{data_dir}/test/test_{two_month_ago.year}{two_month_ago.strftime("%m")}')

toy_df = train_df.sample(False, 0.1, seed=1)
# 파티션 수 조정 (메모리 부족 해결위함)
toy_df = toy_df.repartition(4)


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


## 전처리+모델링 pipeline 생성
lr = LinearRegression(maxIter=30,
                      solver='normal',
                      labelCol='total_amount',
                      featuresCol='feature_vector')

cv_stages = stages + [lr]

cv_pipeline = Pipeline(stages = cv_stages)

# 파라미터 튜닝
## ParamGridBuilder 생성
param_grid = ParamGridBuilder()\
                .addGrid(lr.elasticNetParam, [0.1,0.2, 0.3, 0.4, 0.5])\
                .addGrid(lr.regParam, [0.01, 0.02, 0.03, 0.04, 0.05])\
                .build()

## cross validation
cross_val = CrossValidator(
    estimator=cv_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=RegressionEvaluator(labelCol='total_amount'),
    numFolds =5) 


cv_model = cross_val.fit(toy_df)

best_alpha = cv_model.bestModel.stages[-1]._java_obj.getElasticNetParam()
best_reg_param = cv_model.bestModel.stages[-1]._java_obj.getRegParam()

hyper_param = {'alpha': [best_alpha],
               'reg_param': [best_reg_param]}

hyper_df = pd.DataFrame(hyper_param).to_csv(f'{data_dir}/param/hyper_param_{two_month_ago.year}{two_month_ago.strftime("%m")}.csv')
print(hyper_df)

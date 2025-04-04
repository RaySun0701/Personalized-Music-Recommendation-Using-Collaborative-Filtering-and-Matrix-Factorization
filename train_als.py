from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("ALSRecommender").getOrCreate()

df = spark.read.csv("s3://your-bucket/input/ratings.tsv", sep="\t", inferSchema=True).toDF("user", "item", "rating")

(training, test) = df.randomSplit([0.8, 0.2])

als = ALS(
    userCol="user", itemCol="item", ratingCol="rating",
    nonnegative=True, coldStartStrategy="drop", maxIter=10, rank=10, regParam=0.1
)

model = als.fit(training)

predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print(f"Test RMSE = {rmse}")

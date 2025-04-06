# generate_recommendations.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, row_number, broadcast
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("UBCF-GenerateRecommendations").getOrCreate()

# Load data
ratings = spark.read.csv("s3://musicproject6240/processed/ratings_preprocessed.csv", header=True, inferSchema=True)
similarity = spark.read.csv("s3://musicproject6240/processed/user_similarity_matrix_top25.csv", header=True, inferSchema=True)

# Optional optimization: cache and repartition
ratings = ratings.cache()
similarity = similarity.cache()

# Rename columns for join clarity
similarity = similarity.withColumnRenamed("user1", "user_id").withColumnRenamed("user2", "neighbor")

# Join similarity with neighbor ratings
joined = broadcast(similarity).join(ratings.withColumnRenamed("user_id", "neighbor"), on="neighbor") \
    .select("user_id", "item_id", "rating", "similarity")

# Weighted average score
predicted = joined.groupBy("user_id", "item_id").agg(
    expr("sum(rating * similarity) / sum(similarity)").alias("predicted_score")
)

# Remove items already rated
rated = ratings.select("user_id", "item_id")
recommend = predicted.join(rated, on=["user_id", "item_id"], how="left_anti")

# Top-N per user
windowSpec = Window.partitionBy("user_id").orderBy(col("predicted_score").desc())
topN = recommend.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") <= 10)

# Save to S3
topN.select("user_id", "item_id", "predicted_score") \
    .write.csv("s3://musicproject6240/output/user_recommendations.csv", header=True, mode="overwrite")

spark.stop()


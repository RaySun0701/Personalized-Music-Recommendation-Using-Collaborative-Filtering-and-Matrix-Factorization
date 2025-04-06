from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, count, sum as _sum, row_number, broadcast
from pyspark.sql.window import Window

# ------------------------
# Parameters
# ------------------------
MAX_USERS_PER_ITEM = 1000
MIN_USERS_PER_ITEM = 50
TOP_K = 25

# ------------------------
# Start Spark
# ------------------------
spark = SparkSession.builder.appName("UBCF-CenteredCosineSimilarity").getOrCreate()

# ------------------------
# Load Data
# ------------------------
ratings = spark.read.csv(
    "s3://musicproject6240/processed/ratings_preprocessed.csv",
    header=True, inferSchema=True
)

# ------------------------
# Filter by item popularity
# ------------------------
item_counts = ratings.groupBy("item_id").agg(count("*").alias("user_count"))
filtered_items = item_counts.filter(
    (col("user_count") >= MIN_USERS_PER_ITEM) &
    (col("user_count") <= MAX_USERS_PER_ITEM)
)
ratings = ratings.join(broadcast(filtered_items), on="item_id")

# ------------------------
# Compute user average
# ------------------------
user_avg = ratings.groupBy("user_id") \
    .agg(_sum("rating").alias("total_rating"), count("rating").alias("rating_count")) \
    .withColumn("avg_rating", col("total_rating") / col("rating_count")) \
    .select("user_id", "avg_rating")

# ------------------------
# Center ratings
# ------------------------
ratings_centered = ratings.join(user_avg, on="user_id") \
    .withColumn("centered_rating", col("rating") - col("avg_rating")) \
    .select("user_id", "item_id", "centered_rating") \
    .repartition("item_id") \
    .cache()

ratings_centered.count()  # Force caching

# ------------------------
# Join on co-rated items
# ------------------------
joined = ratings_centered.alias("a").join(
    ratings_centered.alias("b"),
    on="item_id"
).filter(col("a.user_id") < col("b.user_id"))

# ------------------------
# Cosine Similarity
# ------------------------
pairwise_scores = joined.withColumn("product", col("a.centered_rating") * col("b.centered_rating")) \
    .groupBy(
        col("a.user_id").alias("user1"),
        col("b.user_id").alias("user2")
    ).agg(
        _sum("product").alias("dot_product"),
        _sum(col("a.centered_rating") ** 2).alias("norm_a"),
        _sum(col("b.centered_rating") ** 2).alias("norm_b")
    )

similarities = pairwise_scores.withColumn(
    "similarity",
    col("dot_product") / (sqrt(col("norm_a")) * sqrt(col("norm_b")))
)

# ------------------------
# Top-K similar users
# ------------------------
windowSpec = Window.partitionBy("user1").orderBy(col("similarity").desc())
topk_similarities = similarities.withColumn("rank", row_number().over(windowSpec)) \
                                .filter(col("rank") <= TOP_K)

# ------------------------
# Save
# ------------------------
topk_similarities.select("user1", "user2", "similarity") \
    .coalesce(1) \
    .write.csv("s3://musicproject6240/processed/user_similarity_matrix.csv", header=True, mode="overwrite")

spark.stop()







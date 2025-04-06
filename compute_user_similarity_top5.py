# compute_user_similarity.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, count, sum as _sum, row_number, broadcast
from pyspark.sql.window import Window

# ------------------------
# Configuration Parameters
# ------------------------
MAX_USERS_PER_ITEM = 1000  # filter items rated by too many users (hot items)
MIN_USERS_PER_ITEM = 50     # optionally filter items rated by too few users
TOP_K = 5                 # top-K most similar users to keep per user

# ------------------------
# Start Spark Session
# ------------------------
spark = SparkSession.builder \
    .appName("UBCF-ComputeUserSimilarity") \
    .getOrCreate()

# ------------------------
# Load Preprocessed Ratings
# ------------------------
ratings = spark.read.csv(
    "s3://musicproject6240/processed/ratings_preprocessed.csv",
    header=True,
    inferSchema=True
)

# ------------------------
# Filter Popular Items with Broadcast Join
# ------------------------
item_counts = ratings.groupBy("item_id").agg(count("*").alias("user_count"))
filtered_items = item_counts.filter(
    (col("user_count") >= MIN_USERS_PER_ITEM) &
    (col("user_count") <= MAX_USERS_PER_ITEM)
)
ratings = ratings.join(broadcast(filtered_items), on="item_id", how="inner")

# Repartition and Cache to Optimize Join Performance
ratings = ratings.repartition("item_id").cache()
ratings.count()  # Force caching

# ------------------------
# Generate User-User Pairs (Co-rated Items)
# ------------------------
joined = ratings.alias("a").join(
    ratings.alias("b"),
    on="item_id"
).filter(col("a.user_id") < col("b.user_id"))  # avoid duplicate & self-pairs

# ------------------------
# Compute Cosine Similarity Components
# ------------------------
pairwise_scores = joined.withColumn("product", col("a.rating") * col("b.rating")) \
    .groupBy("a.user_id", "b.user_id") \
    .agg(
        _sum("product").alias("dot_product"),
        _sum(col("a.rating") ** 2).alias("norm_a"),
        _sum(col("b.rating") ** 2).alias("norm_b")
    )

# ------------------------
# Calculate Cosine Similarity
# ------------------------
similarities = pairwise_scores.withColumn(
    "similarity",
    col("dot_product") / (sqrt(col("norm_a")) * sqrt(col("norm_b")))
)

# ------------------------
# Top-K Filtering Per User
# ------------------------
windowSpec = Window.partitionBy("a.user_id").orderBy(col("similarity").desc())
topk_similarities = similarities.withColumn("rank", row_number().over(windowSpec)) \
                                .filter(col("rank") <= TOP_K)

# ------------------------
# Save Output to S3
# ------------------------
topk_similarities.selectExpr(
    "a.user_id as user1",
    "b.user_id as user2",
    "similarity"
).coalesce(1) \
 .write.csv("s3://musicproject6240/processed/user_similarity_matrix.csv", header=True, mode="overwrite")

# ------------------------
# Shutdown
# ------------------------
spark.stop()
# Personalized-Music-Recommendation-Using-Collaborative-Filtering-and-Matrix-Factorization

## **💡 Implementation Steps**

This project consists of **five main modules**:

1. **Data Preparation & Preprocessing**
2. **Data Transformation & Similarity Computation using MapReduce**
3. **Matrix Factorization using Spark ALS**
4. **Model Evaluation & Optimization**
5. **Scalability Testing & Performance Analysis**

------

## **📌 1. Data Preparation & Preprocessing**

### **1.1 Acquire and Understand the Data**

Use the **Yahoo! Music KDDCup 2011 (Track 1) dataset**, which has the following format:

The dataset comprises 262,810,175 ratings of 624,961 music items by 1,000,990 users collected during **1999-2010**.

First line for a user is formatted as:  <UsedId>|<UserRatings>.
Each of the next <#UserRatings> lines describes a single rating by <UsedId>, sorted in chronological order.
Rating line format is:  <ItemId>|<Score>|<RatingTime>|<TimeStamp>.
The scores are integers lying between 0 and 100.

```
0|2
550452   |   90   |   5229   |  10:30:00
323933   |   100  |   5802   |  11:05:00
```

- Dataset Characteristics:
  - **262M records**, making it a large-scale dataset.
  - **Ratings range from 0-100** (needs normalization to 0-1).
  - **Timestamp** can be used to analyze user behavior trends.

### **1.2 Preprocessing Steps**

**Data Cleaning using PySpark**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp

spark = SparkSession.builder.appName("MusicRecSys").getOrCreate()

# Load data
df = spark.read.csv("yahoo_music_kddcup2011.csv", header=True, inferSchema=True)

# Convert data format
df = df.withColumn("Rating", col("Rating") / 100.0)  # Normalize ratings to [0,1]
df = df.withColumn("Timestamp", unix_timestamp("Timestamp"))  # Convert timestamps
df = df.dropDuplicates()  # Remove duplicates

# Filter out users with fewer than 5 interactions
df_filtered = df.groupBy("UserID").count().filter(col("count") > 5).drop("count")

df_filtered.show(5)
```

- Data Cleaning Steps:
  - **Normalize ratings** (0-100 → 0-1)
  - **Convert time format** to Unix timestamps

------

## **📌 2. Similarity Computation using MapReduce**

### **2.1 Compute User Similarity**

Use **MapReduce** to compute **user-user similarity** using **Cosine Similarity**:
$$
\text{Similarity}(u,v) = \frac{\sum_i r_{ui} \cdot r_{vi}}{\sqrt{\sum_i r_{ui}^2} \cdot \sqrt{\sum_i r_{vi}^2}}
$$

#### **MapReduce Implementation**

**Mapper (Python)**

```python
import sys

for line in sys.stdin:
    user, item, rating = line.strip().split(",")
    print(f"{item}\t{user}:{rating}")
```

**Reducer (Python)**

```python
import sys
from collections import defaultdict
import math

user_ratings = defaultdict(dict)

# Read data
for line in sys.stdin:
    item, user_rating = line.strip().split("\t")
    user, rating = user_rating.split(":")
    user_ratings[user][item] = float(rating)

# Compute Cosine Similarity
users = list(user_ratings.keys())
for i in range(len(users)):
    for j in range(i+1, len(users)):
        user1, user2 = users[i], users[j]
        common_items = set(user_ratings[user1].keys()) & set(user_ratings[user2].keys())
        
        if not common_items:
            continue
        
        dot_product = sum(user_ratings[user1][item] * user_ratings[user2][item] for item in common_items)
        norm1 = math.sqrt(sum(r**2 for r in user_ratings[user1].values()))
        norm2 = math.sqrt(sum(r**2 for r in user_ratings[user2].values()))
        
        similarity = dot_product / (norm1 * norm2)
        print(f"{user1}\t{user2}\t{similarity}")
```

- **Input Format**: `UserID, ItemID, Rating`
- **Output Format**: `User1 \t User2 \t Similarity`

------

## **📌 3. Matrix Factorization using Spark ALS**

### **3.1 Train the Model**

ALS (Alternating Least Squares) is a matrix factorization technique used for rating prediction:

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Train ALS model
als = ALS(
    userCol="UserID", 
    itemCol="ItemID", 
    ratingCol="Rating", 
    rank=10, 
    maxIter=10, 
    regParam=0.1,
    coldStartStrategy="drop"
)

model = als.fit(df_filtered)

# Predictions
predictions = model.transform(df_filtered)
predictions.show(5)
```

------

## **📌 4. Model Evaluation**

Use the following evaluation metrics:

- **RMSE (Root Mean Squared Error)** → Measures rating prediction error.
- **Precision@K / Recall@K** → Measures top-K recommendation quality.
- **NDCG (Normalized Discounted Cumulative Gain)** → Evaluates ranking effectiveness.

```python
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Square Error (RMSE): {rmse}")
```

------

## **📌 5. Scalability Testing**

1. Increase Data Size

   - Run the ALS model with increasing dataset size to observe execution time.

   - Monitor Spark job execution performance:

     ```python
     df_large = df_filtered.union(df_filtered)  # Duplicate data for scalability testing
     model_large = als.fit(df_large)
     ```

2. Test with Different Spark Cluster Sizes

   - Run on AWS or a local Spark cluster with 1/4/8 worker nodes.

   - Record execution time:

     ```bash
     time spark-submit music_recommendation.py
     ```

------

## **✅ Summary**

### **💡 Implement the Project as Follows**

| **Module**                           | **Implementation Details**                                |
| ------------------------------------ | --------------------------------------------------------- |
| **Data Preprocessing**               | Normalize ratings, remove duplicates, process timestamps  |
| **MapReduce Similarity Computation** | Compute user-user similarity (Cosine Similarity)          |
| **Spark ALS Training**               | Predict user ratings                                      |
| **Model Evaluation**                 | RMSE, Precision@K, Recall@K, NDCG                         |
| **Scalability Testing**              | Increase data size, test with different compute resources |


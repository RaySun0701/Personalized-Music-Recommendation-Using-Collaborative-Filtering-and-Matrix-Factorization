#!/bin/bash

# Exit immediately on error
set -e

# Optional: Print debug output
echo "Starting Spark job: compute_user_similarity.py"

# Submit Spark job
spark-submit \
  --deploy-mode cluster \
  --master yarn \
  --conf spark.executor.memory=4g \
  --conf spark.driver.memory=4g \
  --conf spark.executor.cores=2 \
  --conf spark.sql.shuffle.partitions=200 \
  s3://musicproject6240/scripts/compute_user_similarity.py

echo "Spark job completed."

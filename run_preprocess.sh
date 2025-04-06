#!/bin/bash

# Download script and data
aws s3 cp s3://musicproject6240/scripts/preprocess_data.py .
aws s3 cp s3://musicproject6240/data/trainIdx1.txt .

# Run preprocessing
python3 preprocess_data.py trainIdx1.txt ratings_preprocessed.csv

# Upload result
aws s3 cp ratings_preprocessed.csv s3://musicproject6240/processed/

# preprocess_data.py

import re
import csv

"""
Convert raw rating logs into Spark-friendly CSV format.
Input: s3://musicproject6240/data/trainIdx1.txt
Output: s3://musicproject6240/processed/ratings_preprocessed.csv
"""

INPUT_FILE = "trainIdx1.txt"
OUTPUT_FILE = "ratings_preprocessed.csv"

def parse_raw_lines(raw_lines):
    i = 0
    while i < len(raw_lines):
        if "|" not in raw_lines[i]:
            i += 1
            continue

        try:
            user_id, num_ratings = raw_lines[i].strip().split("|")
            num_ratings = int(num_ratings)
        except ValueError:
            i += 1
            continue

        i += 1
        for _ in range(num_ratings):
            if i >= len(raw_lines):
                break
            fields = re.split(r"\s+", raw_lines[i].strip())
            if len(fields) >= 2:
                item_id, rating = fields[0], fields[1]
                yield user_id, item_id, rating
            i += 1

if __name__ == "__main__":
    with open(INPUT_FILE, "r") as fin:
        lines = fin.readlines()

    with open(OUTPUT_FILE, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id", "rating"])
        for row in parse_raw_lines(lines):
            writer.writerow(row)


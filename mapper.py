#!/usr/bin/env python3
import sys

lines = [line.strip() for line in sys.stdin if line.strip()]
i = 0

while i < len(lines):
    if "|" not in lines[i]:
        i += 1
        continue

    parts = lines[i].split("|")
    if len(parts) != 2:
        i += 1
        continue

    user_id = parts[0].strip()
    try:
        num_ratings = int(parts[1].strip())
    except ValueError:
        i += 1
        continue

    i += 1
    for _ in range(num_ratings):
        if i >= len(lines):
            break
        rating_parts = lines[i].split()
        if len(rating_parts) >= 2:
            item_id = rating_parts[0].strip()
            rating = rating_parts[1].strip()
            print(f"{item_id}\t{user_id}:{rating}")
        i += 1


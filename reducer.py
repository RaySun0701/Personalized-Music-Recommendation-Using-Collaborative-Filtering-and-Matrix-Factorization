#!/usr/bin/env python3
import sys
from collections import defaultdict
import math

item_users = defaultdict(list)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        item_id, user_rating = line.split("\t")
        user_id, rating = user_rating.split(":")
        item_users[item_id].append((user_id, int(rating)))
    except:
        continue  # 忽略格式错误

# key: (user1, user2) → [dot, norm1, norm2]
user_pair_stats = defaultdict(lambda: [0, 0, 0])

for ratings in item_users.values():
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            u1, r1 = ratings[i]
            u2, r2 = ratings[j]
            if u1 == u2:
                continue
            uid1, uid2 = sorted([u1, u2])
            stats = user_pair_stats[(uid1, uid2)]
            stats[0] += r1 * r2
            stats[1] += r1 ** 2
            stats[2] += r2 ** 2

for (u1, u2), (dot, norm1, norm2) in user_pair_stats.items():
    if norm1 > 0 and norm2 > 0:
        similarity = dot / (math.sqrt(norm1) * math.sqrt(norm2))
        print(f"{u1}\t{u2}\t{similarity:.4f}")


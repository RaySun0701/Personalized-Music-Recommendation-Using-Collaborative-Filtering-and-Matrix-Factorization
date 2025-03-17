## **Compute User-User Similarity Using Cosine Similarity**

## **📌 Theory: Cosine Similarity**

### **What is Cosine Similarity?**

Cosine similarity measures the angle between two user rating vectors. It quantifies how similar two users are based on their **ratings for the same items**.

### **Mathematical Formula**

The cosine similarity between user **u** and user **v** is defined as:

$$
\text{Similarity}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} \cdot r_{vi}}{\sqrt{\sum_{i \in I_u} r_{ui}^2} \cdot \sqrt{\sum_{i \in I_v} r_{vi}^2}}
$$

Where:

- rᵤᵢ = rating given by user **u** for item **i**
- rᵥᵢ = rating given by user **v** for item **i**
- Iᵤᵥ = set of items rated by both **u** and **v**
- Iᵤ = set of items rated by **u**
- Iᵥ = set of items rated by **v**

**Key points:**

- If two users have rated many **common items**, their cosine similarity will be high.
- The **denominator** normalizes the ratings to avoid bias toward users who give uniformly high ratings.

------

## **📌 Step 1: Process the Input Data**

Your dataset format:

```
0|2
550452   |   90   |   5229   |  10:30:00
323933   |   100  |   5802   |  11:05:00
1|3
159248   |   100  |   5802   |  11:27:00
554099   |   100  |   5815   |  16:02:00
70896    |   100  |   5815   |  16:26:00
```

This means:

- User **0** has **2 ratings** (for items `550452` and `323933`).
- User **1** has **3 ratings** (for items `159248`, `554099`, and `70896`).

### **Convert to a Dictionary**

To compute similarity, store the data in a dictionary where:

- **Key = UserID**
- **Value = Dictionary of (ItemID: Rating)**

```python
from collections import defaultdict

def parse_data(file_path):
    user_ratings = defaultdict(dict)  # {user_id: {item_id: rating, item_id2: rating2}}

    with open(file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            # Read user ID and number of ratings
            user_id, num_ratings = map(int, lines[i].strip().split("|"))
            i += 1  # Move to the next line
            
            # Read ratings
            for _ in range(num_ratings):
                item_id, score, _, _ = lines[i].strip().split("|")
                user_ratings[user_id][int(item_id)] = int(score)  # Store as integer
                i += 1  # Move to next line

    return user_ratings

file_path = "yahoo_music.txt"
user_ratings = parse_data(file_path)
print(user_ratings)  # Example output
```

**Output:**

```python
{
    0: {550452: 90, 323933: 100},
    1: {159248: 100, 554099: 100, 70896: 100}
}
```

------

## **📌 Step 2: Compute Cosine Similarity**

To compute similarity, follow these steps:

1. **Find common items** between two users.
2. **Compute the numerator** (dot product of ratings).
3. **Compute the denominator** (magnitude of both vectors).
4. **Compute cosine similarity** using the formula.

### **Python Implementation**

```python
import math
from itertools import combinations

def cosine_similarity(user_ratings):
    user_similarities = {}  # Store similarity between user pairs

    users = list(user_ratings.keys())

    # Iterate over all pairs of users
    for user1, user2 in combinations(users, 2):
        items_u1 = user_ratings[user1]
        items_u2 = user_ratings[user2]

        # Find common items
        common_items = set(items_u1.keys()) & set(items_u2.keys())
        if not common_items:
            continue  # Skip if no common items

        # Compute dot product (numerator)
        dot_product = sum(items_u1[item] * items_u2[item] for item in common_items)

        # Compute magnitude (denominator)
        norm_u1 = math.sqrt(sum(r ** 2 for r in items_u1.values()))
        norm_u2 = math.sqrt(sum(r ** 2 for r in items_u2.values()))

        # Compute similarity
        similarity = dot_product / (norm_u1 * norm_u2)
        
        # Store similarity score
        user_similarities[(user1, user2)] = similarity

    return user_similarities

# Compute user-user similarity
user_sim = cosine_similarity(user_ratings)

# Print results
for pair, similarity in user_sim.items():
    print(f"Similarity between User {pair[0]} and User {pair[1]}: {similarity:.4f}")
```

------

## **📌 Step 3: MapReduce Implementation**

To handle large-scale data, implement **MapReduce**.

### **Mapper**

- Reads user data and emits **(item, (user, rating))** pairs.

```python
import sys

for line in sys.stdin:
    line = line.strip()
    
    if "|" in line:
        parts = line.split("|")
        
        # If it's a user line
        if len(parts) == 2:
            user_id = parts[0]
            num_ratings = int(parts[1])
            continue
        
        # If it's a rating line
        item_id, score, _, _ = parts
        print(f"{item_id}\t{user_id}:{score}")  # Emit item as key
```

**Example Output:**

```
550452	0:90
323933	0:100
159248	1:100
554099	1:100
70896	1:100
```

### **Reducer**

- Groups ratings by **ItemID** and calculates cosine similarity.

```python
import sys
from collections import defaultdict
import math

user_ratings = defaultdict(dict)

# Read input from Mapper
for line in sys.stdin:
    item_id, user_rating = line.strip().split("\t")
    user, rating = user_rating.split(":")
    user_ratings[user][int(item_id)] = int(rating)

users = list(user_ratings.keys())

# Compute cosine similarity
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

------

## **✅ Summary**

| Step                                 | Description                                                 |
| ------------------------------------ | ----------------------------------------------------------- |
| **1. Data Parsing**                  | Convert raw text data into a structured dictionary.         |
| **2. Cosine Similarity Computation** | Compute similarity based on shared item ratings.            |
| **3. MapReduce Implementation**      | Distribute similarity computation across multiple machines. |


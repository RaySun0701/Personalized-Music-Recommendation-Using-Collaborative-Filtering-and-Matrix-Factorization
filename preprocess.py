def preprocess(input_path, output_path):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        lines = fin.readlines()
        i = 0
        while i < len(lines):
            if "|" not in lines[i]:
                i += 1
                continue
            user_id, num = lines[i].strip().split("|")
            user_id = int(user_id)
            i += 1
            for _ in range(int(num)):
                item_id, rating, *_ = lines[i].strip().split()
                fout.write(f"{user_id}\t{item_id}\t{rating}\n")
                i += 1


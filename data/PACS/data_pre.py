from collections import Counter
import collections
import pickle
import os
import numpy as np
files = ["art_painting", "cartoon", "photo", "sketch"]

for file in files:
    count = collections.defaultdict(int)
    for _, dirs, _ in os.walk(file):
        for dir_ in dirs:
            path = os.path.join(file, dir_)
            for _, _, file_num in os.walk(path):
                count[dir_] += len(file_num)
    print(count)

# The training set and test set are split in an 8:2 ratio
for file in files:
    train_data = []
    test_data = []
    for _, dirs, _ in os.walk(file):
        for dir_ in dirs:
            path = os.path.join(file, dir_)
            for _, _, file_num in os.walk(path):
                sample_num = len(file_num)
                train_len = int(sample_num * 0.8)
                train_data.extend((os.path.join(path, image_path), dir_) for image_path in file_num[:train_len])
                test_data.extend((os.path.join(path, image_path), dir_) for image_path in file_num[train_len:])
    with open(f"{file}_train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(f"{file}_test.pkl", "wb") as f:
        pickle.dump(test_data, f)


for file in files:
    for mode in ["train",]:
        with open(f"{file}_{mode}.pkl", "rb") as f:
            data = pickle.load(f)
        print(Counter(np.array(data)[:,1]))
                

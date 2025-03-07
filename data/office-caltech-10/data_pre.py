from collections import Counter
import collections
import pickle
import os
import numpy as np
files = ["amazon", "caltech", "dslr", "webcam"]
# The training set and test set are split in an 8:2 ratio.

# for file in files:
#     count = collections.defaultdict(int)
#     total = collections.defaultdict(int)
#     for _, dirs, _ in os.walk(file):
#         for dir_ in dirs:
#             path = os.path.join(file, dir_)
#             for _, _, file_num in os.walk(path):
#                 count[dir_] += len(file_num)
#                 total[file] += len(file_num)
#     for _, dirs, _ in os.walk(file):
#         for dir_ in dirs:
#             count[dir_] /= total[file]

# for file in files:
#     dirs = os.listdir(file)
#     train_data = []
#     test_data = []
#     total_len = []
#     for dir_ in dirs:
#         path = os.path.join(file, dir_)
#         for _, _, file_num in os.walk(path):
#             train_len = round(len(file_num) * 0.8)
#             train_data.extend([(os.path.join(path, item), dir_) for item in file_num[:train_len]])
#             test_data.extend([(os.path.join(path, item), dir_) for item in file_num[train_len:]])
#     with open(f"{file}_train_new.pkl", "wb") as f:
#         pickle.dump(train_data, f)
#     with open(f"{file}_test_new.pkl", "wb") as f:
#         pickle.dump(test_data, f)

    
for file in files:
    for mode in ["train"]:
        with open(f"{file}_{mode}_new.pkl", "rb") as f:
            data = pickle.load(f)
        for key, count in Counter(np.array(data)[:,1]).items():
            print(file, key, count, count / len(data), len(data))
        print("---------------------------------------------")

# from PIL import Image


# files = ["amazon", "caltech", "dslr", "webcam"]
# for file in files:
#     for mode in ["train"]:
#         with open(f"{file}_{mode}_new.pkl", "rb") as f:
#             data = pickle.load(f)
#         # print(data[0])
#         # img = Image.open(data[100][0])
#         # width, height = image.size
# #         for y in range(height):
# #             for x in range(width):
# #                 r, g, b = image.getpixel((x, y))
# #                 print(f'Pixel at ({x}, {y}): R={r}, G={g}, B={b}')
# #         print("---------------------------------------------")



































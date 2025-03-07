import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import pickle as pkl
import scipy.io as scio
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from collections import  Counter
# import requests
# from tqdm import tqdm
# import zipfile
import collections
import pickle
                                
def stratified_split(X,y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('Train:', Counter(y_train))
        print('Test:', Counter(y_test))

    return (X_train, y_train), (X_test, y_test)

def process_mnist():
    """
    train:
    (56000, 28, 28)
    (56000,)
    test:
    (14000, 28, 28)
    (14000,)
    """
    mnist_train = 'MNIST/training.pt'
    mnist_test = 'MNIST/test.pt'
    train = torch.load(mnist_train)
    test = torch.load(mnist_test)

    train_img = train[0].numpy()
    train_tar = train[1].numpy()

    test_img = test[0].numpy()
    test_tar = test[1].numpy()

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])


    train_stratified ,test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('MNIST/train.pkl', 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('MNIST/test.pkl', 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)

def process_svhn():
    """
    train:
    (79431, 32, 32, 3)
    (79431,)
    test:
    (19858, 32, 32, 3)
    (19858,)
    """
    train = scio.loadmat('SVHN/train_32x32.mat')
    test = scio.loadmat('SVHN/test_32x32.mat')

    train_img = train['X']
    train_tar = train['y'].astype(np.int64).squeeze()

    test_img = test['X']
    test_tar = test['y'].astype(np.int64).squeeze()

    train_img = np.transpose(train_img, (3, 0, 1, 2))
    test_img = np.transpose(test_img, (3, 0, 1, 2))

    np.place(train_tar, train_tar == 10, 0)
    np.place(test_tar, test_tar == 10, 0)

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('SVHN/train.pkl', 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('SVHN/test.pkl', 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)

def process_usps():
    """
    train:
    (7438, 16, 16)
    (7438,)
    test:
    (1860, 16, 16)
    (1860,)
    :return:
    """
    import bz2
    train_path = 'USPS/usps.bz2'
    with bz2.open(train_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
    imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
    imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
    imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
    targets = [int(d[0]) - 1 for d in raw_data]

    train_img = imgs
    train_tar = np.array(targets)

    test_path = 'USPS/usps.t.bz2'
    with bz2.open(test_path) as fp:
        raw_data = [l.decode().split() for l in fp.readlines()]
    imgs = [[x.split(':')[-1] for x in data[1:]] for data in raw_data]
    imgs = np.asarray(imgs, dtype=np.float32).reshape((-1, 16, 16))
    imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
    targets = [int(d[0]) - 1 for d in raw_data]

    test_img = imgs
    test_tar = np.array(targets)

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('USPS/train.pkl', 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)
    
    with open('USPS/test.pkl', 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)

def process_synth():
    """
    (391162, 32, 32, 3)
    (391162,)
    (97791, 32, 32, 3)
    (97791,)
    """
    train = scio.loadmat('SynthDigits/synth_train_32x32.mat')
    test = scio.loadmat('SynthDigits/synth_test_32x32.mat')

    train_img = train['X']
    train_tar = train['y'].astype(np.int64).squeeze()

    test_img = test['X']
    test_tar = test['y'].astype(np.int64).squeeze()

    train_img = np.transpose(train_img, (3,0,1,2))
    test_img = np.transpose(test_img, (3,0,1,2))

    all_img = np.concatenate([train_img, test_img])
    all_tar = np.concatenate([train_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)


    with open('SynthDigits/train.pkl', 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('SynthDigits/test.pkl', 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)

def process_mnistm():
    """
    (56000, 28, 28, 3)
    (56000,)
    (14000, 28, 28, 3)
    (14000,)
    :return:
    """
    data = np.load('MNIST_M/mnistm_data.pkl', allow_pickle=True)
    train_img = data['train']
    train_tar = data['train_label']
    valid_img = data['valid']
    valid_tar = data['valid_label']
    test_img = data['test']
    test_tar = data['test_label']

    all_img = np.concatenate([train_img, valid_img, test_img])
    all_tar = np.concatenate([train_tar, valid_tar, test_tar])

    train_stratified, test_stratified = stratified_split(all_img, all_tar)
    print('# After spliting:')
    print('Train imgs:\t', train_stratified[0].shape)
    print('Train labels:\t', train_stratified[1].shape)
    print('Test imgs:\t', test_stratified[0].shape)
    print('Test labels:\t', test_stratified[1].shape)

    with open('MNIST_M/train.pkl', 'wb') as f:
        pkl.dump(train_stratified, f, pkl.HIGHEST_PROTOCOL)

    with open('MNIST_M/test.pkl', 'wb') as f:
        pkl.dump(test_stratified, f, pkl.HIGHEST_PROTOCOL)

def split(data_path, percentage=0.1):
    """
    split each single dataset into multiple partitions for client scaling training
    """
    images, labels = np.load(os.path.join(data_path, 'train.pkl'), allow_pickle=True)
    part_num = int(1./percentage)
    total_size = len(images)
    size_per_partition = total_size // part_num

    for num in range(part_num):
        start_idx = num * size_per_partition
        end_idx = start_idx + size_per_partition
        images_part = images[start_idx:end_idx]
        labels_part = labels[start_idx:end_idx]
        save_path = os.path.join(data_path, 'partitions')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'train_part{}.pkl'.format(num)), 'wb') as f:
            pkl.dump((images_part, labels_part), f, pkl.HIGHEST_PROTOCOL)
    

if __name__  == '__main__':
    # print('Processing...')        
    # print('--------MNIST---------')
    # process_mnist()
    # print('--------SVHN---------')
    # process_svhn()
    # print('--------USPS---------')
    # process_usps()
    # print('--------SynthDigits---------')
    # process_synth()
    # print('--------MNIST-M---------')
    # process_mnistm()    
    base_paths = [
        'MNIST',
        'SVHN',
        'USPS',
        'SynthDigits',
        'MNIST_M'
    ]
    for path in base_paths:
          split(path)
    for data in base_paths:
        with open(f"{data}/partitions/train_part{0}.pkl", "rb") as f:
            t = pickle.load(f)
        print(Counter(t[1]))


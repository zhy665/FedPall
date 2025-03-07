import copy
import torch
from torchvision import datasets, transforms
import numpy as np
import data_utils
import seaborn as sns
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import pickle
from torch.optim import Optimizer
import torch.nn.functional as F
from collections import Counter
import functools
from torch.utils.data.dataset import Dataset

class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, global_protos=None, mask=None):
        """Compute contrastive loss between feature and global prototype
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_feature = torch.zeros_like(contrast_feature)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # generate anchor_feature
        for i in range(batch_size*anchor_count):
            anchor_feature[i, :] = global_protos[labels[i%batch_size].item()]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MixupDataset_norm(Dataset):
    def __init__(self, mean, fs_all: list, args):
        self.data = []
        self.labels = []
        self.means = mean
        self.num_classes = args.num_classes
        self.device = args.device
        self.crt_feat_num = args.crt_feat_num
        self.fs_all = fs_all
        self.fs_len = len(fs_all)
        self.args = args

        self.__mixup_syn_feat_pure_rand_norm__()

    def __mixup_syn_feat_pure_rand_norm__(self):
        num = self.crt_feat_num
        l = self.args.uniform_left
        r_arg = self.args.uniform_right - l
        for cls in range(self.num_classes):
            fs_shuffle_idx = torch.randperm(self.fs_len)
            for i in range(num):
                lam = np.round(l + r_arg * np.random.random(), 2)
                neg_f = self.fs_all[fs_shuffle_idx[i]]
                mixup_f = lam * self.means[cls] + (1 - lam) * F.normalize(neg_f.view(1, -1), dim=1).view(-1)
                self.data.append(mixup_f)
            self.labels += [cls]*num
        self.data = torch.stack(self.data).to(self.device)
        self.labels = torch.tensor(self.labels).long().to(self.device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]


def generate_random_tensor(size, p):
    assert 0 <= p <= 1, "p should be between 0 and 1"
    num_ones = int(size * p)
    tensor = torch.cat((torch.ones(num_ones), torch.zeros(size - num_ones)))
    tensor = tensor[torch.randperm(tensor.size(0))]
    return tensor.view(size)

def compute_num_list(args, train_loader_list):
    num_list = [Counter() for idx in range(args.num_users)]
    for idx in range(args.num_users):
        train_set = iter(train_loader_list[idx])
        for batch_idx in range(len(train_set)):
            images, labels = next(train_set)
            images, labels = images.to(args.device).float(), labels.to(args.device).long()
            for label in labels:
                if label.item() not in num_list[idx]:
                    num_list[idx][label.item()] = 1
                else:
                    num_list[idx][label.item()]+=1
    return num_list

def model_fusion(list_dicts_local_params: list, list_nums_local_data: list):
    # fedavg
    local_params = copy.deepcopy(list_dicts_local_params[0])
    for name_param in list_dicts_local_params[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        local_params[name_param] = value_global_param
    return local_params

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(args.num_users):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(args.num_users):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.mode.lower() == 'fedrep':
            for key in server_model.features.state_dict().keys():
                temp = torch.zeros_like(server_model.features.state_dict()[key], dtype=torch.float32)
                for client_idx in range(args.num_users):
                    temp += client_weights[client_idx] * models[client_idx].features.state_dict()[key]
                server_model.features.state_dict()[key].data.copy_(temp)
                for client_idx in range(args.num_users):
                    models[client_idx].features.state_dict()[key].data.copy_(server_model.features.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return copy.deepcopy(server_model), copy.deepcopy(models)

def get_cls_ratio(args, num_list):
    total_count = sum(num_list.values())
    proportion_list = {key:num / total_count for key, num in num_list.items()}
    return proportion_list

def cal_norm_mean(args, c_means, c_dis):
    glo_means = dict()
    c_dis_temp = torch.ones((args.num_users, args.num_classes))
    for idx in range(args.num_users):
        for key, value in c_dis[idx].items():
            c_dis_temp[idx][key] = value
    c_dis = c_dis_temp.to(args.device)
    total_num_per_cls = c_dis.sum(dim=0)
    for i in range(args.num_classes):
        for c_idx, c_mean in enumerate(c_means):
            if i not in c_mean.keys():
                continue
            temp = glo_means.get(i, 0)
            # normalize the local prototypes, send the direction to the server
            glo_means[i] = temp + \
                F.normalize(c_mean[i].view(1, -1),
                            dim=1).view(-1) * c_dis[c_idx][i]
        if glo_means.get(i) == None:
            continue
        t = glo_means[i]
        glo_means[i] = t / total_num_per_cls[i]
    return glo_means

def proto_aggregation(local_protos_list, num_list):
    agg_protos_label = dict()
    total_num_list = functools.reduce(lambda x, y: x + y, num_list)
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
    averaged_protos = {}
    for label, proto_list in agg_protos_label.items():
        for idx, proto in enumerate(proto_list):
            if label not in averaged_protos:
                averaged_protos[label] = proto * num_list[idx][label] / total_num_list[label]
            else:
                averaged_protos[label] += proto * num_list[idx][label] / total_num_list[label]
    return averaged_protos

def get_mean(args, proto_list, nums):
    weighted_protos = {}
    total_counts = {}
    for idx, proto in enumerate(proto_list):
        for cls, p in proto.items():
            if cls not in weighted_protos:
                weighted_protos[cls] = torch.zeros_like(p) 
                total_counts[cls] = 0
            weighted_protos[cls] += F.normalize(p, dim=0) * nums[idx][cls]
            total_counts[cls] += nums[idx][cls]
    global_protos = {}
    for cls in weighted_protos.keys():
        if total_counts[cls] > 0:
            global_protos[cls] = weighted_protos[cls] / total_counts[cls]
    return global_protos

class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])

def prepare_data_digit_feature_noniid(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_svhn = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_usps = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_synth = transforms.Compose([
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    if args.num_users == 5:
        # MNIST
        mnist_trainset =data_utils.DigitsDataset(data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
        mnist_testset = data_utils.DigitsDataset(data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
        # SVHN
        svhn_trainset = data_utils.DigitsDataset(data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
        svhn_testset = data_utils.DigitsDataset(data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
        # USPS
        usps_trainset = data_utils.DigitsDataset(data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
        usps_testset = data_utils.DigitsDataset(data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
        # Synth Digits
        synth_trainset = data_utils.DigitsDataset(data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
        synth_testset = data_utils.DigitsDataset(data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
        # MNIST-M
        mnistm_trainset = data_utils.DigitsDataset(data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
        mnistm_testset = data_utils.DigitsDataset(data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
        
        mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True, num_workers=args.number_workers, pin_memory=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

        train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
        test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
    else:
        train_loaders, test_loaders = [], []
        for idx in range(args.num_users // 5):
            # MNIST
            mnist_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist, noise=args.noise)
            # SVHN
            svhn_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn, noise=args.noise)
            # USPS
            usps_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps, noise=args.noise)
            # Synth Digits
            synth_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth, noise=args.noise)
            # MNIST-M
            mnistm_trainset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm, noise=args.noise)
            mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
            svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
            usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
            synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
            mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
            train_loaders.extend([mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader])
        
        mnist_testset = data_utils.DigitsDataset_mul_clients(idx, data_path="../data/Digits/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist, noise=args.noise)
        svhn_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn, noise=args.noise)
        usps_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps, noise=args.noise)
        synth_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth, noise=args.noise)
        mnistm_testset = data_utils.DigitsDataset_mul_clients(idx, data_path='../data/Digits/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm, noise=args.noise)
        mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
        
        test_loaders.extend([mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader])
    return train_loaders, test_loaders

def prepare_data_office_feature_noniid(args):
    data_base_path = '../data/office_caltech_10'
    transform_office = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.ToTensor(),
    ])
    
    # amazon
    amazon_trainset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True)
    
    train_loaders = [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader]
    test_loaders = [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader]
    return train_loaders, test_loaders

def prepare_data_PACS_feature_noniid(args):
    data_base_path = '../data/PACS'
    transform_PACS= transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
            transforms.Resize([args.size, args.size]),            
            transforms.ToTensor(),
    ])
    dataset_name = ["art_painting", "cartoon", "photo", "sketch"]
    train_loaders, test_loaders = [], []
    for dataset in dataset_name:
        train_set = data_utils.PACSDataset(data_base_path, dataset, transform=transform_PACS)
        test_set = data_utils.PACSDataset(data_base_path, dataset, transform=transform_test, train=False)
        train_loaders.append(torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.number_workers, pin_memory=True))
        test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=args.number_workers, pin_memory=True))
    return train_loaders, test_loaders

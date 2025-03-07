#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import copy
from utils import PerAvgOptimizer
import torch.nn.functional as F
import time
from collections import Counter
from torch.utils.data.dataset import Dataset
import numpy as np
import random
import pickle
from utils import *

class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.criterion_CE = nn.CrossEntropyLoss()
        self.old_model = None
        random.seed(args.seed)  
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)  
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        args.device = args.device if torch.cuda.is_available() else 'cpu'

    def update_weights(self, model, train_loader):
        # Set mode to train model
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                _, logits = model(images)
                loss = self.criterion_CE(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model)
    
    def update_weights_fedProx(self, model, train_loader):
        # Set mode to train model
        server_model = copy.deepcopy(model)
        model.train()
        server_model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, output = model(images)
                loss = self.criterion_CE(output, labels)
                if batch_idx>0:
                    w_diff = torch.tensor(0., device=self.args.device)
                    for w, w_t in zip(server_model.parameters(), model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.mu / 2. * w_diff
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model)

    def update_weights_perfedavg(self, model, train_loader):
        # Set mode to train model
        model.train()
        optimizer = PerAvgOptimizer(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                if batch_idx % 2 == 0 and batch_idx != len(train_iter) - 1:
                    # step1
                    temp_model = copy.deepcopy(list(model.parameters()))
                    optimizer.zero_grad()
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    rep, logits = model(images)
                    loss = self.criterion_CE(logits, labels)
                    loss.backward()
                    optimizer.step()
                elif batch_idx % 2 == 0 and batch_idx == len(train_iter) - 1:
                    continue
                else:
                    # step2
                    optimizer.zero_grad()
                    images, labels = next(train_iter)
                    images, labels = images.to(self.device).float(), labels.to(self.device).long()
                    rep, logits = model(images)
                    loss = self.criterion_CE(logits, labels)
                    loss.backward()
                    for old_param, new_param in zip(model.parameters(), temp_model):
                        old_param.data = new_param.data.clone()
                    optimizer.step(beta=self.args.beta)
        return copy.deepcopy(model)

    def update_weights_fedrep(self, model, train_loader):
        # Set mode to train model
        model.train()
        optimizer_feat = torch.optim.SGD(model.features.parameters(), lr=self.args.lr)
        optimizer_class = torch.optim.SGD(model.classifier.parameters(), lr=self.args.lr)
        # freeze feature extractor layers
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True  
        # update local head layers
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer_class.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, output = model(images)
                loss = self.criterion_CE(output, labels)
                loss.backward()
                optimizer_class.step()
        # freeze classifier header layers
        for param in model.features.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = False
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            optimizer_feat.zero_grad()
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, output = model(images)
            loss = self.criterion_CE(output, labels)
            loss.backward()
            optimizer_feat.step()
        return copy.deepcopy(model)

    def update_weights_proto(self, model, train_loader, global_proto):
        # Set mode to train model
        server_model = copy.deepcopy(model)
        model.train()        
        server_model.eval()
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        loss_mse = nn.MSELoss()
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                proto, output = model(images)
                loss1 = self.criterion_CE(output, labels)
                if len(global_proto) == 0:
                    loss2 = loss1 * 0
                else:
                    proto_new = copy.deepcopy(proto.detach())
                    i = 0
                    for label in labels:
                        if label.item() in global_proto.keys():
                            proto_new[i, :] = global_proto[label.item()].data
                        i += 1
                    loss2 = loss_mse(proto_new, proto)
                loss = loss1 + loss2 * self.args.ld
                loss.backward()
                optimizer.step()
        # update local protos
        protos = {}
        model.eval()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            proto, output = model(images)
            for i in range(len(labels)):
                if labels[i].item() in protos:
                    protos[labels[i].item()].append(proto[i,:].detach().clone())
                else:
                    protos[labels[i].item()] = [proto[i,:].detach().clone()]
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return copy.deepcopy(model), averaged_protos

    def update_weights_moon(self, model, train_loader):
        # Set mode to train model
        server_model = copy.deepcopy(model)
        model.train()
        server_model.eval()
        # Set optimizer for the local updates
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            batch_loss = []
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, output = model(images)
                loss1 = self.criterion_CE(output, labels)
                rep_global = server_model(images)[0].detach()
                rep_old = self.old_model(images)[0].detach()
                loss2 = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.args.tau)))
                loss = loss1 + torch.mean(loss2) * self.args.ld
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        self.old_model = copy.deepcopy(model)
        return copy.deepcopy(model)

    def update_weights_adcol(self, model, discriminator,  train_loader):
        # Set mode to train model
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        # Set optimizer for the local updates
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                loss1 = self.criterion_CE(logits, labels)
                client_index = discriminator(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
                loss = loss1 + self.args.pall_beta * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # save features
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
        features = torch.cat(features, dim=0)
        return copy.deepcopy(model), features
  
    def update_weights_ours(self, round, model, discriminator, classifier_model, train_loader, global_proto,momentum = 1):
        # don't change BN parameter and momentum change model classifier layer parameters
        model_state_dict = model.classifier.state_dict()
        classifier_model_state_dict = classifier_model.classifier.state_dict()
        with torch.no_grad():
            for key in model_state_dict:
                if "bn" not in key:
                    model_param = model_state_dict[key]
                    classifier_model_param = classifier_model_state_dict[key]
                    model_state_dict[key].copy_((1 - momentum) * model_param + momentum * classifier_model_param)
        model.classifier.load_state_dict(model_state_dict)
        # Set mode to train model
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        # Set optimizer for the local updates
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                CE_loss = self.criterion_CE(logits, labels)
                if len(global_proto) == 0:
                    loss1 = 0
                else:
                    num_classes = len(global_proto)
                    proto_dim = global_proto[0].shape[0]
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                    rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
                    global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
                    C_y = global_proto_matrix_normalized[labels] 
                    numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                    denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                    loss1 = -torch.log(numerator / denominator).mean()
                client_index = discriminator(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
                loss = CE_loss + self.args.pall_beta * kl_loss + self.args.pall_mu * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # save features
        train_iter = iter(train_loader)
        labels_ = []
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
            labels_.append(labels.detach().clone().cpu())
        features = torch.cat(features, dim=0).to(self.args.device)
        
        labels_ = torch.cat(labels_, dim=0)
        return copy.deepcopy(model), [features, labels_]

    def update_weights_ours_ablation1(self, round, model, discriminator, classifier_model, train_loader, global_proto,momentum = 1):
        # don't change BN parameter and momentum change model classifier layer parameters
        model_state_dict = model.classifier.state_dict()
        classifier_model_state_dict = classifier_model.classifier.state_dict()
        with torch.no_grad():
            for key in model_state_dict:
                if "bn" not in key:
                    model_param = model_state_dict[key]
                    classifier_model_param = classifier_model_state_dict[key]
                    model_state_dict[key].copy_((1 - momentum) * model_param + momentum * classifier_model_param)
        model.classifier.load_state_dict(model_state_dict)
        # Set mode to train model
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        # Set optimizer for the local updates
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                CE_loss = self.criterion_CE(logits, labels)
                if len(global_proto) != 0 and self.args.loss_component in [2,4]:
                    num_classes = len(global_proto)
                    proto_dim = global_proto[0].shape[0]
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                    rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
                    global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
                    C_y = global_proto_matrix_normalized[labels] 
                    numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                    denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                    loss1 = -torch.log(numerator / denominator).mean()
                else:
                    loss1 = 0
                if self.args.loss_component in [3,4]:
                    client_index = discriminator(rep)
                    client_index_softmax = F.log_softmax(client_index, dim=-1)
                    target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                        self.device
                    )
                    target_index_softmax = F.softmax(target_index, dim=-1)
                    kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                    kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
                else:
                    kl_loss = 0
                loss = CE_loss + self.args.pall_beta * kl_loss + self.args.pall_mu * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # save features
        train_iter = iter(train_loader)
        labels_ = []
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
            labels_.append(labels.detach().clone().cpu())
        features = torch.cat(features, dim=0).to(self.args.device)
        labels_ = torch.cat(labels_, dim=0)
        return copy.deepcopy(model), [features, labels_]

    def update_weights_ours_ablation2(self, round, model, discriminator, classifier_model, train_loader, global_proto,momentum = 1):
        # don't change BN parameter and momentum change model classifier layer parameters
        if self.args.cls_component == 1:
            model_state_dict = model.classifier.state_dict()
            classifier_model_state_dict = classifier_model.classifier.state_dict()
            with torch.no_grad():
                for key in model_state_dict:
                    if "bn" not in key:
                        model_param = model_state_dict[key]
                        classifier_model_param = classifier_model_state_dict[key]
                        model_state_dict[key].copy_((1 - momentum) * model_param + momentum * classifier_model_param)
            model.classifier.load_state_dict(model_state_dict)
        # Set mode to train model
        model.train()
        features = []
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        # Set optimizer for the local updates
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                rep, logits = model(images)
                CE_loss = self.criterion_CE(logits, labels)
                if len(global_proto) == 0:
                    loss1 = 0
                else:
                    num_classes = len(global_proto)
                    proto_dim = global_proto[0].shape[0]
                    global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                    for label, proto in global_proto.items():
                        global_proto_matrix[label] = proto
                    rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
                    global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
                    C_y = global_proto_matrix_normalized[labels]
                    numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                    denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                    loss1 = -torch.log(numerator / denominator).mean()
                client_index = discriminator(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)           
                loss = CE_loss + self.args.pall_beta * kl_loss + self.args.pall_mu * loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # save features
        train_iter = iter(train_loader)
        labels_ = []
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            features.append(rep.detach().clone().cpu())
            labels_.append(labels.detach().clone().cpu())
        features = torch.cat(features, dim=0).to(self.args.device)
        labels_ = torch.cat(labels_, dim=0)
        return copy.deepcopy(model), [features, labels_]
    
    def compute_proto(self, model, train_loader):
        # update local protos
        protos = {}
        outputall = []
        num_list = Counter()
        model.eval()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            proto, _ = model(images)
            for i in range(len(labels)):
                num_list[labels[i].item()] += 1
                if labels[i].item() in protos:
                    protos[labels[i].item()].append(proto[i,:].detach().clone())
                else:
                    protos[labels[i].item()] = [proto[i,:].detach().clone()]
                outputall.append(proto[i, :].detach().clone())
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return averaged_protos, torch.stack(outputall, dim=0), num_list

    def compute_proto_ours(self, model, train_loader, classifier_model, momentum):
        # don't change BN parameter and momentum change model classifier layer parameters
        model_state_dict = model.classifier.state_dict()
        classifier_model_state_dict = classifier_model.classifier.state_dict()
        with torch.no_grad():
            for key in model_state_dict:
                if "bn" not in key:
                    model_param = model_state_dict[key]
                    classifier_model_param = classifier_model_state_dict[key]
                    if momentum == 0:
                        model_state_dict[key].copy_(classifier_model_param)
                    else:
                        model_state_dict[key].copy_(model_param + momentum * classifier_model_param)
        model.classifier.load_state_dict(model_state_dict)
        # update local protos
        protos = {}
        num_list = Counter()
        model.eval()
        train_iter = iter(train_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            proto, output = model(images)
            for i in range(len(labels)):
                num_list[labels[i].item()] += 1
                if labels[i].item() in protos:
                    protos[labels[i].item()].append(proto[i,:].detach().clone())
                else:
                    protos[labels[i].item()] = [proto[i,:].detach().clone()]
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return averaged_protos,  num_list

    def compute_proto_debug(self, train_loader):
        # update local protos
        protos = {}
        outputall = []
        num_list = Counter()
        features, labels = train_loader[0], train_loader[1]
        for batch_idx in range(len(features)):
            feature = features[batch_idx]
            label = labels[batch_idx]
            num_list[label.item()] += 1
            if label.item() in protos:
                protos[label.item()].append(feature.detach().clone())
            else:
                protos[label.item()] = [feature.detach().clone()]
        averaged_protos = {}
        for label, proto_list in protos.items():
            averaged_protos[label] = torch.mean(torch.stack(proto_list), dim=0)
        return averaged_protos, num_list

    def update_weights_rucr(self,  model, train_loader, global_proto, ratio_list):
        # Set mode to train model
        global_proto = torch.stack([global_proto[idx] for idx in range(len(global_proto))], dim=0)
        ratio_list = torch.tensor([ratio_list[idx] for idx in range(len(ratio_list))]).to(self.args.device)
        global_F = F.normalize(global_proto, dim=1)
        model.train()
        # Set optimizer for the local update
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for step in range(self.args.wk_iters):
            train_iter = iter(train_loader)
            for batch_idx in range(len(train_iter)):
                optimizer.zero_grad()
                images, labels = next(train_iter)
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                proto, output = model(images)
                loss1 = self.criterion_CE(output, labels)
                feat_loss = self.bal_simclr_imp(proto, labels, global_F, ratio_list)
                loss = loss1 + feat_loss * self.args.feat_loss_arg
                loss.backward()
                optimizer.step()
        return copy.deepcopy(model)

    def bal_simclr_imp(self, f, labels, global_proto, ratio_list):
        f_norm = F.normalize(f, dim=1)
        # cos sim
        sim_logit = f_norm.mm(global_proto.T)
        # temperature
        sim_logit_tau = sim_logit.div(self.args.tau)
        # cls ratio
        src_ratio = ratio_list[labels].log() * self.args.times
        add_src = torch.scatter(torch.zeros_like(sim_logit), 1, labels.unsqueeze(1), src_ratio.view(-1, 1))
        f_out = sim_logit_tau + add_src
        loss = self.criterion_CE(f_out, labels)
        return loss
    
    def local_crt(self, model, glo_means, fs_all):
        for param_name, param in model.named_parameters():
            if 'classifier' not in param_name:
                param.requires_grad = False
        
        crt_dataset = MixupDataset_norm(glo_means, fs_all, self.args)
        model.eval()
        temp_optimizer = torch.optim.SGD(model.classifier.parameters(), lr=self.args.lr)
        for i in range(self.args.wk_iters):
            crt_loader = torch.utils.data.DataLoader(dataset=crt_dataset,
                                    batch_size=self.args.batch,
                                    shuffle=True)
            for feat, cls in crt_loader:
                feat, cls = feat.to(self.device), cls.to(self.device)
                outputs = model.classifier(feat)
                loss = self.criterion_CE(outputs, cls)
                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()
        return copy.deepcopy(model.classifier.state_dict())

class LocalTest(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        
    def test_inference(self, args, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(args.device).float(), target.to(args.device).long()
            rep, output = model(data)
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc

    def test_inference_proto(self, args, model, test_loader, global_proto):
        model.eval()
        test_loss = 0
        correct = 0
        loss_mse = nn.MSELoss()
        for data, target in test_loader:
            data, target = data.to(args.device).float(), target.to(args.device).long()
            proto, output = model(data)
            loss1 = self.criterion(output, target).item()
            proto_new = copy.deepcopy(proto.detach())
            i = 0
            for label in target:
                if label.item() in global_proto.keys():
                    proto_new[i, :] = global_proto[label.item()].data
                i += 1
            loss2 = loss_mse(proto_new, proto)
            test_loss += loss1 + loss2 * self.args.ld
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc

    def test_inference_moon(self, args, old_model, model, global_model, test_loader):
        model.eval()
        old_model.eval()
        global_model.eval()
        test_loss = 0
        correct = 0
        targets = []
        for data, target in test_loader:
            data, target = data.to(self.device).float(), target.to(self.device).long()
            rep, output = model(data)
            loss1 = self.criterion(output, target)
            rep_global = global_model(data)[0].detach()
            rep_old = old_model(data)[0].detach()
            loss2 = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.args.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.args.tau)))
            test_loss += loss1 + torch.mean(loss2) * self.args.ld
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), acc

    def test_inference_adcol(self, args, model, test_loader, dmodel):
        model.eval()
        dmodel.eval()
        test_loss = 0
        correct = 0
        loss_kl = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            loss1 = self.criterion(logits, labels)
            client_index = dmodel(rep)
            client_index_softmax = F.log_softmax(client_index, dim=-1)
            target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                self.device
            )
            target_index_softmax = F.softmax(target_index, dim=-1)
            kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
            kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)  
            loss_kl += kl_loss
            test_loss += loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), loss_kl / len(test_loader), acc
    
    def test_inference_ours(self, args, model, test_loader, dmodel):
        model.eval()
        dmodel.eval()
        test_loss = 0
        correct = 0
        loss_kl = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            loss1 = self.criterion(logits, labels)
            client_index = dmodel(rep)
            client_index_softmax = F.log_softmax(client_index, dim=-1)
            target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                self.device
            )
            target_index_softmax = F.softmax(target_index, dim=-1)
            kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
            kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)  
            loss_kl += kl_loss
            test_loss += loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)        
        return test_loss / len(test_loader), loss_kl / len(test_loader), acc
    
    def test_inference_ablation1(self, args, model, test_loader, global_proto,  dmodel):
        correct = 0
        loss_kl = 0
        model.eval()
        dmodel.eval()
        loss_CE = 0
        loss_info = 0
        # Set optimizer for the local updates
        train_iter = iter(test_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            CE_loss = self.criterion(logits, labels)
            if len(global_proto) != 0 and self.args.loss_component in [2,4]:
                num_classes = len(global_proto)
                proto_dim = global_proto[0].shape[0]
                global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                for label, proto in global_proto.items():
                    global_proto_matrix[label] = proto
                rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
                global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
                C_y = global_proto_matrix_normalized[labels]
                numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                loss1 = -torch.log(numerator / denominator).mean()
            else:
                loss1 = 0
            if self.args.loss_component in [3,4]:
                client_index = dmodel(rep)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
            else:
                kl_loss = 0
            loss_CE += CE_loss
            loss_kl += kl_loss
            loss_info += loss1
            loss = CE_loss + self.args.pall_beta * kl_loss + self.args.pall_mu * loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)  
        return loss_CE / len(test_loader), loss_kl / len(test_loader), loss_info / len(test_loader) , acc
    
    def test_inference_ablation2(self, args, model, test_loader, global_proto,  dmodel):
        correct = 0
        loss_kl = 0
        model.eval()
        dmodel.eval()
        loss_CE = 0
        loss_info = 0
        # Set optimizer for the local updates
        train_iter = iter(test_loader)
        for batch_idx in range(len(train_iter)):
            images, labels = next(train_iter)
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            rep, logits = model(images)
            CE_loss = self.criterion(logits, labels)
            if len(global_proto) != 0:
                num_classes = len(global_proto)
                proto_dim = global_proto[0].shape[0]
                global_proto_matrix = torch.zeros((num_classes, proto_dim), device=self.device)
                for label, proto in global_proto.items():
                    global_proto_matrix[label] = proto
                rep_normalized = F.normalize(rep, dim=1)  # [batch_size, d]
                global_proto_matrix_normalized = F.normalize(global_proto_matrix, dim=1)  # [num_classes, d]
                C_y = global_proto_matrix_normalized[labels]
                numerator = torch.exp((rep_normalized * C_y).sum(dim=1) / self.args.T)
                denominator = torch.exp((rep_normalized @ global_proto_matrix.T) / self.args.T).sum(dim=1)
                loss1 = -torch.log(numerator / denominator).mean()
            else:
                loss1 = 0
            client_index = dmodel(rep)
            client_index_softmax = F.log_softmax(client_index, dim=-1)
            target_index = torch.full(client_index.shape, 1 / self.args.num_users).to(
                self.device
            )
            target_index_softmax = F.softmax(target_index, dim=-1)
            kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
            kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
            loss_CE += CE_loss
            loss_kl += kl_loss
            loss_info += loss1
            loss = CE_loss + self.args.pall_beta * kl_loss + self.args.pall_mu * loss1
            pred = logits.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()
        acc = correct / len(test_loader.dataset)  
        return loss_CE / len(test_loader), loss_kl / len(test_loader), loss_info / len(test_loader) , acc

from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.models as models
from driver.driver import DLDriver
from data.dataloader import DataLoaderCreator
from torch.utils.data import Subset, ConcatDataset



class DLDriver_DRO(DLDriver):
  
    def __init__(self, args, soft_label=False):
        self.soft_label = soft_label
        super(DLDriver_DRO, self).__init__(args)
    
    def _build_data_loader(self, groups=None):
        data_loader = DataLoaderCreator(self.args)
        if groups is None:
            groups = data_loader.train.dataset.groups
        if groups is not None:
            if self.soft_label:
                groups = np.array(groups, dtype="float32")
                if len(groups.shape)==1:
                    groups = (groups.reshape(-1,1)==np.arange(max(groups)+1)).astype(float)
                self.nb_groups = groups.shape[1]
            else:
                self.nb_groups = max(groups)+1
            data_loader.train.dataset.groups = groups
            if self.args.reweight_groups:
                if not self.soft_label:
                    groups = (torch.tensor(groups).unsqueeze(1)==torch.arange(self.nb_groups)).numpy()
                train_dataset = data_loader.train.dataset
                group_sizes = groups.sum(axis=0)
                group_weights = len(train_dataset) / group_sizes
                weights_g = (groups @ group_weights.reshape(-1, 1)).flatten()
                sampler = torch.utils.data.WeightedRandomSampler(weights_g, len(train_dataset))
                data_loader.train = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=sampler)
            self.q = torch.ones(self.nb_groups).to(self.device[0])
        return data_loader.train, data_loader.predict
    
    # def groups_(self, y, g):
    #     idx_g, idx_b = [], []
    #     all_g = g

    #     for g in all_g.unique():
    #         idx_g.append(g)
    #         idx_b.append(all_g == g)

    #     return zip(idx_g, idx_b)

    def compute_loss(self, criterion, datas, model):
        inputs, labels, groups = datas
        logits = model(inputs)
        loss = criterion(logits, labels)

        if not self.soft_label:
            groups = (groups.unsqueeze(1)==torch.arange(self.nb_groups, device=self.device[0])).float()
        
        with torch.no_grad(): 
            self.q *= (self.args.eta * (loss.reshape(1,-1) @ (groups / groups.sum(dim=0)).nan_to_num())).exp().flatten()
        self.q /= self.q.sum()
        loss_weight = ((groups / groups.sum(dim=0)).nan_to_num() @ self.q.reshape(-1,1))
        # loss_weight = (groups @ self.q.reshape(-1,1))/ (groups @ self.q.reshape(-1,1)).sum()
        loss_value = loss.reshape(1, -1) @ loss_weight
        
        # for idx_g, idx_b in self.groups_(labels, groups):
        #     self.q[idx_g] *= (self.args.eta * loss[idx_b].mean()).exp().item()

        # self.q /= self.q.sum()
        # loss_value = 0
        # for idx_g, idx_b in self.groups_(labels, groups):
        #     loss_value += self.q[idx_g] * loss[idx_b].mean()
        return loss_value


class DLDriver_JTT(DLDriver):
    
    def __init__(self, args):
        super(DLDriver_JTT, self).__init__(args)
        self.nb_examples = len(self.train_data_loader.dataset)
    
    def train(self, train_loader=None, model=None, criterion=None, optimizer=None, scheduler=None):
        train_loader = train_loader or self.train_data_loader
        model = model or self.model
        criterion = criterion or self.criterion
        optimizer = optimizer or self.optimizer
        scheduler = scheduler or self.scheduler
        
        epoch_T = self.args.T
        epoch_final = self.args.train_epochs
        up = self.args.up
        
        stage1_model = self._build_model()
        stage1_optimizer = self._select_optimizer(stage1_model.parameters())
        stage1_scheduler = self._select_scheduler(stage1_optimizer)
        for epoch in tqdm(range(epoch_T), total=epoch_T):
            train_loss = self.epoch_train(train_loader, stage1_model, criterion, stage1_optimizer, stage1_scheduler, self.scheduler_step_after_batch, self.device[0])
        
        stage1_model.eval()
        train_loader_not_shuffle = DataLoaderCreator(self.args).train_data_loader(shuffle=False)
        error_indices = []
        for _, datas in enumerate(train_loader_not_shuffle):
            inputs, labels = datas[0].to(self.device[0]), datas[1].to(self.device[0])
            with torch.no_grad():
                logits = stage1_model(inputs)
            wrong_predictions = logits.argmax(1).ne(labels).cpu()
            batch_error_indices = torch.arange(len(error_indices), len(error_indices)+len(labels))[wrong_predictions]
            error_indices.append(batch_error_indices)
        error_indices = torch.cat(error_indices, dim=0)
        
        train_set = self.train_data_loader.dataset
        duplicated_train_dataset = ConcatDataset([train_set, Subset(train_set, error_indices.repeat(up).to(torch.int64))])
        duplicated_train_loader = DataLoader(duplicated_train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        for epoch in tqdm(range(epoch_final), total=epoch_final):
            train_loss = self.epoch_train(duplicated_train_loader, model, criterion, optimizer, scheduler, self.scheduler_step_after_batch, self.device[0])
            if self.record:
                self.record.add_train_log(f'epoch{epoch}', np.average(train_loss), self.metric(), if_print=True)
        if self.record:
            self.record.add_test_outcome(self.metric())


class DomainIndependentClassifier(nn.Module):
    def __init__(self, backbone, num_domain):
        super(DomainIndependentClassifier, self).__init__()
        # self.backbone = self.get_classifier(arch, num_classes, weights=weights)
        self.backbone = backbone
        self.domain_classifier_list = nn.ModuleList([
            nn.Linear(self.backbone.fc.in_features, self.backbone.fc.out_features) for _ in range(num_domain)])
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        logits_per_domain = [classifier(x) for classifier in self.domain_classifier_list]
        logits_per_domain = torch.stack(logits_per_domain, dim=1)

        if self.training:
            return logits_per_domain
        else:
            return logits_per_domain.mean(dim=1)
    
    # def get_classifier(self, arch, num_classes, weights="IMAGENET1K_V1", new_fc=True):
    #     if arch.startswith("resnet"):
    #         model = models.__dict__[arch](weights=weights)
    #         if new_fc:
    #             model.fc = nn.Linear(model.fc.in_features, num_classes)
    #     else:
    #         raise NotImplementedError

    #     return model


class DLDriver_DI(DLDriver):
    def __init__(self, args, soft_label=False):
        self.soft_label = soft_label
        super(DLDriver_DI, self).__init__(args)
        

    def _build_model(self, model_path=None, di_num_domain=None):
        self.di_num_domain = di_num_domain or self.args.di_num_domain
        model = super()._build_model()
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model = DomainIndependentClassifier(model, self.di_num_domain).to(self.device[0])
        return model
    
    def _build_data_loader(self, groups=None):
        data_loader = DataLoaderCreator(self.args)
        # if groups is not None:
        #     data_loader.train.dataset.groups = groups
        # self.nb_groups = max(data_loader.train.dataset.groups)+1
        if groups is None:
            groups = data_loader.train.dataset.groups
        if groups is not None:
            if self.soft_label:
                groups = np.array(groups, dtype="float32")
                if len(groups.shape)==1:
                    groups = (groups.reshape(-1,1)==np.arange(max(groups)+1)).astype(float)
                self.nb_groups = groups.shape[1]
            else:
                self.nb_groups = max(groups)+1
            data_loader.train.dataset.groups = groups
        return data_loader.train, data_loader.predict

 
    def compute_loss(self, criterion, datas, model):
        inputs, labels, groups = datas
        logits_per_domain = model(inputs)
        if not self.soft_label:
            logits = logits_per_domain[range(inputs.shape[0]), groups]
        else:
            logits = (groups.unsqueeze(1) @ logits_per_domain).squeeze(1)
        loss = self.criterion(logits, labels)
        loss = criterion(logits, labels).mean()
        return loss
    



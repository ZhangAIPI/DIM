import sys
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from sklearn.metrics import accuracy_score
from data.dataloader import DataLoaderCreator
from model.model import ModelCreator


class DLDriver(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.train_data_loader, self.pred_data_loader = self._build_data_loader()
        self.model = self._build_model()
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler()
    
    def compute_loss(self, criterion, datas, model):
        inputs, labels = datas[0], datas[1]
        logits = model(inputs)
        loss = criterion(logits, labels).mean()
        self.N += len(labels)
        self.C += (logits.argmax(dim=1) == labels).sum().item()
        return loss
    
    def train(self, epochs=None, train_loader=None, model=None, criterion=None, optimizer=None, scheduler=None):
        epochs = epochs or self.args.train_epochs
        train_loader = train_loader or self.train_data_loader
        model = model or self.model
        criterion = criterion or self.criterion
        optimizer = optimizer or self.optimizer
        scheduler = scheduler or self.scheduler
        
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss = self.epoch_train(train_loader, model, criterion, optimizer, scheduler, self.scheduler_step_after_batch, self.device[0])


        if self.args.model_save_path:
            torch.save(model.state_dict(), self.args.model_save_path)

    def epoch_train(self, train_loader, model, criterion, optimizer, scheduler, scheduler_step_after_batch=False, device=None):
        self.N = np.array(0)
        self.C = np.array(0)
        model.train()
        train_loss=[]
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, datas in pbar:
            datas = [data.to(device) for data in datas]
            optimizer.zero_grad(set_to_none=True)
            loss = self.compute_loss(criterion, datas, model)
            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler_step_after_batch or i == len(train_loader)-1:
                scheduler.step()
            pbar.set_description(f"loss: {np.average(train_loss):.4f}, acc: {self.C/self.N:.4f}")
        return train_loss

    def predict(self, pred_loader=None):
        pred_loader = pred_loader or self.pred_data_loader
        model = self.model
        device = self.device[0]

        model.eval()
        with torch.no_grad():
            preds = torch.tensor([])
            for _, (inputs, _, _) in enumerate(pred_loader):
                pred = model(inputs.to(device)).cpu().detach()
                preds = torch.concat([preds, pred], axis=0)
        return preds

    def metric(self):
        y_true = self.pred_data_loader.dataset.targets
        y_pred = self.predict(self.pred_data_loader).argmax(dim=1)
        accuracy = accuracy_score(y_true, y_pred)
        return pd.Series({'test_accuracy': accuracy})
    
    def _acquire_device(self):
        if isinstance(self.args.device, str):
            device = [torch.device(self.args.device)]
        else:
            device = [torch.device(device_name) for device_name in self.args.device]
        print(f'Use: {self.args.device}')
        return device
    
    def _build_data_loader(self):
        data_loader = DataLoaderCreator(self.args)
        return data_loader.train, data_loader.predict
    
    def _build_model(self):
        model = ModelCreator(self.args).model.to(self.device[0])
        return model

    def _select_criterion(self):
        if self.args.criterion == 'L1':
            criterion = nn.L1Loss(reduction='none')
        elif self.args.criterion == 'CE':
            criterion = nn.CrossEntropyLoss(reduction='none')
        elif self.args.criterion == 'nll':
            criterion = nn.NLLLoss(reduction='none')
        elif self.args.criterion == "mse":
            criterion = nn.MSELoss(reduction='none')
        return criterion

    def _select_optimizer(self, params=None, lr=None):
        params = params or self.model.parameters()
        lr=lr or self.args.lr
        if self.args.optimizer == 'Adam':
            model_optim = optim.Adam(params, lr=lr, weight_decay=0.0001789)
        elif self.args.optimizer == 'AdamW':
            model_optim = optim.AdamW(params, lr=lr, weight_decay=0, eps=0.001)
        elif self.args.optimizer == 'SGD':
            model_optim = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0001)
        return model_optim
    
    def _select_scheduler(self, optimizer=None):
        optimizer = optimizer or self.optimizer
        self.scheduler_step_after_batch = False
        if self.args.scheduler == 'OneCycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, div_factor=25,
                                                        steps_per_epoch=len(self.data_loader.train), 
                                                        epochs=self.args.train_epochs)
            self.scheduler_step_after_batch = True
        elif self.args.scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        elif self.args.scheduler == 'multistep':
            scheduler = MultiStepLR(optimizer, milestones=[30,60,90,120,150,180,210,240], gamma=0.5)
        elif self.args.scheduler == 'multistep2':
            scheduler = MultiStepLR(optimizer, milestones=[30,35,40,45,50,55,60,65], gamma=0.5)
        elif self.args.scheduler == 'multistep3':
            scheduler = MultiStepLR(optimizer, milestones=[4,8,12,16,20,24], gamma=0.5)
        elif self.args.scheduler == 'multistep4':
            scheduler = MultiStepLR(optimizer, milestones=[81,122], gamma=0.1)
        elif self.args.scheduler == 'multistep5':
            scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        else:
            scheduler = MultiStepLR(optimizer, milestones=[self.args.train_epochs], gamma=1)
        return scheduler
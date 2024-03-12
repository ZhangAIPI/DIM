import os
import sys
sys.path.append('.')
from tqdm import tqdm
from types import MethodType
# import submitit
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
from torchvision import transforms
from driver.driver import DLDriver
from driver.other_driver import DLDriver_DRO, DLDriver_JTT, DLDriver_DI
from data.dataloader import DataLoaderCreator
from data.augmentation_supercifar100 import AugmentSupercifar100
from model.model import ModelCreator



class GroupDataset:
    def __init__(self, dataset, groups=None, subsample_what=None):
        self.dataset = dataset
        
        self.i = list(range(len(self.dataset)))
        
        self.targets = self.dataset.targets
        self.groups = groups or self.dataset.groups
        self.nb_labels = len(set(self.dataset.targets))
        self.nb_groups = len(set(self.groups))

        self.count_groups()

        if subsample_what is not None:
            self.subsample_(subsample_what)
            self.count_groups()
 
    def count_groups(self):   
        self.group_sizes = (torch.tensor(self.groups)==torch.arange(self.nb_groups).unsqueeze(1)).sum(dim=1)
        self.class_sizes = (torch.tensor(self.targets)==torch.arange(self.nb_labels).unsqueeze(1)).sum(dim=1)

        group_weights = len(self) / self.group_sizes
        label_weights = len(self) / self.class_sizes
        
        self.weights_g = group_weights[self.groups]
        self.weights_y = label_weights[self.targets]

    def subsample_(self, subsample_what):
        

        if subsample_what == "groups":
            group_sizes = np.array(self.group_sizes)
            min_size = group_sizes[group_sizes>0].min()
            max_size = group_sizes[group_sizes>0].max()
        else:
            min_size = min(list(self.class_sizes))
            max_size = max(list(self.class_sizes))

        print('if max > min ', max_size > min_size)
        if max_size > min_size:
            counts_g = [0] * self.nb_groups
            counts_y = [0] * self.nb_labels
            new_i = []
            perm = torch.randperm(len(self)).tolist()
            for p in perm:
                label, group = self.targets[self.i[p]], self.groups[self.i[p]]
                if (subsample_what == "groups"and counts_g[int(group)] < min_size) or (
                    subsample_what == "classes" and counts_y[int(label)] < min_size):
                    counts_g[int(group)] += 1
                    counts_y[int(label)] += 1
                    new_i.append(self.i[p])
            self.i = new_i

        
    
    def __getitem__(self, idx):
        x, target, _ = self.dataset[self.i[idx]]
        group = self.groups[self.i[idx]]
        return x, target, group

    def __len__(self):
        return len(self.i)


def metric(self):
    y_true = torch.tensor(self.pred_data_loader.dataset.targets)
    y_pred = self.predict(self.pred_data_loader).argmax(dim=1)
    corrects = y_true == y_pred
    groups = torch.tensor(self.pred_data_loader.dataset.groups)

    self.nb_groups = len(set(groups.tolist()))
    self.nb_labels = len(set(y_true.tolist()))
    
    avg_acc = sum(corrects) / len(corrects)
    group_acc = torch.tensor([corrects[groups==g].sum()/(groups==g).sum() for g in range(self.nb_groups)])
    
    acc_matrix = []
    for supid in range(self.nb_labels):
        subids = self.pred_data_loader.dataset.supid_to_subid_dict[supid]
        sup_sub_acc = group_acc[subids]
        sup_sub_acc_sorted, _ = torch.sort(sup_sub_acc)
        acc_matrix.append(sup_sub_acc_sorted.reshape(1, -1))
    acc_matrix = torch.cat(acc_matrix, dim=0)
    sorted_intersupavg_acc = acc_matrix.mean(dim=0)
    metric = pd.Series({'test_accuracy': avg_acc.item(), 'sorted_intersupavg_acc': sorted_intersupavg_acc.tolist()})
    return metric


def run_mitigation(args, mitigation_method, bias_method):
    print('--'*20, '\n', mitigation_method, bias_method)
    
    bias_groups = None
    
    if mitigation_method == "rwg_soft":
        driver = DLDriver(args)
        

        train_dataset = driver.train_data_loader.dataset
        if bias_groups is None:
            bias_groups = train_dataset.groups
        bias_groups = np.array(bias_groups, dtype="float32")
        if len(bias_groups.shape)==1:
            bias_groups = (bias_groups.reshape(-1,1)==np.arange(max(bias_groups)+1)).astype(float)
        train_dataset.groups = bias_groups
        
        group_sizes = bias_groups.sum(axis=0)
        group_weights = len(train_dataset) / group_sizes
        weights_g = (bias_groups @ group_weights.reshape(-1, 1)).flatten()
        sampler = torch.utils.data.WeightedRandomSampler(weights_g, len(train_dataset))
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, 
                                       sampler=sampler, num_workers=4, pin_memory=True)
        driver.train_data_loader = train_data_loader
        
        driver.train()
        metric = driver.metric()
    elif  mitigation_method == 'dro_soft':
        driver = DLDriver_DRO(args, soft_label=(mitigation_method == 'dro_soft'))

        driver.train_data_loader, driver.pred_data_loader = driver._build_data_loader(groups=bias_groups)
        
        
        driver.train()
        metric = driver.metric()

    elif  mitigation_method == 'di_soft':
        driver = DLDriver_DI(args, soft_label=(mitigation_method == 'di_soft'))
        
        bias_groups = bias_groups.reshape(bias_groups.shape[0], -1, args.di_num_domain).sum(axis=1)
        driver.train_data_loader, driver.pred_data_loader = driver._build_data_loader(groups=bias_groups)
        
        driver.model = driver._build_model(args.model_path)
        
        driver.optimizer = driver._select_optimizer(lr=args.fine_tune_lr)
        driver.scheduler = driver._select_scheduler()
        
        driver.train(epochs=args.fine_tune_epoch)
        metric = driver.metric()
    elif mitigation_method == 'intervention': # data-centic methods in our paper
        driver = DLDriver(args)
        
        dataloader_creator = DataLoaderCreator(driver.args, auto_create=False)
        intervention_dataset = dataloader_creator.train_data_loader(dataset_name=args.intervention_dataset, shuffle=False).dataset
        if bias_method not in ['random']:
            intervention_sim_df = pd.read_csv(args.intervention_sim_df_path, header=[0,1])
            intervention_sim = intervention_sim_df[bias_method].values
            intervention_slices = []
            for spid in range(args.num_classes):
                for cpid in range(args.component_num):
                    intervention_slices += np.abs(intervention_sim[:, spid*5+cpid]).argsort()[-args.topK:].tolist()
            intervention_slices = torch.tensor(intervention_slices) 
        elif bias_method == 'random':
            intervention_slices = torch.randint(len(intervention_dataset), (args.num_classes*args.component_num*args.topK,))
        selected_intervention_dataset = Subset(intervention_dataset, intervention_slices.to(torch.int64))
        train_intervention_dataset = ConcatDataset([driver.train_data_loader.dataset, selected_intervention_dataset])
        train_data_loader = DataLoader(train_intervention_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        driver.train_data_loader = train_data_loader
        
        driver.train()
        metric = driver.metric()

    print(metric)
    return metric['test_accuracy'], metric['sorted_intersupavg_acc']


def main(args, save_path):
    combination = []
    
    bias_method = None
    bias_method_list = ['DIM']
    # for mitigation_method in ['augmentation', 'rwg_soft', 'di_soft', 'dro_soft']:    
    for mitigation_method in ['rwg_soft', 'di_soft', 'dro_soft']:
        for bias_method in bias_method_list:
            combination.append([mitigation_method, bias_method])
            

    acc_df = pd.DataFrame(columns=['avg_acc']+[f'sorted_intersupavg_acc_{i}' for i in range(5)], 
                          index=pd.MultiIndex.from_tuples(combination))
    for mitigation_method, bias_method in combination:
        avg_acc, sorted_intersupavg_acc = run_mitigation(args, mitigation_method, bias_method)
        acc_df.loc[(mitigation_method, bias_method)] = [avg_acc]+sorted_intersupavg_acc
        if not args.not_save:
            acc_df.to_csv(os.path.join(save_path, f'acc_df_{args.scheduler}_{args.lr}.csv'), float_format='%.4f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, default=None)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--num_classes", type=int, default=20)
    # parser.add_argument("--model_path", type=str, default='save/001_train_model/resnet18_imbalancedsupercifar100/epoch30.pt')
    parser.add_argument("--model_path", type=str, default='save/001_train_model/resnet18_imbalancedsupercifar100.pt')
    parser.add_argument("--model_save_path", type=str, default=None)
    
    parser.add_argument("--only_train_fc", action='store_true', default=False)
    parser.add_argument("--dataset_name", type=str, default='imbalancedsupercifar100')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--criterion", type=str, default='CE')
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--scheduler", type=str, default='multistep5')
    parser.add_argument("--not_save", action='store_true', default=False)
    # param for DRO
    parser.add_argument("--reweight_groups", action='store_true', default=False)
    parser.add_argument("--eta", type=float, default=0.01)
    # param for JTT
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--up", type=int, default=4)
    
    # param for DI
    parser.add_argument("--di_num_domain", type=int, default=5)
    parser.add_argument("--fine_tune_epoch", type=int, default=10)
    parser.add_argument("--fine_tune_lr", type=float, default=0.01)
    
    # param for Intervention and Augmentation
    parser.add_argument("--intervention_dataset", type=str, default='imbalancedsupercifar100_intervention')
    parser.add_argument("--intervention_sim_df_path", type=str, default='intervention_soft_bias_group.csv')
    parser.add_argument("--topK", type=int, default=200)
    parser.add_argument("--component_num", type=int, default=2)
    
    args = parser.parse_args()
    
    
    DLDriver.metric = metric
    
    save_path = 'model_mitigation/'
    main(args, save_path)
    
    
    # --lr 0.005 --reweight_groups --eta 0.8
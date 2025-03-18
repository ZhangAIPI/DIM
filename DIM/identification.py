import os
import sys
sys.path.append('.')
import torch
import open_clip
import argparse
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from dataloader import DataLoaderCreator


class ComponetToGroup(object):
    
    def __init__(self, args, component_path, save_path='.'):
        self.args = args
        self.device = torch.device(self.args.device)
        self.component_path = component_path
        self.save_path = save_path
        
        self.clip_model, self.preprocess, self.tokenizer = self._build_clip_model()
        self.loader, self.train_loader_for_clip, self.intervention_loader_for_clip = self._build_data_loader()
        
        self.sub_class_names = self.loader.dataset.subclasses
        self.super_class_names = self.loader.dataset.classes
        self.super_sub_class_dict = self.loader.dataset.super_sub_class_dict
        # self.super_target_list = np.array(self.loader.dataset.targets)
        # self.sub_target_list = np.array(self.loader.dataset.subclass_targets)


    def _build_clip_model(self):
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_model.eval()
        clip_model = clip_model.to(self.device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return clip_model, preprocess, tokenizer


    def _build_data_loader(self):
        dataloader_creator = DataLoaderCreator(self.args, auto_create=False)
        
        CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
        CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        loader = dataloader_creator.train_data_loader(transform=transform, shuffle=False)

        train_loader_for_clip = dataloader_creator.train_data_loader(transform=self.preprocess, shuffle=False)
        intervention_loader_for_clip = dataloader_creator.train_data_loader('imbalancedsupercifar100_intervention', transform=self.preprocess, shuffle=False)
        return loader, train_loader_for_clip, intervention_loader_for_clip
     

    def _compute_feature(self, dataloader):
        feature_list = []
        for clip_batch in tqdm(dataloader, total=len(dataloader)):
            images_for_clip, _, _ = clip_batch
            images_for_clip = images_for_clip.to(self.device)
            with torch.no_grad():
                features = self.clip_model.encode_image(images_for_clip)
                feature_list.append(features.cpu())
        feature_list = torch.cat(feature_list, dim=0).numpy()
        return feature_list


    def _compute_data_bias_group(self, dataloader, name):
        self.feature_list = self._compute_feature(dataloader)
        self.components = pd.read_csv(self.component_path, index_col=[0,1], header=[0,1])
        methods_used = self.components.index.get_level_values(0).drop_duplicates().tolist()
        
        super_target_list = np.array(dataloader.dataset.targets)
        random_group = torch.randint(max(self.loader.dataset.groups)+1, (len(super_target_list),)).tolist()
        total_bias_group_list = [dataloader.dataset.groups, random_group]
        total_soft_bias_group_list = []
        for bias_method in methods_used:
            component = self.components.loc[bias_method]
            # bias_group_list = np.array([0]*len(super_target_list))
            bias_group_list = np.zeros(len(super_target_list), dtype=int)
            soft_bias_group_list = np.zeros((len(super_target_list), 100), dtype=float)
            for supid in range(20):
                super_class = self.super_class_names[supid]
                sup_component = component[super_class].values.astype(float)
                sup_feature_list = self.feature_list[super_target_list == supid]
                sup_feature_mean = sup_feature_list.mean(axis=0)
                sup_component_feature_sim = np.abs(np.matmul(sup_feature_list-sup_feature_mean, sup_component.T))
                sup_component_feature_sim = sup_component_feature_sim/sup_component_feature_sim.sum(axis=1, keepdims=True)
                bias_group_list[super_target_list == supid] = sup_component_feature_sim.argmax(axis=1)+5*supid
                soft_bias_group_list[super_target_list == supid, 5*supid:5*(supid+1)] = sup_component_feature_sim
            total_bias_group_list.append(bias_group_list)
            total_soft_bias_group_list.append(soft_bias_group_list)
        total_bias_group_list = np.array(total_bias_group_list).T
        total_soft_bias_group_list = np.hstack(total_soft_bias_group_list)
        
        total_bias_group_df = pd.DataFrame(total_bias_group_list, columns=['true_group', 'random_group'] + methods_used)
        total_bias_group_df.to_csv(os.path.join(self.save_path, f'{name}_bias_group.csv'), index=False)

        soft_df_columns = pd.MultiIndex.from_product([methods_used, list(range(100))])
        total_soft_bias_group_list = pd.DataFrame(total_soft_bias_group_list, columns=soft_df_columns)
        total_soft_bias_group_list.to_csv(os.path.join(self.save_path, f'{name}_soft_bias_group.csv'), index=False)
        
        
    def run(self):
        self._compute_data_bias_group(self.train_loader_for_clip, 'train')
        self._compute_data_bias_group(self.intervention_loader_for_clip, 'intervention')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='imbalancedsupercifar100')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    component_path = 'save/multi_bias_identification/result/component.csv'
    save_path = 'save/build_bias_group'
    c2g = ComponetToGroup(args, component_path, save_path)
    c2g.run()
    
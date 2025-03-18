import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
import open_clip
import argparse
import torch.nn.functional as F
from dataloader import DataLoaderCreator
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import CCA
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import itertools
import torchvision


class ModelCreator(object):
    def __init__(self, args=None):
        self.args = args
        self.model = self.create_model()
        
    def create_model(self, model_name=None, num_classes=None):
        model_name = model_name or self.args.model
        num_classes = num_classes or self.args.num_classes
        return torchvision.models.resnet18(weights=None, num_classes=num_classes)
        



class Trainer(object):
    
    def __init__(self, args, save_path='.'):
        self.args = args
        self.device = torch.device(self.args.device)
        self.save_path = save_path
        
        self.clip_model, self.preprocess, self.tokenizer = self._build_clip_model()
        self.classifier = self._build_classifier()
        self.loader, self.test_loader, self.loader_for_clip = self._build_data_loader()
        
        self.sub_class_names = self.loader.dataset.subclasses
        self.super_class_names = self.loader.dataset.classes
        self.super_sub_class_dict = self.loader.dataset.super_sub_class_dict
        self.super_target_list = np.array(self.loader.dataset.targets)
        self.sub_target_list = np.array(self.loader.dataset.subclass_targets)
    

    def _build_clip_model(self):
        clip_model_name = self.args.clip_model_name
        clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained="openai")
        clip_model.eval()
        clip_model = clip_model.to(self.device)
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        return clip_model, preprocess, tokenizer
        

    def _build_classifier(self, path=None):
        path = path or self.args.model_path
        classifier = ModelCreator(self.args).model.to(self.device)
        classifier.load_state_dict(torch.load(path))
        return classifier


    def _build_data_loader(self):
        dataloader_creator = DataLoaderCreator(self.args, auto_create=False)
        
        CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
        CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        loader = dataloader_creator.train_data_loader(transform=transform, shuffle=False)
        
        test_loader = dataloader_creator.predict_data_loader()
        loader_for_clip = dataloader_creator.train_data_loader(transform=self.preprocess, shuffle=False)
        return loader, test_loader, loader_for_clip
     

    def _compute_feature(self):
        feature_list = []
        for clip_batch in tqdm(self.loader_for_clip, total=len(self.loader_for_clip)):
            images_for_clip = clip_batch[0].to(self.device)
            with torch.no_grad():
                features = self.clip_model.encode_image(images_for_clip)
                features = F.normalize(features, dim=1)
                feature_list.append(features.cpu())
        feature_list = torch.cat(feature_list, dim=0).numpy()
        return feature_list


    def _compute_supervision(self):
        correct_list_all = []
        logits_list_all = []
        loss_list_all = []
        prob_list_all = []
        target_logit_list_all = []
        
        classifier = ModelCreator(self.args).model.to(self.device)
        for i in range(30):
            classifier.load_state_dict(torch.load(os.path.join(self.args.model_path[:-3], f'epoch{i}.pt')))
            
            correct_list = []
            logits_list = []
            loss_list = []
            prob_list = []
            target_logit_list = []

            for batch in tqdm(self.loader, total=len(self.loader)):
                images, super_targets, sub_targets = batch
                images = images.to(self.device)
                super_targets = super_targets.to(self.device)
                sub_targets = sub_targets.to(self.device)
                with torch.no_grad():
                    logits = classifier(images)
                    correct = (logits.argmax(dim=-1) == super_targets).reshape(-1,1)
                    loss = F.cross_entropy(logits, super_targets, reduction="none").reshape(-1,1)
                    prob = F.softmax(logits, dim=-1)
                    target_logit = logits[torch.arange(len(images)), super_targets].reshape(-1,1)
                    
                    logits_list.append(logits.cpu())
                    correct_list.append(correct.cpu())
                    loss_list.append(loss.cpu())
                    prob_list.append(prob.cpu())
                    target_logit_list.append(target_logit.cpu())
            
            correct_list_all.append(torch.cat(correct_list, dim=0))
            logits_list_all.append(torch.cat(logits_list, dim=0))
            loss_list_all.append(torch.cat(loss_list, dim=0))
            prob_list_all.append(torch.cat(prob_list, dim=0))
            target_logit_list_all.append(torch.cat(target_logit_list, dim=0))
            
        
        self.correct_list = torch.cat(correct_list_all, dim=1).float()
        self.logits_list = torch.cat(logits_list_all, dim=1)
        self.loss_list = torch.cat(loss_list_all, dim=1)
        self.prob_list_ = torch.cat(prob_list_all, dim=1)
        self.target_logit_list = torch.cat(target_logit_list_all, dim=1)
        
        self.subclass_text_feature_list = []
        for sub_class in self.sub_class_names:
            sub_class_text = f"A photo of a {sub_class}"
            subclass_text_feature = self.clip_model.encode_text(self.tokenizer(sub_class_text).to(self.device))
            subclass_text_feature = F.normalize(subclass_text_feature, dim=1)
            self.subclass_text_feature_list.append(subclass_text_feature.cpu())
        self.subclass_text_feature_list = torch.cat(self.subclass_text_feature_list, dim=0).detach().numpy()


    def _compute_subclass_test_accuracy(self):
        test_correct_list = []
        for batch in tqdm(self.test_loader, total=len(self.test_loader)):
            images, super_targets, sub_targets = batch
            images = images.to(self.device)
            super_targets = super_targets.to(self.device)
            sub_targets = sub_targets.to(self.device)
            with torch.no_grad():
                logits = self.classifier(images)
                correct = (logits.argmax(dim=-1) == super_targets).reshape(-1,1)
                test_correct_list.append(correct.cpu())
        test_correct_list = torch.cat(test_correct_list, dim=0)
        
        sup_sub_class_tuple = ([sup, sub] for sup, subs in self.super_sub_class_dict.items() for sub in subs)
        super_sub_class_test_accuracy = pd.Series(index=pd.MultiIndex.from_tuples(sup_sub_class_tuple, name=['superclass', 'subclass']))
        
        for super_class, sub_classes in self.super_sub_class_dict.items():
            for sub_class in sub_classes:
                sub_class_idx = self.sub_class_names.index(sub_class)
                sub_class_sample_idx = np.array(self.test_loader.dataset.subclass_targets) == sub_class_idx
                sub_class_accuracy = np.average(test_correct_list[sub_class_sample_idx])
                super_sub_class_test_accuracy[(super_class, sub_class)] = sub_class_accuracy
        return super_sub_class_test_accuracy
      
             
    def _compute_superclass_ground_truth(self):
        superclass_ground_truth = dict()
        if self.args.ground_truth == 'subclass_text_feature':
            for i, super_class in enumerate(self.super_class_names):
                super_text_feature = self.clip_model.encode_text(self.tokenizer(f"A photo of a {super_class}").to(self.device))
                
                sub_classes = self.super_sub_class_dict[super_class]
                subclasses_text_features = []
                for sub_class in sub_classes:
                    text = f"A photo of a {sub_class}"
                    text_feature = self.clip_model.encode_text(self.tokenizer(text).to(self.device))
                    # text_feature = text_feature - super_text_feature
                    text_feature = F.normalize(text_feature, dim=1)
                    subclasses_text_features.append(text_feature)
                superclass_ground_truth[super_class] = torch.cat(subclasses_text_features, dim=0)
        elif self.args.ground_truth == 'subclass_image_feature':
            for idx_target, super_class in enumerate(self.super_class_names):
                superclass_sample_idx = self.super_target_list == idx_target
                super_image_feature = torch.from_numpy(self.feature_list[superclass_sample_idx]).mean(dim=0).reshape(1,-1)
                
                sub_classes = self.super_sub_class_dict[super_class]
                subclasses_image_features = []
                for sub_class in sub_classes:
                    sub_class_idx = self.sub_class_names.index(sub_class)
                    sub_class_sample_idx = self.sub_target_list == sub_class_idx
                    image_feature = torch.from_numpy(self.feature_list[sub_class_sample_idx]).mean(dim=0).reshape(1,-1)
                    image_feature = image_feature - super_image_feature
                    image_feature = F.normalize(image_feature, dim=1).to(self.device)
                    subclasses_image_features.append(image_feature)
                superclass_ground_truth[super_class] = torch.cat(subclasses_image_features, dim=0)
        return superclass_ground_truth


    def _compute_component(self):
        self.weight=torch.tensor([])
        NUM_COMPONENTS = self.NUM_COMPONENTS
        component_df = pd.DataFrame(columns=pd.MultiIndex.from_product([self.super_class_names, list(range(self.feature_list.shape[1]))]))
        for idx_target, super_class in enumerate(self.super_class_names):
            print(f"super_class name: {super_class}")
            superclass_sample_idx = self.super_target_list == idx_target
            
            cur_feature_list = self.feature_list[superclass_sample_idx]
            cur_correct_list = self.correct_list[superclass_sample_idx]
            cur_logits_list = self.logits_list[superclass_sample_idx]
            cur_loss_list = self.loss_list[superclass_sample_idx]
            
            self.cur_early_stop_correct_list = cur_correct_list[:, self.args.early_stop_model_epoch]

            
            pls_loss_components = self.pls_components(cur_feature_list, np.hstack([-cur_loss_list, cur_feature_list]), NUM_COMPONENTS)
            component_df = self.store_component(component_df, 'DIM', super_class, pls_loss_components)
            

            
            
            
        return component_df


    def _compute_sim(self):
        sup_sub_class_tuple = ([sup, sub] for sup, subs in self.super_sub_class_dict.items() for sub in subs)
        similarity_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(sup_sub_class_tuple, name=['superclass', 'subclass']))
        for super_class in self.super_class_names:
            for method_name, component_superclass_method  in self.component_df[super_class].groupby(level=0, sort=False):
                component_superclass_method = torch.from_numpy(component_superclass_method.values.astype(float)).to(self.device).to(torch.float32)
                sim_super_class_method = self.compute_sim(component_superclass_method[:self.NUM_SHOW_COMPONENTS], super_class)
                similarity_df = self.store_sim(similarity_df, method_name, super_class, sim_super_class_method)
        return similarity_df


    


            

    def run(self):
        self.feature_list = self._compute_feature()
        self._compute_supervision()
        self.super_sub_class_test_accuracy = self._compute_subclass_test_accuracy()
        self.superclass_ground_truth = self._compute_superclass_ground_truth()
        
        self.NUM_COMPONENTS = 5
        self.NUM_SHOW_COMPONENTS=5
        
        self.component_df = self._compute_component()
        self.component_df.to_csv(f'{self.save_path}/result/component.csv', index=True) 
        
        self.similarity_df = self._compute_sim()
        self.similarity_df.to_csv(f'{self.save_path}/result/sim_({self.args.ground_truth}).csv', index=True) 




    def pls_components(self, feature, supervision, num_component):
        pls = PLSRegression(n_components=num_component)
        pls.fit(feature, supervision)
        pls_components = pls.x_rotations_.T / pls._x_std # n_components x n_feat
        
        
        sim = (F.normalize(torch.tensor(feature)-feature.mean(axis=0)) @ F.normalize(torch.tensor(pls_components)).float().T).numpy()
        

        groups = np.abs(sim).argmax(axis=1)
        group_acc = np.array([self.cur_early_stop_correct_list[groups==g].sum()/(groups==g).sum() for g in range(num_component)])
        pls_components = pls.x_rotations_.T
        pls_components = pls_components[group_acc.argsort()]
        
        pls_components = torch.from_numpy(pls_components).to(self.device).float()
        
        return pls_components


    


    def compute_sim(self, components, super_class):
        components = F.normalize(components, dim=-1)
        super_class_feature = F.normalize(self.superclass_ground_truth[super_class], dim=-1)
        sim = components @ super_class_feature.t()
        sim = sim.abs().tolist()
        return sim


    def store_sim(self, similarity_df, name, super_class, sim):
        multiindex = pd.MultiIndex.from_tuples(([name, 'component_' + str(i)] for i in range(len(sim))))
        if name not in similarity_df.index.get_level_values(0): 
            similarity_df_temp = pd.DataFrame(None, index=multiindex, columns=similarity_df.loc[:,([super_class],)].columns)
            similarity_df = pd.concat([similarity_df, similarity_df_temp], axis=0)
            similarity_df.index = pd.MultiIndex.from_tuples(similarity_df.index)
        similarity_df.loc[multiindex, (super_class,)] = sim
        return similarity_df
   
    
    def store_component(self, component_df, name, super_class, components):
        multiindex = pd.MultiIndex.from_tuples(([name, 'component_' + str(i)] for i in range(len(components))))
        if name not in component_df.index.get_level_values(0): 
            component_df_temp = pd.DataFrame(None, index=multiindex, columns=component_df.loc[:,([super_class],)].columns)
            component_df = pd.concat([component_df, component_df_temp], axis=0)
            component_df.index = pd.MultiIndex.from_tuples(component_df.index)
        component_df.loc[multiindex, (super_class,)] = components.cpu()
        return component_df
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model_name", type=str, default='ViT-B-32', choices=['ViT-B-32','ViT-L-14'])
    parser.add_argument("--early_stop_model_epoch", type=int, default='10')
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--model_path", type=str, default='save/001_train_model/resnet18_imbalancedsupercifar100.pt')
    parser.add_argument("--dataset_name", type=str, default='imbalancedsupercifar100')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ground_truth", type=str, default='subclass_text_feature', 
                        choices=['subclass_image_feature', 'subclass_text_feature'])
    parser.add_argument("--match", type=str, default='maximize_sim_sum')
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    
    save_path = 'save/002_multi_bias_identification'
    trainer = Trainer(args, save_path)
    trainer.run()

    
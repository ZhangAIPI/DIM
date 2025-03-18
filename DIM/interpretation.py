import sys
sys.path.append('.')
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import requests
import json
import time
import argparse
import os
from .identification import Trainer
from dataloader import DataLoaderCreator
import open_clip
from clip_retrieval.clip_client import ClipClient, Modality
from PIL import Image
from torchvision import transforms
from BLIP.models.blip import blip_decoder


class ComponentRetriever(object):
    
    def __init__(self, args, component_df_path, save_path):
        self.args = args
        self.device = torch.device(self.args.device)
        self.save_path = save_path
        
        self.clip_model, self.preprocess, self.tokenizer = self._build_clip_model()
        self.train_loader_for_clip = self._build_data_loader()
        
        self.sub_class_names = self.train_loader_for_clip.dataset.subclasses
        self.super_class_names = self.train_loader_for_clip.dataset.classes
        self.super_sub_class_dict = self.train_loader_for_clip.dataset.super_sub_class_dict
        self.super_target_list = np.array(self.train_loader_for_clip.dataset.targets)
        self.sub_target_list = np.array(self.train_loader_for_clip.dataset.subclass_targets)
        
        self.superclass_feature_dict = self._compute_superclass_feature_dict()
        self.components = pd.read_csv(component_df_path, index_col=[0,1], header=[0,1])
        self.client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", aesthetic_score=9, 
                    aesthetic_weight=0.5, modality=Modality.IMAGE, num_images=10)


    def _build_clip_model(self):
        clip_model_name = self.args.clip_model_name
        clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained="openai")
        clip_model.eval()
        clip_model = clip_model.to(self.device)
        tokenizer = open_clip.get_tokenizer(clip_model_name)
        return clip_model, preprocess, tokenizer
        

    def _build_data_loader(self):
        dataloader_creator = DataLoaderCreator(self.args, auto_create=False)
        train_loader_for_clip = dataloader_creator.train_data_loader(transform=self.preprocess, shuffle=False)
        return train_loader_for_clip


    def _compute_feature(self):
        feature_list = []
        for clip_batch in tqdm(self.train_loader_for_clip, total=len(self.train_loader_for_clip)):
            images_for_clip, _, _ = clip_batch
            images_for_clip = images_for_clip.to(self.device)
            with torch.no_grad():
                features = self.clip_model.encode_image(images_for_clip)
                feature_list.append(features.cpu())
        feature_list = torch.cat(feature_list, dim=0).numpy()
        return feature_list
    
    
    def _compute_superclass_feature_dict(self):
        super_class_feature_dict = dict()
        if self.args.constrain == 'image':
            self.feature_list = self._compute_feature()
            for idx_target, super_class in enumerate(self.super_class_names):
                superclass_sample_idx = self.super_target_list == idx_target
                super_image_feature = torch.tensor(self.feature_list[superclass_sample_idx]).mean(dim=0)
                super_image_feature = F.normalize(super_image_feature, dim=-1).cpu()
                super_class_feature_dict[super_class] = super_image_feature
        elif self.args.constrain == 'text':
            for super_class in self.super_class_names:
                text = f"A photo of a {super_class}"
                super_text_feature = self.clip_model.encode_text(self.tokenizer(text).to(self.device)).reshape(-1)
                super_text_feature = F.normalize(super_text_feature, dim=-1).cpu()
                super_class_feature_dict[super_class] = super_text_feature
        else:
            for super_class in self.super_class_names:
                super_class_feature_dict[super_class] = 0
        return super_class_feature_dict


    def download_save_retrieval_result(self, retrieval_results, save_path):
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(f'{save_path}/meta', exist_ok=True)
        for img_id, result in enumerate(retrieval_results):
            try:
                res = requests.get(result['url'])
            except:
                continue
            with open(f'{save_path}/img_{img_id}.jpg', 'wb') as file:
                file.write(res.content)
            with open(f'{save_path}/meta/img_{img_id}.json', 'w') as file:
                json.dump(result, file)
            time.sleep(1)


    def retrieval(self, embedding, save_path):
        embedding = F.normalize(embedding, dim=-1).tolist()
        try:
            print('start_retrevial')
            retrieval_results = self.client.query(embedding_input=embedding)
            print(len(retrieval_results))
            self.download_save_retrieval_result(retrieval_results, save_path)
        except KeyboardInterrupt: 
            raise
        except: 
            pass


    def run(self, done_list=[], methods=[]):
        print('superclass constrained clip retrieval')
        for super_class, components_superclass  in self.components.groupby(axis=1, level=0):
            if super_class in done_list:
                continue
            print('--'*20)
            for method_component, component_superclass_mothod in components_superclass.iterrows():
                method, component_id = method_component
                if method not in methods:
                    continue
                print(super_class, method, component_id)
                
                save_path_methodlevel = os.path.join(self.save_path, self.args.constrain+'_constrain', super_class, method.replace("/","-"))
                superclass_feature = self.superclass_feature_dict[super_class]
                component_superclass_mothod = torch.tensor(component_superclass_mothod.values).float()
                component_superclass_mothod = F.normalize(component_superclass_mothod, dim=-1)
                
                embedding_positive = superclass_feature + component_superclass_mothod
                save_path_positive = os.path.join(save_path_methodlevel, component_id)
                self.retrieval(embedding_positive, save_path_positive)

                embedding_negative = superclass_feature - component_superclass_mothod
                save_path_negative = os.path.join(save_path_methodlevel, component_id+'_n')
                self.retrieval(embedding_negative, save_path_negative)


class ImageCaptioner(object):
    
    def __init__(self, images_path):
        self.images_path = images_path
        self.image_size = 384
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
        self.caption_model = blip_decoder(pretrained=model_url, image_size=self.image_size, vit='base').to(self.device)
        self.caption_model.eval()
    

    def load_images(self, dir_path, image_size, device):
        image_name_list = []
        image_list = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                path = os.path.join(dir_path, filename)
                try:
                    # import pdb; pdb.set_trace()
                    raw_image = Image.open(path).convert('RGB')
                    raw_image = raw_image.resize((image_size, image_size))
                    # import pdb; pdb.ssset_trace()
                    raw_image = transforms.ToTensor()(raw_image)
                    image_list.append(raw_image)
                    image_name_list.append(filename)
                except:
                    print("Error: ", path)
                    continue
            else:
                continue
        image_list = torch.stack(image_list)
        
        # w,h = raw_image.size
        # display(raw_image.resize((w//5,h//5)))
        
        transform = transforms.Compose([
            # transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            # transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image_list = transform(image_list).to(device)
        return image_name_list, image_list


    def run(self, images_path=None, superclasses=None, methods=None):
        images_path = images_path or self.images_path
        
        image_superclasses = sorted(os.listdir(images_path))
        if superclasses:
            image_superclasses = list(set(image_superclasses) & set(superclasses))
        for super_class in image_superclasses:
            superclass_level_path = os.path.join(images_path, super_class)
            superclass_methods = sorted(os.listdir(superclass_level_path))
            if methods:
                superclass_methods = list(set(superclass_methods) & set(methods))
            for method in superclass_methods:
                method_level_path = os.path.join(superclass_level_path, method)
                for component_id in sorted(os.listdir(method_level_path)):
                    print(super_class, method, component_id)
                    component_level_path = os.path.join(method_level_path, component_id)
                    image_name_list, image = self.load_images(component_level_path, self.image_size, self.device)
                    with torch.no_grad():
                        captions = self.caption_model.generate(image, sample=True, num_beams=3, max_length=20, min_length=5) # beam search
                        # caption = caption_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) # nucleus sampling
                    captions_series = pd.Series(captions, index=image_name_list)
                    captions_series.to_csv(os.path.join(component_level_path, 'caption.csv'))
                    print(captions_series)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model_name", type=str, default='ViT-B-32', choices=['ViT-B-32','ViT-L-14'])
    parser.add_argument("--dataset_name", type=str, default='imbalancedsupercifar100')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--constrain", type=str, default='text', choices=['text', 'image', 'none'])
    args = parser.parse_args()
    
    
    images_path = f'save/component_interpretion/clip_retrieval/{args.constrain}_constrain'
    image_captioner = ImageCaptioner(images_path)
    image_captioner.run()

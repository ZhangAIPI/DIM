import os
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms


CIFAR_100_CLASS_MAP = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": ["apple","mushroom","orange","pear","sweet_pepper"],
    "household electrical devices": ["clock","keyboard","lamp","telephone","television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge","castle","house","road","skyscraper"],
    "large natural outdoor scenes": [ "cloud","forest","mountain","plain","sea"],
    "large omnivores and herbivores": ["camel","cattle","chimpanzee","elephant","kangaroo",],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree","oak_tree","palm_tree","pine_tree","willow_tree"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}


class AugmentSupercifar100(torch.utils.data.Dataset):
    def __init__(self, path, method_name, component_num, images_num_per_component=100):
        # path = 'save/005_component_interpretion/clip_retrieval/text_constrain'
        self.path = path
        self.method_name = method_name.replace('/', '-')
        self.super_sub_class_dict = CIFAR_100_CLASS_MAP
        self.class_names = sorted(list(CIFAR_100_CLASS_MAP.keys()))
        
        self.data_paths = []
        self.targets = []
        
        for class_name in self.class_names:
            print(class_name)
            class_level_path = os.path.join(path, class_name)
            assert os.path.exists(class_level_path)
            method_level_path = os.path.join(class_level_path, method_name)
            assert os.path.exists(method_level_path)
            for i in range(component_num):
                print(i)
                component_level_path = os.path.join(method_level_path, f'component_{i}')
                assert os.path.exists(component_level_path)
                class_data_paths = []
                for filename in os.listdir(component_level_path):
                    if len(class_data_paths) <=images_num_per_component:
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            class_data_paths.append(filename)
                self.data_paths += [os.path.join(component_level_path, class_data_path) for class_data_path in class_data_paths]
                self.targets += [self.class_names.index(class_name)]*len(class_data_paths)
        
        CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
        CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
        self.transform = transforms.Compose([transforms.ToTensor(), 
                                             transforms.Resize((32, 32)),
                                             transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    def __getitem__(self, index):
        data = cv2.imread(self.data_paths[index])
        target = self.targets[index]
        return self.transform(data), target, None
    
    def __len__(self):
        return len(self.targets)

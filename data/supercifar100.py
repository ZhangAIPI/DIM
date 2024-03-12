import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100


CIFAR_100_CLASS_MAP = {
            'aquatic mammals':	['beaver', 'dolphin', 'otter', 'seal', 'whale'],
            'fish':	['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
            'flowers':	['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
            'food containers':	['bottle', 'bowl', 'can', 'cup', 'plate'],
            'fruit and vegetables':	['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
            'household electrical devices':	['clock', 'keyboard', 'lamp', 'telephone', 'television'],
            'household furniture':	['bed', 'chair', 'couch', 'table', 'wardrobe'],
            'insects':	['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
            'large carnivores':	['bear', 'leopard', 'lion', 'tiger', 'wolf'],
            'large man-made outdoor things':	['bridge', 'castle', 'house', 'road', 'skyscraper'],
            'large natural outdoor scenes':	['cloud', 'forest', 'mountain', 'plain', 'sea'],
            'large omnivores and herbivores':	['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
            'medium-sized mammals':	['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
            'non-insect invertebrates':	['crab', 'lobster', 'snail', 'spider', 'worm'],
            'people':	['baby', 'boy', 'girl', 'man', 'woman'],
            'reptiles':	['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
            'small mammals':	['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
            'trees':	['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
            'vehicles 1':	['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
            'vehicles 2':	['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


class SuperCIFAR100(Dataset):
    def __init__(self, **kwargs):
        self.ds = CIFAR100(**kwargs)
        
        self.super_sub_class_dict = CIFAR_100_CLASS_MAP
        self.classes = sorted(list(CIFAR_100_CLASS_MAP.keys()))
        self.subclasses = self.ds.classes
        
        self.subid_to_supid_dict = {}
        self.supid_to_subid_dict = {}
        for i, superclass in enumerate(self.classes):
            for subclass in CIFAR_100_CLASS_MAP[superclass]:
                self.subid_to_supid_dict[self.subclasses.index(subclass)] = i
            self.supid_to_subid_dict[i] = [self.subclasses.index(subclass) for subclass in CIFAR_100_CLASS_MAP[superclass]]
        
        self.targets = [self.subid_to_supid_dict[u] for u in self.ds.targets]
        self.groups = self.ds.targets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        target = self.targets[idx]
        group = self.groups[idx]
        return x, target, group

import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd


import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset



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





bottom_two_subclass_indices_list = [
    [4, 55],
    [73, 1],
    [54, 92],
    [10, 16],
    [51, 57],
    [22, 40],
    [84, 25],
    [18, 14],
    [3, 97],
    [37, 12],
    [33, 49],
    [38, 31],
    [66, 64],
    [45, 77],
    [46, 98],
    [78, 93],
    [74, 80],
    [96, 56],
    [13, 8],
    [81, 89],
]

bottom_two_subclass_names = [
    ["beaver", "otter"],
    ["shark", "aquarium_fish"],
    ["orchid", "tulip"],
    ["bowl", "can"],
    ["mushroom", "pear"],
    ["clock", "lamp"],
    ["table", "couch"],
    ["caterpillar", "butterfly"],
    ["bear", "wolf"],
    ["house", "bridge"],
    ["forest", "mountain"],
    ["kangaroo", "elephant"],
    ["raccoon", "possum"],
    ["lobster", "snail"],
    ["man", "woman"],
    ["snake", "turtle"],
    ["shrew", "squirrel"],
    ["willow_tree", "palm_tree"],
    ["bus", "bicycle"],
    ["streetcar", "tractor"],
]


class ImbalancedSuperCIFAR100(Dataset):
    def __init__(self, split, **kwargs):
        assert split in ["train", "val", "test", "intervention"]
        super(ImbalancedSuperCIFAR100, self).__init__()
        is_original_train = split != "test"
        kwargs["train"] = is_original_train
        ds = CIFAR100(**kwargs)

        self.super_sub_class_dict = CIFAR_100_CLASS_MAP
        self.classes = sorted(list(CIFAR_100_CLASS_MAP.keys()))
        self.subclasses = ds.classes
        self.hundreds_to_20_map = {}
        for i, k in enumerate(self.classes):
            for v_ in CIFAR_100_CLASS_MAP[k]:
                self.hundreds_to_20_map[self.subclasses.index(v_)] = i

        self.subclass_targets = ds.targets
        self.targets = [self.hundreds_to_20_map[u] for u in ds.targets]
        self.ds = ds

        bottom_two_subclass_indices = set()
        for idx_1, idx_2 in bottom_two_subclass_indices_list:
            bottom_two_subclass_indices.add(idx_1)
            bottom_two_subclass_indices.add(idx_2)

        if is_original_train:
            indices = []
            for idx_subclass in range(len(self.subclasses)):
                sub_class_mask = torch.tensor(self.subclass_targets) == idx_subclass
                sub_class_indices = torch.nonzero(sub_class_mask).squeeze()

                # 40% for train; 40% for intervention; 20% for val
                num_fourty_percent = int(len(sub_class_indices) * 0.4)
                num_eighty_percent = int(len(sub_class_indices) * 0.8)
                if split == "train":
                    sub_class_indices = sub_class_indices[:num_fourty_percent]

                    # for bottom two subclasses, only keep 25% of the data for training
                    if idx_subclass in bottom_two_subclass_indices:
                        sub_class_indices = sub_class_indices[:int(len(sub_class_indices) * 0.25)]

                elif split == "intervention":
                    sub_class_indices = sub_class_indices[num_fourty_percent:num_eighty_percent]
                elif split == "val":
                    sub_class_indices = sub_class_indices[num_eighty_percent:]
                else:
                    raise ValueError("Unknown split: {}".format(split))

                indices.extend(sub_class_indices.tolist())
                
            torch_indices = torch.tensor(indices)
            self.subclass_targets = torch.tensor(self.subclass_targets)[torch_indices].tolist()
            self.targets = torch.tensor(self.targets)[torch_indices].tolist()
            self.ds = Subset(self.ds, indices)
            

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, self.hundreds_to_20_map[y], y




class DataLoaderCreator(object):
    def __init__(self, args=None):
        self.args = args
        if self.args:
            self.train = self.train_data_loader()
            self.predict = self.predict_data_loader()

    def train_data_loader(self, dataset_name=None, batch_size=None, transform=None, shuffle=None, sampler=None):
        dataset_name = dataset_name or self.args.dataset_name
        batch_size = batch_size or self.args.batch_size
        shuffle = shuffle or True
        if dataset_name == 'mnist':
            if transform is None:
                transform = transforms.Compose([transforms.RandomRotation(20),
                                                transforms.RandomAffine(0, translate=(0.2, 0.2)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST('../../data/ABC/mnist', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif dataset_name == 'cifar10':
            if transform is None:
                transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=[0.75, 4/3]),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandAugment(num_ops=2, magnitude=9),
                                                transforms.ColorJitter(0.4,0.4,0.4),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                transforms.RandomErasing(p=0.25),])
            dataset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        elif dataset_name == 'cifar100':
            if transform is None:
                transform = transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=[0.75, 4/3]),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandAugment(num_ops=2, magnitude=9),
                                                transforms.ColorJitter(0.4,0.4,0.4),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
                                                transforms.RandomErasing(p=0.25),])
            dataset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        elif dataset_name == 'supercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            trainset = SuperCIFAR100(root="data", train=True, transform=transform, download=True)
            train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,)
        elif dataset_name == 'imbalancedsupercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                    transforms.ToTensor(), 
                                                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            trainset = ImbalancedSuperCIFAR100(split='train', root="data", train=True, transform=transform, download=True)
            train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,)
        return train

    def predict_data_loader(self, dataset_name=None, batch_size=None, transform=None):
        dataset_name = dataset_name or self.args.dataset_name
        batch_size = batch_size or self.args.batch_size
        if dataset_name == 'mnist':
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])
            testset = datasets.MNIST('../../data/ABC/mnist', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=batch_size, shuffle=False)
        elif dataset_name == "cifar10":
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            testset = datasets.CIFAR10(root='../../data/ABC/CIFAR10', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        elif dataset_name == "cifar100":
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),])
            testset = datasets.CIFAR100(root='../../data/ABC/CIFAR100', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        elif dataset_name == 'supercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            test_set = SuperCIFAR100(root="data", train=False, transform=transform, download=True)
            predict = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        elif dataset_name == 'imbalancedsupercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            test_set = ImbalancedSuperCIFAR100(split='test',root="data", train=False, transform=transform, download=True)
            predict = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return predict
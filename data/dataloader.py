import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import default_collate
from torch.utils.data import Dataset,DataLoader
import gc
import pandas as pd
from data.supercifar100 import SuperCIFAR100
from data.imbalanced_supercifar100 import ImbalancedSuperCIFAR100

class DataLoaderCreator(object):
    def __init__(self, args=None, auto_create=True):
        self.args = args
        self.data_path = '/scratch/mfeng7/data'
        if self.args and auto_create:
            self.train = self.train_data_loader()
            self.predict = self.predict_data_loader()

    def train_data_loader(self, dataset_name=None, batch_size=None, transform=None, shuffle=None):
        dataset_name = dataset_name or self.args.dataset_name
        batch_size = batch_size or self.args.batch_size
        if shuffle is None:
            shuffle = True
        if dataset_name == 'mnist':
            if transform is None:
                transform = transforms.Compose([transforms.RandomRotation(20),
                                                transforms.RandomAffine(0, translate=(0.2, 0.2)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.1307,), (0.3081,))])
            dataset = datasets.MNIST(f'{self.data_path}/mnist', train=True, download=True, transform=transform)
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
            dataset = datasets.CIFAR10(root=f'{self.data_path}/CIFAR10', train=True, download=True, transform=transform)
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
            dataset = datasets.CIFAR100(root=f'{self.data_path}/CIFAR100', train=True, download=True, transform=transform)
            train = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        elif dataset_name == 'supercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            trainset = SuperCIFAR100(root=f'{self.data_path}/CIFAR100', train=True, transform=transform, download=True)
            train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,)
        elif dataset_name == 'imbalancedsupercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([
                                                # transforms.RandomResizedCrop(size=32, scale=(0.75, 1.0), ratio=[0.75, 4/3]),
                                                transforms.RandomHorizontalFlip(),
                                                # transforms.RandomVerticalFlip(),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                                                # transforms.RandomErasing(p=0.25),
                                                ])
                # transform = transforms.Compose([transforms.ToTensor(), 
                #                                 transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                #                                 ])
                # from torchvision.transforms import v2
                # cutmix = v2.CutMix(num_classes=20)
                # mixup = v2.MixUp(num_classes=20)
                # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
                # def collate_fn(batch):
                #     return cutmix_or_mixup(*default_collate(batch))
                collate_fn = None
            else:
                collate_fn = None
            trainset = ImbalancedSuperCIFAR100(split='train', root=f'{self.data_path}/CIFAR100', 
                                               train=True, transform=transform, download=True)
            train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        elif dataset_name == 'imbalancedsupercifar100_validation':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
            trainset = ImbalancedSuperCIFAR100(split='val', root=f'{self.data_path}/CIFAR100', 
                                               train=True, transform=transform, download=True)
            train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
        elif dataset_name == 'imbalancedsupercifar100_intervention':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
            trainset = ImbalancedSuperCIFAR100(split='intervention', root=f'{self.data_path}/CIFAR100', 
                                               train=True, transform=transform, download=True)
            train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
        return train
    

    def predict_data_loader(self, dataset_name=None, batch_size=None, transform=None):
        dataset_name = dataset_name or self.args.dataset_name
        batch_size = batch_size or self.args.batch_size
        if dataset_name == 'mnist':
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])
            testset = datasets.MNIST(f'{self.data_path}/mnist', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=batch_size, shuffle=False)
        elif dataset_name == "cifar10":
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            testset = datasets.CIFAR10(root=f'{self.data_path}/CIFAR10', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        elif dataset_name == "cifar100":
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),])
            testset = datasets.CIFAR100(root=f'{self.data_path}/CIFAR100', train=False, download=True, transform=transform)
            predict = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        elif dataset_name == 'supercifar100':
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            test_set = SuperCIFAR100(root=f'{self.data_path}/CIFAR100', train=False, transform=transform, download=True)
            predict = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        elif dataset_name in ['imbalancedsupercifar100', 'imbalancedsupercifar100_validation', 'imbalancedsupercifar100_intervetion']:
            if transform is None:
                CIFAR_MEAN = [x / 255.0 for x in [125.307, 122.961, 113.8575]]
                CIFAR_STD = [x / 255.0 for x in [51.5865, 50.847, 51.255]]
                transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),])
            test_set = ImbalancedSuperCIFAR100(split='test',root=f'{self.data_path}/CIFAR100', 
                                               train=False, transform=transform, download=True)
            predict = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return predict
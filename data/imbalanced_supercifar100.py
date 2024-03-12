import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset


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
        
        # self.hundreds_to_20_map = {}
        # for i, k in enumerate(self.classes):
        #     for v_ in CIFAR_100_CLASS_MAP[k]:
        #         self.hundreds_to_20_map[self.subclasses.index(v_)] = i

        self.subid_to_supid_dict = {}
        self.supid_to_subid_dict = {}
        for i, superclass in enumerate(self.classes):
            for subclass in CIFAR_100_CLASS_MAP[superclass]:
                self.subid_to_supid_dict[self.subclasses.index(subclass)] = i
            self.supid_to_subid_dict[i] = [self.subclasses.index(subclass) for subclass in CIFAR_100_CLASS_MAP[superclass]]

        self.subclass_targets = ds.targets
        self.targets = [self.subid_to_supid_dict[u] for u in ds.targets]
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

                    # for bottom two subclasses, only keep 25% of the splited data for training
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
            
        self.groups = self.subclass_targets

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        target = self.targets[idx]
        group = self.groups[idx]
        return x, target, group

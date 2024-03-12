import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
from utils.tools import dotdict
from model.convmixer import ConvMixer

class ModelCreator(object):
    def __init__(self, args=None):
        self.args = args
        self.model = self.create_model()
        
    def create_model(self, model_name=None, num_classes=None):
        model_name = model_name or self.args.model
        num_classes = num_classes or self.args.num_classes
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        elif model_name == 'resnet34':
            model = torchvision.models.resnet34(weights=None, num_classes=num_classes)
        elif model_name == 'convmixer':
            model = ConvMixer(256,16,8,1, num_classes)
        return model
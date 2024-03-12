import sys
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from driver.driver import DLDriver


class DLDriver_save_epoch(DLDriver):
    
    def train(self):
        epochs = self.args.train_epochs
        train_loader = self.train_data_loader
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler
        
        for epoch in tqdm(range(epochs), total=epochs):
            train_loss = self.epoch_train(train_loader, model, criterion, optimizer, scheduler, self.scheduler_step_after_batch, self.device[0])
            if self.args.model_save_path:
                os.makedirs(self.args.model_save_path[:-3], exist_ok=True)
                torch.save(model.state_dict(), os.path.join(self.args.model_save_path[:-3], f'epoch{epoch}.pt'))
    
        if self.args.model_save_path:
            torch.save(model.state_dict(), self.args.model_save_path)



if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, default='record/records/record.csv')
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--model_save_path", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--dataset_name", type=str, default='imbalancedsupercifar100')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--lr", type=int, default=0.1)
    parser.add_argument("--criterion", type=str, default='CE')
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--scheduler", type=str, default='none')
    args = parser.parse_args()
    setattr(args, 'model_save_path', f"save/001_train_model/{args.model}_{args.dataset_name}.pt")
    
    driver = DLDriver_save_epoch(args)
    driver.train()
    
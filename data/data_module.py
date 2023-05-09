import os
import torch 

import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from torch.nn.functional import pad
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from data.my_dataset import MyDataset

class AugmentedDataModule(pl.LightningDataModule):
    '''
    Data Module for the augmented data prepared by Speech_Augment
    '''

    def __init__(self, root, config, batch_size=8):
        self.batch_size = batch_size
        self.root = root
        self.config = config
        self.prepare_data_per_node = False

        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config.model.model)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.config.model.model)
        

    def setup(self, stage=None):
        self.train = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.train = MyDataset(self.train)

    def train_dataloader(self):
        # def pad_collate(batch):
        #     (x, sr, len, y) = zip(*batch)
        #     max_len = max(len)
        #     x_pad = []
        #     for i in x:
        #         p1d = (0, max_len - i.size(dim=1))
        #         i = pad(i, p1d, 'constant', 0)
        #         x_pad.append(i)
            
        #     return torch.squeeze(torch.stack(x_pad, dim=0)), sr[0], y

        def pad_collate(batch):
            (x, sr, y) = zip(*batch)
            print(x)
            x_fe = self.feature_extractor(x, sr[0]).input_features[0]
            y_to = self.tokenizer(y).input_ids
            
            return torch.stack(x_fe, dim=0), sr[0], y_to
    
        train_dl = DataLoader(self.train, batch_size=self.batch_size, collate_fn=pad_collate)

        return train_dl
import os
import torch 

import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from data.my_dataset import MyDataset

class AugmentedDataModule(pl.LightningDataModule):
    '''
    Data Module for the augmented data prepared by Speech_Augment
    '''

    def __init__(self, root, config, batch_size=2):
        self.batch_size = batch_size
        self.root = root
        self.config = config
        self.prepare_data_per_node = False

        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.processor = WhisperProcessor.from_pretrained(self.config.model.whisper)        

    def setup(self, stage=None):
        self.train = pd.read_csv(os.path.join(self.root, 'train.csv'))     
        self.train = MyDataset(self.train)

    def train_dataloader(self):
        def pad_collate(batch):
            (x, sr, y) = zip(*batch)
            x = self.processor.feature_extractor(x, sampling_rate=sr[0], return_tensors='pt').input_features
            y = torch.Tensor(list(y))
            
            return x, sr[0], y
    
        train_dl = DataLoader(self.train, batch_size=self.batch_size, collate_fn=pad_collate)

        return train_dl
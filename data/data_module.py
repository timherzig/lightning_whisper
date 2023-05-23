import os
import torch

import pandas as pd
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from data.my_dataset import MyDataset


class AugmentedDataModule(pl.LightningDataModule):
    """
    Data Module for the augmented data prepared by Speech_Augment
    """

    def __init__(self, root, config, batch_size=2):
        self.batch_size = batch_size
        self.root = root
        self.config = config
        self.prepare_data_per_node = False

        self.save_hyperparameters()
        self.allow_zero_length_dataloader_with_multiple_devices = False

        self.processor = WhisperProcessor.from_pretrained(self.config.model.whisper)

    def pad_collate(self, batch):
        (x, sr, y) = zip(*batch)
        x = self.processor.feature_extractor(
            x, sampling_rate=sr[0], return_tensors="pt"
        ).input_features
        y = torch.Tensor(list(y))

        return x, sr[0], y

    def setup(self, stage=None):
        self.train = pd.read_csv(os.path.join(self.root, "train.csv"))
        self.train = MyDataset(self.train)
        self.test = pd.read_csv(os.path.join(self.root, "test.csv"))
        self.test = MyDataset(self.test)
        self.val = pd.read_csv(os.path.join(self.root, "val.csv"))
        self.val = MyDataset(self.val)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, collate_fn=self.pad_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, collate_fn=self.pad_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, collate_fn=self.pad_collate
        )

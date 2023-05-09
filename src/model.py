import torch

import lightning.pytorch as pl
import torch.nn.functional as F

from torch import nn
from transformers import WhisperForConditionalGeneration


class Whisper(pl.LightningModule):
    '''
    Simple model implementation for testing purposes
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.model.model)
        self.model.config.forces_decoder_ids = None

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return opt

    def training_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.model.generate(x)
        print(out)

        loss = F.mse_loss(out, y)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, sr, y = batch

        out = self.model.generate(x)
        print(out)

        loss = F.mse_loss(out, y)
        return loss
    
    def forward(self, x):
        x = self.processor(x, sampling_rate=sr, return_tensors="pt").input_features
        out = self.model.generate(x)
        return self.processor.batch_decode(out, skip_special_tokens=True)
import torch

import lightning.pytorch as pl
import torch.nn.functional as F

from torch import nn
from torchmetrics import Accuracy, F1Score
from transformers import WhisperModel, WhisperProcessor


class Whisper(pl.LightningModule):
    '''
    Simple model implementation for testing purposes
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_acc = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        self.valid_acc = Accuracy(task='binary')
        self.valid_f1 = F1Score(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.test_f1 = F1Score(task='binary')

        self.processor = WhisperProcessor.from_pretrained(self.config.model.whisper) 

        self.whisper_encoder = WhisperModel.from_pretrained(self.config.model.whisper).get_encoder()
        self.whisper_encoder.config.forces_decoder_ids = None
        self.whisper_encoder._freeze_parameters()

        # self.lin1 = nn.Linear(in_features=self.whisper_encoder.embed_positions.embedding_dim, out_features=1)
        # self.lin2 = nn.Linear(in_features=1500, out_features=1)
        self.bilstm = nn.LSTM(self.whisper_encoder.embed_positions.embedding_dim, self.config.model.bilstm.hidden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(192000, 256) #self.config.model.bilstm.hidden_size*4, 512)
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, sr=16000):
        # x = self.processor.feature_extractor(x, sampling_rate=sr, return_tensors='pt').input_features
        print(f'Before size: {x.size()}')
        out = self.whisper_encoder(x).last_hidden_state
        print(f'After Whisper Encoder size: {out.size()}')
        out, _ = self.bilstm(out)
        out = torch.flatten(out, start_dim=1)
        print(f'After BILSTM size: {out.size()}')
        out = self.linear1(out)
        print(f'After Linear1 size: {out.size()}')
        out = self.linear2(out)
        print(f'After Linear2 size: {out.size()}')
        out = torch.squeeze(self.sigmoid(out))
        print(f'After Sigmoid size: {out.size()}')
        return out

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return opt

    def training_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.forward(x)

        self.train_acc.update(out, y)
        self.train_f1.update(out, y)
        loss = F.binary_cross_entropy(out, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.forward(x)

        self.valid_acc.update(out, y)
        self.valid_f1.update(out, y)
        loss = F.binary_cross_entropy(out, y)
        self.log('valid_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.forward(x)

        self.test_acc.update(out, y)
        self.test_f1.update(out, y)
        loss = F.binary_cross_entropy(out, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

import torch

import lightning.pytorch as pl
import torch.nn.functional as F

from torch import nn
from torchmetrics import Accuracy, F1Score
from transformers import WhisperModel, WhisperProcessor


class Whisper(pl.LightningModule):
    """
    Simple model implementation for testing purposes
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.train_acc = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.valid_acc = Accuracy(task="binary")
        self.valid_f1 = F1Score(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.processor = WhisperProcessor.from_pretrained(self.config.model.whisper)

        self.whisper_encoder = WhisperModel.from_pretrained(
            self.config.model.whisper
        ).get_encoder()
        self.whisper_encoder.config.forces_decoder_ids = None
        self.whisper_encoder._freeze_parameters()

        self.bilstm = nn.LSTM(
            self.whisper_encoder.embed_positions.embedding_dim,
            self.config.model.bilstm.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(
            2
            * self.whisper_encoder.embed_positions.num_embeddings
            * self.bilstm.hidden_size,
            256,
        )
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sr=16000):
        out = self.whisper_encoder(x).last_hidden_state
        out, _ = self.bilstm(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = torch.squeeze(self.sigmoid(out))
        return out

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return opt

    def training_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.forward(x)

        acc = self.train_acc(out, y)
        f1 = self.train_f1(out, y)
        loss = F.binary_cross_entropy(out, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.forward(x)

        acc = self.valid_acc(out, y)
        f1 = self.valid_f1(out, y)
        loss = F.binary_cross_entropy(out, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", acc, prog_bar=True)
        self.log("valid_f1", f1, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, sr, y = batch

        out = self.forward(x)

        acc = self.test_acc(out, y)
        f1 = self.test_f1(out, y)
        loss = F.binary_cross_entropy(out, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)
        self.log("test_f1", f1, prog_bar=True, sync_dist=True)
        return loss

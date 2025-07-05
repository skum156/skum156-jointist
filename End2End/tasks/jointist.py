"""
Instrument Recognition +
Transcription
"""

import torch
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl

from omegaconf import OmegaConf
import pandas as pd

class Jointist(pl.LightningModule):
    def __init__(
        self,
        detection_model: pl.LightningModule,
        transcription_model: pl.LightningModule,
        lr_lambda,
        cfg
    ):
        """
        Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.
        """
        super().__init__()
        self.detection_model = detection_model
        self.transcription_model = transcription_model
        self.lr_lambda = lr_lambda
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        detection_loss = self.detection_model.training_step(batch, batch_idx, self)
        transcription_loss = self.transcription_model.training_step(batch, batch_idx, self)

        total_loss = transcription_loss + detection_loss
        self.log('Total_Loss/Train', total_loss, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs, detection_loss = self.detection_model.validation_step(batch, batch_idx, self)
        transcription_loss = self.transcription_model.validation_step(batch, batch_idx, self)

        total_loss = transcription_loss + detection_loss
        self.log('Total_Loss/Valid', total_loss, on_step=False, on_epoch=True)

        return outputs, detection_loss

    def validation_epoch_end(self, outputs):
        self.detection_model.validation_epoch_end(outputs, self)

    def test_step(self, batch, batch_idx):
        # Call detection model directly instead of predict_step
        detection_out = self.detection_model(batch)
        # Provide detection_out (or a relevant part of it) to transcription
        return self.transcription_model.test_step(batch, batch_idx, detection_out, self)

    def test_epoch_end(self, outputs):
        self.transcription_model.test_epoch_end(outputs, self)

    def predict_step(self, batch, batch_idx):
        # Call detection model directly (forward pass)
        detection_out = self.detection_model(batch)
        # Provide detection_out (or a relevant part of it) to transcription
        self.transcription_model.predict_step(batch, batch_idx, detection_out)

        return detection_out

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.transcription_model.parameters()) + list(self.detection_model.parameters()),
            **self.cfg.detection.model.optimizer,
        )

        if self.cfg.scheduler.type == "MultiStepLR":
            scheduler = {
                'scheduler': MultiStepLR(
                    optimizer,
                    milestones=list(self.cfg.scheduler.milestones),
                    gamma=self.cfg.scheduler.gamma
                ),
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self.cfg.scheduler.type == "LambdaLR":
            scheduler = {
                'scheduler': LambdaLR(optimizer, self.lr_lambda),
                'interval': 'step',
                'frequency': 1,
            }
        else:
            scheduler = None

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

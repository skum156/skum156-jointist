import torch
import torch.nn.functional as F
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
import pytorch_lightning as pl
from End2End.loss import HungarianMatcherv2, SetCriterion
from pytorch_lightning.utilities import rank_zero_only

from End2End.constants import SAMPLE_RATE
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from sklearn.metrics import (precision_recall_curve,
                             average_precision_score,
                             auc,
                             roc_auc_score,
                             precision_recall_fscore_support
                            )  
from End2End.models.instrument_detection.utils import obtain_segments, summarized_output

class Binary(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        lr_lambda,
        cfg
    ):
        super().__init__()
        self.network = network
        self.lr_lambda = lr_lambda
        self.plugin_labels_num = cfg.MIDI_MAPPING.plugin_labels_num
        self.IX_TO_NAME = cfg.MIDI_MAPPING.IX_TO_NAME
        self.instrument_type = cfg.MIDI_MAPPING.type
        self.cfg = cfg
        self.segment_samples = cfg.segment_seconds*SAMPLE_RATE
        self.segment_batch_size = cfg.seg_batch_size

    def calculate_loss(self, batch):
        loss_dict = {}
        output = self.network(batch['waveform'])
        if 'pred_logits' in output.keys():
            src_logits = output['pred_logits']
            pred = torch.sigmoid(src_logits)
        else:
            src_logits = None
            pred = output['pred']
        target = batch['instruments'][:,:-1] # remove empty class
       
        target_list = []
        for sample in target:
            target_list.append({'labels': sample.nonzero().flatten()})  
            
        loss_dict['loss_bce'] = F.binary_cross_entropy(pred.flatten(1), target)
        
        if 'aux_outputs' in output.keys():
            for idx, i in enumerate(output['aux_outputs']):
                pred = i['pred_logits'].flatten(1)
                pred = torch.sigmoid(pred)
                loss_dict[f'loss_bce_{idx}'] = F.binary_cross_entropy(pred, target)
        return loss_dict, src_logits, target_list, output, pred
        
    def training_step(self, batch, batch_idx, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self

        loss_dict, src_logits, target_classes, output, pred = self.calculate_loss(batch)
        loss = sum(loss_dict[k] for k in loss_dict if 'loss' in k)
        for key in loss_dict:
            logger.log(f"{key}/Train", loss_dict[key], on_step=False, on_epoch=True)
        logger.log('Detection_Loss/Train', loss, on_step=False, on_epoch=True)

        if (self.current_epoch+1)%(self.trainer.check_val_every_n_epoch)==0 or self.current_epoch==0:      
            if batch_idx==0:
                self.log_images(output['spec'].squeeze(1), f'Train/spectrogram', logger=logger)
                self._log_text(batch['instruments'], "Train/Labels", max_sentences=4, logger=logger)
                if isinstance(src_logits, torch.Tensor):
                    self._log_text(torch.sigmoid(src_logits)>0.5, "Train/Prediction", max_sentences=4, logger=logger)
                else:
                    self._log_text(pred>0.5, "Train/Prediction", max_sentences=4, logger=logger)
            if isinstance(src_logits, torch.Tensor):
                output_batch = {
                    'loss': loss,
                    'instruments': batch['instruments'].detach(),
                    'src_logits':  src_logits.detach(),
                    'target_classes': target_classes
                }
            else:
                output_batch = {
                    'loss': loss,
                    'instruments': batch['instruments'].detach(),
                    'src_logits':  None,
                    'target_classes': target_classes,
                    'sigmoid_output':  pred,                    
                }
            return output_batch
        return loss

    @rank_zero_only
    def training_epoch_end(self, outputs, jointist=None):
        if (self.current_epoch+1)%(self.trainer.check_val_every_n_epoch)==0 or self.current_epoch==0:
            if jointist:
                logger=jointist
            else:
                logger=self

            torch.save(outputs, 'trainset_outputs.pt')
            results = self.calculate_epoch_end(outputs)
            torch.save(results, 'trainset_results.pt')
            _ = self.barplot(results['pred_stat'], 'train_pred_counts (log_e)', (4,12), 0.2, log=True)
            _ = self.barplot(results['label_stat'], 'train_label_counts (log_e)', (4,12), 0.2, log=True)
            logger.logger.experiment.add_figure(f"Train/F1 scores",
                                                self.barplot(results['f1_stat'], 'F1', (4,12), 0.05, log=False),
                                                global_step=self.current_epoch)
            for key in results['metrics']:
                for instrument in results['metrics'][key]:
                    logger.log(f"{key}_Train/{instrument}", results['metrics'][key][instrument], on_step=False, on_epoch=True, rank_zero_only=True)
            for key in results['f1_dicts']:
                if key!='none':
                    logger.log(f"F1_average_Train/{key}", results['f1_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
                else:
                    for idx, f1_score in enumerate(results['f1_dicts'][key]):
                        instrument = self.IX_TO_NAME[idx]
                        logger.log(f"F1_Train/{instrument}", results['f1_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)
            for key in results['mAP_dicts']:
                if key!='none':
                    logger.log(f"mAP_average_Train/{key}", results['mAP_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
                else:
                    for idx, f1_score in enumerate(results['mAP_dicts'][key]):
                        instrument = self.IX_TO_NAME[idx]
                        logger.log(f"mAP_Train/{instrument}", results['mAP_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)

    def validation_step(self, batch, batch_idx, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self        
        metrics = {}     
        loss_dict, src_logits, target_classes, output, pred = self.calculate_loss(batch)
        loss = sum(loss_dict[k] for k in loss_dict if 'loss' in k)
        for key in loss_dict:
            logger.log(f"{key}/Valid", loss_dict[key], on_step=False, on_epoch=True)
        metrics['Detection_Loss/Valid']=loss
        if batch_idx==0:
            if self.current_epoch==0:
                self._log_text(batch['instruments'], "Valid/Labels", max_sentences=4, logger=logger)
                self.log_images(output['spec'].squeeze(1), f'Valid/spectrogram', logger=logger)
            if isinstance(src_logits, torch.Tensor):
                self._log_text(torch.sigmoid(src_logits)>0.5, "Valid/Prediction", max_sentences=4, logger=logger)
            else:
                self._log_text(pred>0.5, "Valid/Prediction", max_sentences=4, logger=logger)
        logger.log_dict(metrics)
        if isinstance(src_logits, torch.Tensor):
            output_batch = {
                'loss': loss,
                'instruments': batch['instruments'].detach(),
                'src_logits':  src_logits.detach(),
                'target_classes': target_classes
            }
        else:
            output_batch = {
                'loss': loss,
                'instruments': batch['instruments'].detach(),
                'src_logits':  None,
                'target_classes': target_classes,
                'sigmoid_output':  pred,                
            }
        return output_batch 

    @rank_zero_only
    def validation_epoch_end(self, outputs, jointist=None):
        if jointist:
            logger=jointist
        else:
            logger=self
        results = self.calculate_epoch_end(outputs)
        for key in results['metrics']:
            for instrument in results['metrics'][key]:
                logger.log(f"{key}_Valid/{instrument}", results['metrics'][key][instrument], on_step=False, on_epoch=True, rank_zero_only=True)
        for key in results['f1_dicts']:
            print(f"f1_dict key={key}")
            if key != 'none':
                logger.log(f"F1_average_Valid/{key}", results['f1_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
            else:
                for idx, f1_score in enumerate(results['f1_dicts'][key]):
                    instrument = self.IX_TO_NAME[idx]
                    logger.log(f"F1_Valid/{instrument}", results['f1_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)
        for key in results['mAP_dicts']:
            if key != 'none':
                logger.log(f"mAP_average_Valid/{key}", results['mAP_dicts'][key], on_step=False, on_epoch=True, rank_zero_only=True)
            else:
                for idx, f1_score in enumerate(results['mAP_dicts'][key]):
                    instrument = self.IX_TO_NAME[idx]
                    logger.log(f"mAP_Valid/{instrument}", results['mAP_dicts'][key][idx], on_step=False, on_epoch=True, rank_zero_only=True)

    # ----------------- Add these two methods at the end -----------------
    def forward(self, batch):
        # For inference/prediction: expects a dict with 'waveform' key
        return self.network(batch['waveform'])

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        # PyTorch Lightning will use this for .predict()
        return self.forward(batch)

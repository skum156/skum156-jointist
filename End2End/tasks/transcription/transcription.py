import torch
import torch.nn.functional as F
from torch import nn as nn, optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import End2End.inference_instruments_filter as PostProcessor
from End2End.constants import SAMPLE_RATE
from End2End.transcription_utils import (
    postprocess_probabilities_to_midi_events,
    predict_probabilities,
    write_midi_events_to_midi_file,
    predict_probabilities_baseline
)
from End2End.tasks.transcription.utils import (
    calculate_mean_std,
    calculate_intrumentwise_statistics,
    evaluate_F1,
    evaluate_flat_F1,
    piecewise_evaluation,
    get_flat_average,
    barplot
)
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def deep_tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        # Handle scalar tensors as well
        if obj.numel() == 1:
            return obj.item()
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: deep_tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_tensor_to_list(x) for x in obj]
    else:
        return obj

class Transcription(pl.LightningModule):
    def __init__(self, network: nn.Module, loss_function, lr_lambda, batch_data_preprocessor, cfg):
        super().__init__()
        self.network = network
        self.loss_function = loss_function
        self.learning_rate = cfg.lr
        self.lr_lambda = lr_lambda
        self.classes_num = cfg.transcription.model.args.classes_num
        self.plugin_labels_num = cfg.MIDI_MAPPING.plugin_labels_num
        self.IX_TO_NAME = cfg.MIDI_MAPPING.IX_TO_NAME
        self.instrument_type = cfg.MIDI_MAPPING.type
        self.cfg = cfg
        self.seg_batch_size = cfg.transcription.evaluation.seg_batch_size
        self.segment_samples = cfg.segment_seconds * SAMPLE_RATE
        if hasattr(cfg.datamodule, 'dataset_cfg'):
            self.test_segment_size = cfg.datamodule.dataset_cfg.test.segment_seconds
        elif hasattr(cfg.datamodule, 'dataloader_cfg'):
            self.test_segment_size = cfg.segment_seconds
        self.evaluation_output_path = os.path.join(os.getcwd(), 'MIDI_output')
        if cfg.datamodule.type == 'slakh':
            self.pkl_dir = cfg.datamodule.pkl_dir
        os.makedirs(self.evaluation_output_path, exist_ok=True)
        print(f"[DEBUG] MIDI output directory: {self.evaluation_output_path}")
        self.frame_threshold = cfg.transcription.postprocessor.args.frame_threshold
        self.onset_threshold = cfg.transcription.postprocessor.args.onset_threshold
        self.batch_data_preprocessor = batch_data_preprocessor
        self.post_processor = getattr(PostProcessor, cfg.transcription.postprocessor.type)(**cfg.transcription.postprocessor.args)

    def training_step(self, batch, batch_idx, jointist=None):
        logger = jointist if jointist else self
        if self.batch_data_preprocessor:
            batch = self.batch_data_preprocessor(batch)
        target_dict = batch['target_dict']
        outputs = self.network(batch['waveforms'], batch['conditions'])
        loss = self.loss_function(self.network, outputs, target_dict)
        logger.log('Transcription_Loss/Train', loss)
        output_dict = {'loss': loss, 'outputs': outputs}
        return output_dict

    def validation_step(self, batch, batch_idx, jointist=None):
        logger = jointist if jointist else self
        valid_metrics = {}
        if self.batch_data_preprocessor:
            batch = self.batch_data_preprocessor(batch)
        target_dict = batch['target_dict']
        outputs = self.network(batch['waveforms'], batch['conditions'])
        if batch_idx < 4:
            for key in target_dict:
                if self.current_epoch == 0:
                    # Assuming log_images handles tensors appropriately
                    # Make sure target_dict[key] is compatible with log_images
                    self.log_images(target_dict[key].squeeze(1), batch['conditions'], f'Valid/{key}', batch_idx, logger)
            for key in outputs:
                # Assuming log_images handles tensors appropriately
                # Make sure outputs[key] is compatible with log_images
                self.log_images(outputs[key].squeeze(1), batch['conditions'], f'Valid/{key}', batch_idx, logger)
        loss = self.loss_function(self.network, outputs, target_dict)
        valid_metrics['Transcription_Loss/Valid'] = loss
        logger.log_dict(valid_metrics)
        return loss, outputs

    def test_step(self, batch, batch_idx, plugin_ids=None, export=True, jointist=None):
        logger = jointist if jointist else self
        print(f"[DEBUG] test_step called with export={export}, batch_idx={batch_idx}")

        # Determine plugin IDs and conditions
        if isinstance(plugin_ids, torch.Tensor):
            if len(plugin_ids) == 0:
                plugin_ids = torch.arange(self.plugin_labels_num - 1)
                conditions = torch.eye(self.plugin_labels_num)[plugin_ids] # Select from identity matrix
            else:
                conditions = torch.eye(self.plugin_labels_num)[plugin_ids] # Select from identity matrix
        elif plugin_ids is None:
            if 'instruments' in batch:
                # Assuming batch['instruments'] is a one-hot encoded tensor or similar
                plugin_ids = torch.where(batch['instruments'][0] == 1)[0]
            else:
                print("[WARNING] 'instruments' not in batch — defaulting to all plugins")
                plugin_ids = torch.arange(self.plugin_labels_num - 1)
            # Create conditions from plugin_ids
            conditions = torch.zeros((len(plugin_ids), self.plugin_labels_num), device=plugin_ids.device)
            conditions.scatter_(1, plugin_ids.view(-1, 1), 1)
        else:
            raise ValueError(f"plugin_ids has an unknown type: {type(plugin_ids)}")

        audio = batch['waveform']
        trackname = batch.get('hdf5_name', [f"track_{batch_idx}"])[0]
        print(f"[DEBUG] Processing track: {trackname}, Plugin IDs: {plugin_ids.tolist()}")

        output_dict = {'reg_onset_output': [], 'frame_output': []}

        # Process each condition
        for condition_idx, condition in enumerate(conditions):
            current_plugin_id = plugin_ids[condition_idx].item()
            try:
                # Ensure condition is unsqueezed for batch dimension if needed by network
                # If network expects single condition, pass condition, else condition.unsqueeze(0)
                if self.test_segment_size is not None:
                    # Assuming network can handle a single condition tensor directly
                    _output_dict = self.network(batch['waveform'], condition.unsqueeze(0))
                else:
                    _output_dict = predict_probabilities(
                        self.network, audio.squeeze(0), condition, # Pass single condition
                        self.segment_samples, self.seg_batch_size
                    )
                print(f"[DEBUG] Processed condition idx: {current_plugin_id}, name: {self.IX_TO_NAME[current_plugin_id]}")
                for key in ['reg_onset_output', 'frame_output']:
                    if key in _output_dict:
                        output_dict[key].append(_output_dict[key])
                    else:
                        print(f"[WARNING] Key '{key}' not found in network output for plugin {self.IX_TO_NAME[current_plugin_id]}")
            except Exception as e:
                print(f"[ERROR] Exception processing plugin {self.IX_TO_NAME[current_plugin_id]} (ID: {current_plugin_id}): {e}")
                # Optionally, append a placeholder if an error occurs to maintain list length
                # This depends on how postprocess_probabilities_to_midi_events expects inputs
                # For now, just print the error and let subsequent steps fail if incomplete
                pass # Continue to the next plugin even if one fails

        # Convert lists to tensors
        for key in ['reg_onset_output', 'frame_output']:
            if isinstance(output_dict[key], list) and len(output_dict[key]) > 0:
                try:
                    # Concatenate along a new dimension (instrument dimension) if it doesn't exist
                    # Assuming outputs are (time_steps, features) and we want (num_plugins, time_steps, features)
                    # Or (batch, time_steps, features) -> (num_plugins * batch, time_steps, features)
                    # The current setup seems to assume each appended item is already (1, time_steps, features)
                    # and we want to stack them to (num_plugins, time_steps, features)
                    output_dict[key] = torch.cat(output_dict[key], dim=0) # Assuming dim=0 concatenates plugins
                    mean_val = output_dict[key].mean().item()
                    print(f"[DEBUG] {key} shape: {output_dict[key].shape}, mean: {mean_val:.6f}")
                    if mean_val < 1e-5:
                        print(f"[WARNING] {key} appears empty or near-zero — check model or input audio.")
                except Exception as e:
                    print(f"[ERROR] Failed to concatenate {key}: {e}. Skipping concatenation for this key.")
                    output_dict[key] = [] # Keep as list or empty tensor if concatenation fails

        # Check shapes before postprocessing
        # Ensure all required outputs are present and correctly shaped
        if 'frame_output' in output_dict and 'reg_onset_output' in output_dict:
            print(f"[DEBUG] Final frame_output shape: {output_dict['frame_output'].shape if isinstance(output_dict['frame_output'], torch.Tensor) else 'Not a Tensor'}")
            print(f"[DEBUG] Final reg_onset_output shape: {output_dict['reg_onset_output'].shape if isinstance(output_dict['reg_onset_output'], torch.Tensor) else 'Not a Tensor'}")
            if isinstance(output_dict['frame_output'], torch.Tensor) and output_dict['frame_output'].shape[0] != len(plugin_ids):
                print(f"[WARNING] Frame output first dim ({output_dict['frame_output'].shape[0]}) does not match plugin_ids count ({len(plugin_ids)}) — postprocessor may fail.")
        else:
            print("[WARNING] Missing 'frame_output' or 'reg_onset_output' in output_dict. MIDI export may fail.")


        if export:
            print(f"[DEBUG] Exporting MIDI files for track {trackname}...")
            try:
                # The post_processor is an instance of a class from inference_instruments_filter
                # Ensure postprocess_probabilities_to_midi_events can handle the output_dict format
                midi_events = postprocess_probabilities_to_midi_events(
                    output_dict, plugin_ids, self.IX_TO_NAME, self.classes_num, self.post_processor
                )

                # Convert all tensors inside midi_events to Python lists before writing
                # This is crucial to avoid the "Boolean value of Tensor with more than one value is ambiguous" error
                # in subsequent Pythonic checks or non-PyTorch functions.
                midi_events_converted = deep_tensor_to_list(midi_events)

                print(f"[DEBUG] midi_events keys and types after deep_tensor_to_list: {[(k, type(v)) for k, v in midi_events_converted.items()]}" if midi_events_converted else "[DEBUG] midi_events is None or empty")

                has_events = False
                if midi_events_converted: # Check if the dictionary itself is not empty
                    for k, v in midi_events_converted.items():
                        # Explicitly handle numpy arrays or lists that might contain events
                        if isinstance(v, (list, tuple)):
                            if len(v) > 0:
                                has_events = True
                                break
                        elif isinstance(v, np.ndarray):
                            # For numpy arrays, check if they are not empty
                            if v.size > 0:
                                has_events = True
                                break
                        # Handle scalar values that represent presence (e.g., non-zero)
                        # This covers numbers and single booleans that deep_tensor_to_list might produce
                        elif isinstance(v, (int, float, bool)):
                            if v: # For numbers, checks if non-zero; for bool, checks if True
                                has_events = True
                                break
                        # If after deep_tensor_to_list, you still find a torch.Tensor here,
                        # it means deep_tensor_to_list missed something or a tensor was re-introduced.
                        # This block is a safeguard.
                        elif isinstance(v, torch.Tensor):
                            if v.numel() == 1: # Single element tensor
                                if v.item(): # Get its Python scalar value
                                    has_events = True
                                    break
                            else: # Multi-element tensor (this should ideally be avoided here)
                                print(f"[WARNING] Multi-element torch.Tensor found for key '{k}' after deep_tensor_to_list. Shape: {v.shape}")
                                if v.any(): # Decide how to evaluate truthiness for multi-element tensors
                                    has_events = True
                                    break
                        # Catch other non-empty/non-None types if applicable
                        elif v is not None and v != {} and v != "": # Catches non-empty strings, other objects
                            has_events = True
                            break

                if has_events:
                    print(f"[DEBUG] MIDI events generated: { {k: (len(v) if hasattr(v, '__len__') else 'scalar') for k, v in midi_events_converted.items()} }")
                    if not os.access(self.evaluation_output_path, os.W_OK):
                        print(f"[ERROR] Cannot write to {self.evaluation_output_path} — check permissions or path.")
                    else:
                        midi_path = os.path.join(self.evaluation_output_path, f"{trackname}.mid")
                        # Pass the fully converted midi_events_converted to the writer
                        write_midi_events_to_midi_file(midi_events_converted, midi_path, self.instrument_type)
                        print(f"[DEBUG] MIDI file saved successfully at {midi_path}")
                else:
                    print(f"[WARNING] No MIDI events generated (empty predictions or post-processing issue?) for track {trackname}.")

            except Exception as e:
                print(f"[ERROR] Exception while exporting MIDI files for track {trackname}: {e}")
                # Log the full traceback for more detailed debugging if needed
                import traceback
                traceback.print_exc()

        return {}

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        # Pass kwargs through, including potentially 'export'
        return self.test_step(batch, batch_idx, export=kwargs.get('export', True), **kwargs)
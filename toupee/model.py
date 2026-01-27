#!/usr/bin/python
"""
Run a MLP experiment from a yaml file

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

import time
import copy
import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np # type: ignore
import uuid

import wandb
import toupee as tp
from toupee.model_builder import build_model_from_yaml

#TODO: backprop to the inputs
#TODO: sample weights
#TODO: early stopping
#TODO: checkpointing?


# Map Keras optimizer names to PyTorch optimizers
OPTIMIZER_MAP = {
    'Adam': optim.Adam,
    'adam': optim.Adam,
    'SGD': optim.SGD,
    'sgd': optim.SGD,
    'RMSprop': optim.RMSprop,
    'rmsprop': optim.RMSprop,
    'Adagrad': optim.Adagrad,
    'adagrad': optim.Adagrad,
    'Adadelta': optim.Adadelta,
    'adadelta': optim.Adadelta,
    'AdamW': optim.AdamW,
    'adamw': optim.AdamW,
}


def get_loss_function(loss_config: Any) -> nn.Module:
    """Convert Keras loss configuration to PyTorch loss function."""
    if isinstance(loss_config, str):
        loss_name = loss_config
    elif isinstance(loss_config, dict):
        loss_name = loss_config.get('class_name', 'categorical_crossentropy')
    else:
        loss_name = 'categorical_crossentropy'

    loss_map = {
        'categorical_crossentropy': nn.CrossEntropyLoss(),
        'CategoricalCrossentropy': nn.CrossEntropyLoss(),
        'binary_crossentropy': nn.BCEWithLogitsLoss(),
        'BinaryCrossentropy': nn.BCEWithLogitsLoss(),
        'mse': nn.MSELoss(),
        'mean_squared_error': nn.MSELoss(),
        'MeanSquaredError': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'mean_absolute_error': nn.L1Loss(),
        'MeanAbsoluteError': nn.L1Loss(),
        'kl_divergence': nn.KLDivLoss(reduction='batchmean'),
        'KLDivergence': nn.KLDivLoss(reduction='batchmean'),
    }

    if loss_name not in loss_map:
        logging.warning(f"Unknown loss function {loss_name}, defaulting to CrossEntropyLoss")
        return nn.CrossEntropyLoss()

    return loss_map[loss_name]


class OptimizerSchedule:
    """Schedules Optimizers and Learning Rates according to config."""

    def __init__(self, params: Dict, epochs: int, model_parameters):
        """
        Create an optimizer schedule from params.

        Args:
            params: Optimizer configuration (Keras format)
            epochs: Total number of training epochs
            model_parameters: PyTorch model parameters to optimize
        """
        self.params = copy.deepcopy(params)
        self.epochs = epochs
        # Store parameters as a list to allow multiple optimizer creation
        self.model_parameters = list(model_parameters)
        self.optimizers: Dict[int, optim.Optimizer] = {}

        # Force this to be an epoch schedule even if it's not
        if 'class_name' in self.params:
            self.params = {0: self.params}

        # Build optimizers for each epoch threshold
        for thresh, opt_params in self.params.items():
            self.optimizers[thresh] = self._build_optimizer(opt_params)

    def _build_optimizer(self, opt_params: Dict) -> optim.Optimizer:
        """Build a PyTorch optimizer from Keras-style config."""
        conf = copy.deepcopy(opt_params)
        class_name = conf.get('class_name', 'Adam')
        config = conf.get('config', {})

        # Get learning rate (handle dict schedules)
        lr = config.get('learning_rate', 0.001)
        if isinstance(lr, dict):
            lr = lr[min(lr.keys())]

        # Get other optimizer params
        opt_kwargs = {'lr': lr}

        # Map Keras params to PyTorch params
        if 'momentum' in config:
            opt_kwargs['momentum'] = config['momentum']
        if 'beta_1' in config:
            opt_kwargs['betas'] = (config['beta_1'], config.get('beta_2', 0.999))
        if 'epsilon' in config:
            opt_kwargs['eps'] = config['epsilon']
        if 'weight_decay' in config:
            opt_kwargs['weight_decay'] = config['weight_decay']
        if 'nesterov' in config:
            opt_kwargs['nesterov'] = config['nesterov']

        optimizer_class = OPTIMIZER_MAP.get(class_name, optim.Adam)
        return optimizer_class(self.model_parameters, **opt_kwargs)

    def _params_scheduler(self, epoch: int) -> Dict:
        """Get optimizer params for a given epoch."""
        for thresh in sorted(self.params.keys(), reverse=True):
            if epoch >= thresh:
                return self.params[thresh]
        return self.params[0]

    def _opt_scheduler(self, epoch: int) -> optim.Optimizer:
        """Get optimizer for a given epoch."""
        for thresh in sorted(self.optimizers.keys(), reverse=True):
            if epoch >= thresh:
                return self.optimizers[thresh]
        return self.optimizers[0]

    def __getitem__(self, epoch: int) -> optim.Optimizer:
        """Subscripting operator to get the right optimizer for any epoch."""
        return self._opt_scheduler(epoch)

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for a given epoch."""
        params = self._params_scheduler(epoch)
        lr = params['config']['learning_rate']
        if isinstance(lr, dict):
            for thresh, value in lr.items():
                if epoch >= thresh:
                    lr = value
        return lr

    def update_lr(self, optimizer: optim.Optimizer, epoch: int):
        """Update optimizer learning rate based on epoch schedule."""
        new_lr = self.get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class ReduceLROnPlateau:
    """Learning rate reducer on plateau, mimicking Keras callback."""

    def __init__(self, optimizer: optim.Optimizer, mode: str = 'min',
                 factor: float = 0.1, patience: int = 10,
                 min_lr: float = 0, min_delta: float = 0.0001,
                 verbose: int = 0, **kwargs):
        self.optimizer = optimizer
        self.verbose = verbose > 0
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=min_delta
        )

    def step(self, metric: float):
        """Step the scheduler with the current metric value."""
        self.scheduler.step(metric)


class Model:
    """Representation of a model using PyTorch."""

    def __init__(self, params, model_yaml: str = None, optimizer: Dict = None):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model YAML
        if not model_yaml:
            with open(params.model_file, 'r') as model_file:
                self.model_yaml = model_file.read()
        else:
            self.model_yaml = model_yaml

        # Build PyTorch model from YAML
        self._model = build_model_from_yaml(self.model_yaml)
        self._model.to(self.device)

        # Load weights if specified
        if params.model_weights:
            self._model.load_state_dict(torch.load(params.model_weights, map_location=self.device))

        # Set up optimizer and loss
        self.optimizer = optimizer or params.optimizer
        self._optimizer_schedule = OptimizerSchedule(
            self.optimizer,
            self.params.epochs,
            self._model.parameters()
        )
        self._loss = get_loss_function(params.loss)
        self._training_metrics = ['accuracy']

        # Image generator for standardization (set during fit)
        self.img_aug = None

    def inject_layers(self, additional_layers: List[Dict], predecessor: str) -> str:
        """Add layers after a specific layer. Returns name of last injected layer."""
        # Note: This is a complex operation that requires rebuilding the model
        # For PyTorch, we need to modify the model architecture
        logging.warning("inject_layers is not fully supported in PyTorch version")

        # Parse current model config and inject layers
        # This is a simplified version - full implementation would need
        # to handle the functional API model structure
        inject_uuid = uuid.uuid1()
        last_new_layer = f"autogenerated-{additional_layers[-1]['name']}-{inject_uuid}"

        # Rebuild model from modified YAML
        # TODO: Implement full layer injection for functional models
        return last_new_layer

    def copy_weights(self, other_model: 'Model', early_stop: bool = False):
        """Copy weights from another model."""
        self_state = self._model.state_dict()
        other_state = other_model._model.state_dict()

        for name, param in other_state.items():
            if name in self_state:
                if self_state[name].shape == param.shape:
                    self_state[name].copy_(param)
                elif early_stop:
                    break

        self._model.load_state_dict(self_state)

    def fit(self, data: 'tp.data.Dataset', epochs: int = None, verbose: int = None,
            log_wandb: bool = False, adversarial_testing: bool = False,
            tensorboard: bool = False):
        """Train the model."""
        start_time = time.perf_counter()
        epochs = epochs or self.params.epochs
        verbose = verbose if verbose is not None else self.params.verbose

        # Store image augmentation for standardization during inference
        self.img_aug = data.img_aug

        # Set up callbacks/schedulers
        reduce_lr_scheduler = None
        if self.params.reduce_lr_on_plateau:
            reduce_lr_scheduler = ReduceLROnPlateau(
                self._optimizer_schedule[0],
                **self.params.reduce_lr_on_plateau
            )

        # Set up TensorBoard
        tb_writer = None
        if tensorboard:
            tb_writer = SummaryWriter(log_dir=self.params.tb_log_dir)

        # Multi-GPU support
        if self.params.multi_gpu:
            logging.warning("!!! WARNING - EXPERIMENTAL !!! running on multi gpu")
            self._model = nn.DataParallel(self._model)

        # Training loop
        for epoch in range(epochs):
            self._model.train()
            optimizer = self._optimizer_schedule[epoch]

            # Update learning rate based on schedule
            self._optimizer_schedule.update_lr(optimizer, epoch)

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (batch_x, batch_y) in enumerate(data.get_training_handle()):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self._model(batch_x)

                # CrossEntropyLoss expects class indices, not one-hot
                if isinstance(self._loss, nn.CrossEntropyLoss):
                    targets = batch_y.argmax(dim=1)
                else:
                    targets = batch_y

                loss = self._loss(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y.argmax(dim=1)).sum().item()

            # Calculate epoch metrics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation
            val_loss, val_acc = self._evaluate_epoch(data.get_validation_handle(standardized=True))

            # Update learning rate on plateau
            if reduce_lr_scheduler:
                reduce_lr_scheduler.step(val_loss)

            # Logging
            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                    f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f} - "
                    f"lr: {current_lr:.6f}"
                )

            # TensorBoard logging
            if tb_writer:
                tb_writer.add_scalar('Loss/train', train_loss, epoch)
                tb_writer.add_scalar('Loss/val', val_loss, epoch)
                tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
                tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
                tb_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # wandb logging
            if log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })

        if tb_writer:
            tb_writer.close()

        end_time = time.perf_counter()
        logging.info('Model trained for %.2fm' % ((end_time - start_time) / 60.))

        # Evaluate on test set
        self.test_metrics = self.evaluate(
            data.get_testing_handle(),
            adversarial=adversarial_testing
        )

        if log_wandb:
            for metric, value in self.test_metrics.items():
                if isinstance(value, (int, float)):
                    wandb.run.summary[metric] = value

    def _evaluate_epoch(self, data_loader) -> tuple:
        """Evaluate model for one epoch, return loss and accuracy."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self._model(batch_x)

                if isinstance(self._loss, nn.CrossEntropyLoss):
                    targets = batch_y.argmax(dim=1)
                else:
                    targets = batch_y

                loss = self._loss(outputs, targets)

                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y.argmax(dim=1)).sum().item()

        return total_loss / total, correct / total

    def evaluate(self, test_data, adversarial: bool = False) -> Dict:
        """Evaluate model on some test data handle."""
        return tp.metrics.evaluate(
            self,
            test_data,
            adversarial_gradient_source=self if adversarial else None
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Output probabilities for input data."""
        self._model.eval()

        # Standardize if we have image augmentation
        if self.img_aug:
            X = self.img_aug.standardize(X.copy())

        # Convert to tensor
        X = torch.from_numpy(X.astype(np.float32))

        # Handle image data: NHWC -> NCHW
        if X.dim() == 4:
            X = X.permute(0, 3, 1, 2)

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self._model(X)
            # Apply softmax if model doesn't have it
            probs = torch.softmax(outputs, dim=-1)

        return probs.cpu().numpy()

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Output class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, filename: str):
        """Save the model."""
        torch.save(self._model.state_dict(), filename)

    def get_pytorch_model(self) -> nn.Module:
        """Return raw PyTorch model."""
        return self._model

    # Alias for backward compatibility
    def get_keras_model(self) -> nn.Module:
        """Return raw model (PyTorch). Kept for API compatibility."""
        logging.warning("get_keras_model() is deprecated, use get_pytorch_model()")
        return self._model

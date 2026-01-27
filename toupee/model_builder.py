#!/usr/bin/python
"""
YAML Model Builder - Converts Keras YAML model definitions to PyTorch

Alan Mosca
Department of Computer Science and Information Systems
Birkbeck, University of London

All code released under Apachev2.0 licensing.
"""
__docformat__ = 'restructedtext en'

from typing import Dict, List, Any, Optional, Tuple, Callable
import yaml
import torch


class KerasYamlLoader(yaml.FullLoader):
    """Custom YAML loader that handles Keras/TensorFlow-specific tags."""
    pass


def _construct_python_tuple(loader, node):
    """Construct a Python tuple from YAML."""
    return tuple(loader.construct_sequence(node))


def _construct_tf_list_wrapper(loader, node):
    """Convert TensorFlow ListWrapper to plain Python list."""
    # The node contains a sequence with a single element (the actual list)
    value = loader.construct_sequence(node)
    if len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


# Register constructors for Keras/TensorFlow YAML tags
KerasYamlLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    _construct_python_tuple
)
KerasYamlLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:tensorflow.python.training.tracking.data_structures.ListWrapper',
    _construct_tf_list_wrapper
)
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def get_activation(activation: str) -> nn.Module:
    """Convert Keras activation name to PyTorch module."""
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=-1),
        'linear': nn.Identity(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'softplus': nn.Softplus(),
        'softsign': nn.Softsign(),
        'leaky_relu': nn.LeakyReLU(),
    }
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    return activations[activation]


def parse_tuple(value: Any) -> Tuple:
    """Convert YAML tuple representation to Python tuple."""
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(parse_tuple(v) if isinstance(v, (list, tuple)) else v for v in value)
    return value


def get_padding_from_keras(padding: str, kernel_size: Tuple[int, int]) -> Any:
    """Convert Keras padding to PyTorch padding."""
    if padding == 'same':
        return 'same'
    elif padding == 'valid':
        return 0
    else:
        return padding


class ChannelPermute(nn.Module):
    """Permute channels from NHWC to NCHW or vice versa."""
    def __init__(self, to_channels_first: bool = True):
        super().__init__()
        self.to_channels_first = to_channels_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.to_channels_first:
            # NHWC -> NCHW
            return x.permute(0, 3, 1, 2)
        else:
            # NCHW -> NHWC
            return x.permute(0, 2, 3, 1)


class YAMLModelBuilder:
    """
    Builds PyTorch models from Keras YAML definitions.

    Supports both Sequential and Functional API model definitions.
    """

    def __init__(self):
        self.layer_builders: Dict[str, Callable] = {
            'InputLayer': self._build_input_layer,
            'Conv2D': self._build_conv2d,
            'Dense': self._build_dense,
            'BatchNormalization': self._build_batchnorm,
            'MaxPooling2D': self._build_maxpool2d,
            'AveragePooling2D': self._build_avgpool2d,
            'GlobalAveragePooling2D': self._build_global_avgpool2d,
            'Dropout': self._build_dropout,
            'Flatten': self._build_flatten,
            'Activation': self._build_activation,
            'Softmax': self._build_softmax,
            'ZeroPadding2D': self._build_zeropadding2d,
            'Concatenate': self._build_concatenate,
            'Add': self._build_add,
        }
        self._input_shape: Optional[Tuple] = None
        self._current_shape: Optional[List[int]] = None
        self._data_format: str = 'channels_last'
        self._layer_outputs: Dict[str, Any] = {}

    def build(self, yaml_content: str) -> nn.Module:
        """Build a PyTorch model from Keras YAML string."""
        config = yaml.load(yaml_content, Loader=KerasYamlLoader)

        class_name = config.get('class_name', 'Sequential')

        if class_name == 'Sequential':
            return self._build_sequential(config)
        elif class_name == 'Model':
            return self._build_functional(config)
        else:
            raise ValueError(f"Unknown model class: {class_name}")

    def _build_sequential(self, config: Dict) -> nn.Module:
        """Build a Sequential model."""
        layers_config = config.get('config', [])
        layers = []

        for i, layer_config in enumerate(layers_config):
            layer_class = layer_config['class_name']
            layer_params = layer_config.get('config', {})

            # Track input shape from first layer
            if i == 0 and 'batch_input_shape' in layer_params:
                self._input_shape = parse_tuple(layer_params['batch_input_shape'])
                self._data_format = layer_params.get('data_format', 'channels_last')
                # Initialize current shape (exclude batch dim)
                self._current_shape = list(self._input_shape[1:])

            if layer_class in self.layer_builders:
                layer = self.layer_builders[layer_class](layer_params)
                if layer is not None:
                    if isinstance(layer, list):
                        layers.extend(layer)
                    else:
                        layers.append(layer)
            else:
                raise ValueError(f"Unsupported layer type: {layer_class}")

        return nn.Sequential(*layers)

    def _build_functional(self, config: Dict) -> nn.Module:
        """Build a Functional API model."""
        model_config = config.get('config', {})
        layers_config = model_config.get('layers', [])

        # Build layer map for functional model
        return FunctionalModel(layers_config, self)

    def _get_in_channels(self) -> int:
        """Get current number of input channels."""
        if self._current_shape is None:
            raise ValueError("Input shape not set")
        if self._data_format == 'channels_first':
            return self._current_shape[0]
        else:
            return self._current_shape[-1]

    def _get_spatial_dims(self) -> Tuple[int, int]:
        """Get current spatial dimensions (H, W)."""
        if self._current_shape is None:
            raise ValueError("Input shape not set")
        if self._data_format == 'channels_first':
            return tuple(self._current_shape[1:3])
        else:
            return tuple(self._current_shape[0:2])

    def _update_shape_conv(self, out_channels: int, kernel_size: Tuple, stride: Tuple, padding: str):
        """Update tracked shape after convolution."""
        if self._current_shape is None:
            return
        h, w = self._get_spatial_dims()
        kh, kw = kernel_size
        sh, sw = stride

        if padding == 'same':
            new_h, new_w = (h + sh - 1) // sh, (w + sw - 1) // sw
        else:  # valid
            new_h = (h - kh) // sh + 1
            new_w = (w - kw) // sw + 1

        if self._data_format == 'channels_first':
            self._current_shape = [out_channels, new_h, new_w]
        else:
            self._current_shape = [new_h, new_w, out_channels]

    def _update_shape_pool(self, pool_size: Tuple, stride: Tuple, padding: str = 'valid'):
        """Update tracked shape after pooling."""
        if self._current_shape is None:
            return
        h, w = self._get_spatial_dims()
        ph, pw = pool_size
        sh, sw = stride if stride else pool_size

        if padding == 'same':
            new_h, new_w = (h + sh - 1) // sh, (w + sw - 1) // sw
        else:  # valid
            new_h = (h - ph) // sh + 1
            new_w = (w - pw) // sw + 1

        channels = self._get_in_channels()
        if self._data_format == 'channels_first':
            self._current_shape = [channels, new_h, new_w]
        else:
            self._current_shape = [new_h, new_w, channels]

    def _build_input_layer(self, config: Dict) -> Optional[nn.Module]:
        """Input layer - just tracks shape, no actual layer needed."""
        if 'batch_input_shape' in config:
            self._input_shape = parse_tuple(config['batch_input_shape'])
            self._data_format = config.get('data_format', 'channels_last')
            self._current_shape = list(self._input_shape[1:])
        return None

    def _build_conv2d(self, config: Dict) -> nn.Module:
        """Build Conv2D layer."""
        in_channels = self._get_in_channels()
        out_channels = config['filters']
        kernel_size = parse_tuple(config['kernel_size'])
        stride = parse_tuple(config.get('strides', (1, 1)))
        padding = config.get('padding', 'valid')
        use_bias = config.get('use_bias', True)

        # Convert Keras padding to PyTorch
        if padding == 'same':
            # PyTorch 'same' padding requires stride=1, otherwise compute manually
            if stride == (1, 1):
                pad = 'same'
            else:
                # For non-unit strides, we need to add ZeroPadding manually
                kh, kw = kernel_size
                pad_h = kh // 2
                pad_w = kw // 2
                pad = (pad_h, pad_w)
        else:
            pad = 0

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=use_bias
        )

        self._update_shape_conv(out_channels, kernel_size, stride, padding)
        return conv

    def _build_dense(self, config: Dict) -> nn.Module:
        """Build Dense (Linear) layer."""
        out_features = config['units']
        use_bias = config.get('use_bias', True)

        # Calculate in_features from current shape
        if self._current_shape is not None:
            import functools
            import operator
            in_features = functools.reduce(operator.mul, self._current_shape, 1)
        else:
            raise ValueError("Cannot determine input features for Dense layer")

        linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias
        )

        self._current_shape = [out_features]
        return linear

    def _build_batchnorm(self, config: Dict) -> nn.Module:
        """Build BatchNormalization layer."""
        momentum = config.get('momentum', 0.99)
        epsilon = config.get('epsilon', 0.001)

        # Determine if 1D or 2D based on current shape
        if self._current_shape is not None and len(self._current_shape) == 1:
            # 1D batchnorm
            num_features = self._current_shape[0]
            return nn.BatchNorm1d(
                num_features=num_features,
                eps=epsilon,
                momentum=1 - momentum  # PyTorch momentum is 1 - Keras momentum
            )
        else:
            # 2D batchnorm
            num_features = self._get_in_channels()
            return nn.BatchNorm2d(
                num_features=num_features,
                eps=epsilon,
                momentum=1 - momentum
            )

    def _build_maxpool2d(self, config: Dict) -> nn.Module:
        """Build MaxPooling2D layer."""
        pool_size = parse_tuple(config['pool_size'])
        stride = parse_tuple(config.get('strides', pool_size))
        padding = config.get('padding', 'valid')

        pad = 0 if padding == 'valid' else pool_size[0] // 2

        pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=stride,
            padding=pad
        )

        self._update_shape_pool(pool_size, stride, padding)
        return pool

    def _build_avgpool2d(self, config: Dict) -> nn.Module:
        """Build AveragePooling2D layer."""
        pool_size = parse_tuple(config['pool_size'])
        stride = parse_tuple(config.get('strides', pool_size))
        padding = config.get('padding', 'valid')

        pad = 0 if padding == 'valid' else pool_size[0] // 2

        pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=stride,
            padding=pad
        )

        self._update_shape_pool(pool_size, stride, padding)
        return pool

    def _build_global_avgpool2d(self, config: Dict) -> nn.Module:
        """Build GlobalAveragePooling2D layer."""
        channels = self._get_in_channels()
        self._current_shape = [channels]
        return nn.AdaptiveAvgPool2d((1, 1))

    def _build_dropout(self, config: Dict) -> nn.Module:
        """Build Dropout layer."""
        rate = config.get('rate', 0.5)
        return nn.Dropout(p=rate)

    def _build_flatten(self, config: Dict) -> nn.Module:
        """Build Flatten layer."""
        if self._current_shape is not None:
            import functools
            import operator
            flat_size = functools.reduce(operator.mul, self._current_shape, 1)
            self._current_shape = [flat_size]
        return nn.Flatten()

    def _build_activation(self, config: Dict) -> nn.Module:
        """Build Activation layer."""
        activation = config.get('activation', 'relu')
        return get_activation(activation)

    def _build_softmax(self, config: Dict) -> nn.Module:
        """Build Softmax layer."""
        axis = config.get('axis', -1)
        return nn.Softmax(dim=axis)

    def _build_zeropadding2d(self, config: Dict) -> nn.Module:
        """Build ZeroPadding2D layer."""
        padding = parse_tuple(config['padding'])

        # Handle nested tuples ((top, bottom), (left, right))
        if isinstance(padding[0], tuple):
            top, bottom = padding[0]
            left, right = padding[1]
            pad = (left, right, top, bottom)  # PyTorch order: left, right, top, bottom
        else:
            # Single tuple interpreted as symmetric padding
            pad = (padding[1], padding[1], padding[0], padding[0])

        # Update shape
        if self._current_shape is not None:
            h, w = self._get_spatial_dims()
            new_h = h + pad[2] + pad[3]  # top + bottom
            new_w = w + pad[0] + pad[1]  # left + right
            channels = self._get_in_channels()
            if self._data_format == 'channels_first':
                self._current_shape = [channels, new_h, new_w]
            else:
                self._current_shape = [new_h, new_w, channels]

        return nn.ZeroPad2d(pad)

    def _build_concatenate(self, config: Dict) -> nn.Module:
        """Build Concatenate layer - placeholder for functional models."""
        return None

    def _build_add(self, config: Dict) -> nn.Module:
        """Build Add layer - placeholder for functional models."""
        return None


class FunctionalModel(nn.Module):
    """
    PyTorch model built from Keras Functional API definition.

    Handles complex architectures with multiple inputs, skip connections, etc.
    """

    def __init__(self, layers_config: List[Dict], builder: YAMLModelBuilder):
        super().__init__()
        self.layers_config = layers_config
        self.builder = builder
        self._layer_modules: Dict[str, nn.Module] = {}
        self._layer_configs: Dict[str, Dict] = {}
        self._input_names: List[str] = []
        self._output_name: Optional[str] = None

        self._build_layers()

    def _build_layers(self):
        """Build all layers from config."""
        for layer_config in self.layers_config:
            name = layer_config['name']
            class_name = layer_config['class_name']
            config = layer_config.get('config', {})
            inbound = layer_config.get('inbound_nodes', [])

            self._layer_configs[name] = {
                'class_name': class_name,
                'config': config,
                'inbound_nodes': inbound
            }

            if class_name == 'InputLayer':
                self._input_names.append(name)
                if 'batch_input_shape' in config:
                    self.builder._input_shape = parse_tuple(config['batch_input_shape'])
                    self.builder._data_format = config.get('data_format', 'channels_last')
                    self.builder._current_shape = list(self.builder._input_shape[1:])
                continue

            # Track shapes through the network
            if inbound and len(inbound) > 0 and len(inbound[0]) > 0:
                # Get shape from previous layer
                prev_name = inbound[0][0][0]
                if prev_name in self._layer_configs:
                    prev_config = self._layer_configs[prev_name]
                    # Shape tracking is handled by builder

            if class_name in self.builder.layer_builders:
                layer = self.builder.layer_builders[class_name](config)
                if layer is not None:
                    self._layer_modules[name] = layer
                    self.add_module(name.replace('-', '_').replace('.', '_'), layer)

            self._output_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the functional model."""
        outputs = {}

        # Set initial input
        if self._input_names:
            outputs[self._input_names[0]] = x

        for layer_config in self.layers_config:
            name = layer_config['name']
            class_name = layer_config['class_name']
            inbound = layer_config.get('inbound_nodes', [])

            if class_name == 'InputLayer':
                continue

            # Get input tensors
            if inbound and len(inbound) > 0 and len(inbound[0]) > 0:
                if class_name in ('Concatenate', 'Add'):
                    # Multiple inputs
                    inputs = [outputs[node[0]] for node in inbound[0]]
                    if class_name == 'Concatenate':
                        out = torch.cat(inputs, dim=1)
                    else:  # Add
                        out = sum(inputs)
                else:
                    # Single input
                    prev_name = inbound[0][0][0]
                    layer_input = outputs[prev_name]

                    if name in self._layer_modules:
                        out = self._layer_modules[name](layer_input)
                    else:
                        out = layer_input
            else:
                out = x

            outputs[name] = out

        return outputs[self._output_name]


def build_model_from_yaml(yaml_content: str) -> nn.Module:
    """Convenience function to build a model from YAML string."""
    builder = YAMLModelBuilder()
    return builder.build(yaml_content)

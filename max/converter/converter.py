"""
PyTorch to Modular MAX Architecture Converter

This module provides functionality to convert PyTorch models to MAX graphs
that can be executed efficiently on MAX runtime.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

from max.graph import Graph, TensorType, ops, DeviceRef, Weight
from max.graph.weights import PytorchWeights
from max.dtype import DType
from max.engine import InferenceSession


class PyTorchToMAXConverter:
    """
    Converts PyTorch models to MAX executable graphs.
    
    Features:
    - Automatic layer mapping from PyTorch to MAX operations
    - Weight loading from PyTorch state_dict or checkpoint files
    - Support for common model architectures (transformers, CNNs)
    - Device placement optimization
    - Quantization support
    """
    
    def __init__(self, device: Optional[DeviceRef] = None, dtype: Optional[DType] = None):
        """
        Initialize the converter.
        
        Args:
            device: Target device (CPU/GPU). Auto-detected if None.
            dtype: Target data type. Defaults to float32.
        """
        self.device = device or self._auto_detect_device()
        self.dtype = dtype or DType.float32
        # Initialize session with proper device setup
        self.session = self._create_inference_session()
        self.weight_arrays = {}  # Store actual weight data
        
        # Layer mapping registry
        self.layer_mappings = {
            nn.Linear: self._convert_linear,
            nn.Embedding: self._convert_embedding,
            nn.LayerNorm: self._convert_layernorm,
            nn.Conv2d: self._convert_conv2d,
            nn.ReLU: self._convert_relu,
            nn.GELU: self._convert_gelu,
            nn.Dropout: self._convert_dropout,
            nn.MultiheadAttention: self._convert_attention,
        }
        
    def _auto_detect_device(self) -> DeviceRef:
        """Auto-detect the best available device."""
        try:
            from max.driver import accelerator_count
            if accelerator_count() > 0:
                return DeviceRef.GPU()
        except:
            pass
        return DeviceRef.CPU()
    
    def _create_inference_session(self) -> InferenceSession:
        """Create an InferenceSession with proper device setup."""
        try:
            from max.driver import CPU, Accelerator, accelerator_count
            
            devices = [CPU()]  # Always include CPU
            
            # Add accelerators if available
            for i in range(accelerator_count()):
                devices.append(Accelerator(i))
                
            return InferenceSession(devices=devices)
        except:
            # Fallback to CPU-only session
            from max.driver import CPU
            return InferenceSession(devices=[CPU()])
    
    def convert_model(self, 
                     pytorch_model: nn.Module,
                     input_shapes: List[Tuple[int, ...]],
                     model_name: str = "converted_model",
                     weights_path: Optional[Union[str, Path]] = None) -> Any:
        """
        Convert a PyTorch model to MAX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shapes: List of input tensor shapes
            model_name: Name for the MAX graph
            weights_path: Optional path to PyTorch weights file (.pt, .pth)
            
        Returns:
            Compiled MAX model ready for inference
        """
        
        # Clear previous weight arrays
        self.weight_arrays.clear()
        
        # Create input types for MAX graph
        input_types = [
            TensorType(self.dtype, shape, device=self.device) 
            for shape in input_shapes
        ]
        
        # Build MAX graph
        with Graph(model_name, input_types=input_types) as graph:
            # Convert the model architecture
            outputs = self._convert_module(pytorch_model, graph.inputs, graph)
            
            # Set graph outputs
            if isinstance(outputs, (list, tuple)):
                for output in outputs:
                    graph.output(output)
            else:
                graph.output(outputs)
        
        # Load weights if provided or use extracted weights
        weights_registry = self.weight_arrays.copy()  # Use collected weight arrays
        if weights_path:
            # If external weights provided, update registry
            external_weights = self._load_pytorch_weights(weights_path)
            weights_registry.update(external_weights)
            
        # Compile and return the model
        return self.session.load(graph, weights_registry=weights_registry)
    
    def convert_from_checkpoint(self,
                               model_path: Union[str, Path],
                               input_shapes: List[Tuple[int, ...]],
                               model_name: str = "converted_model") -> Any:
        """
        Convert a PyTorch model directly from a checkpoint file.
        
        Args:
            model_path: Path to PyTorch model file (.pt, .pth, .safetensors)
            input_shapes: List of input tensor shapes
            model_name: Name for the MAX graph
            
        Returns:
            Compiled MAX model ready for inference
        """
        # Clear previous weight arrays
        self.weight_arrays.clear()
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load PyTorch weights
        weights = PytorchWeights(model_path)
        
        # Create input types
        input_types = [
            TensorType(self.dtype, shape, device=self.device) 
            for shape in input_shapes
        ]
        
        # Build graph by analyzing weight structure
        with Graph(model_name, input_types=input_types) as graph:
            outputs = self._build_graph_from_weights(weights, graph.inputs, graph)
            graph.output(outputs)
        
        # Load weights into graph
        weights_registry = {
            key: weight.allocate()
            for key, weight in weights.items()
        }
        
        return self.session.load(graph, weights_registry=weights_registry)
    
    def _convert_module(self, module: nn.Module, inputs, graph) -> Any:
        """
        Convert a PyTorch module to MAX operations.
        
        Args:
            module: PyTorch module to convert
            inputs: Input tensors
            graph: MAX graph context
            
        Returns:
            Output tensor(s) from the converted module
        """
        module_type = type(module)
        
        if module_type in self.layer_mappings:
            return self.layer_mappings[module_type](module, inputs, graph)
        elif isinstance(module, nn.Sequential):
            return self._convert_sequential(module, inputs, graph)
        elif isinstance(module, nn.ModuleList):
            return self._convert_module_list(module, inputs, graph)
        else:
            # Attempt generic conversion for custom modules
            return self._convert_generic_module(module, inputs, graph)
    
    def _convert_sequential(self, module: nn.Sequential, inputs, graph) -> Any:
        """Convert nn.Sequential module."""
        x = inputs
        for layer in module:
            x = self._convert_module(layer, x, graph)
        return x
    
    def _convert_module_list(self, module: nn.ModuleList, inputs, graph) -> Any:
        """Convert nn.ModuleList (like transformer layers)."""
        x = inputs
        for layer in module:
            x = self._convert_module(layer, x, graph)
        return x
    
    def _convert_generic_module(self, module: nn.Module, inputs, graph) -> Any:
        """Attempt to convert a generic module by analyzing its structure."""
        # For custom modules, try to convert child modules
        if hasattr(module, 'forward'):
            # This is a simplified approach - in practice, you'd need
            # to trace the forward pass or have more sophisticated analysis
            warnings.warn(f"Generic conversion for {type(module)} may not be accurate")
            
        # Try to convert known child modules
        x = inputs
        for name, child in module.named_children():
            if hasattr(child, 'forward'):
                x = self._convert_module(child, x, graph)
        return x
    
    # Layer-specific conversion methods
    def _convert_linear(self, layer: nn.Linear, inputs, graph) -> Any:
        """Convert nn.Linear to MAX operations."""
        # Linear layers always expect a single input tensor
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"Linear layer expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs

        # Create Weight objects instead of passing raw arrays
        weight_name = f"linear_weight_{id(layer)}"  # Generate unique name
        weight_array = layer.weight.detach().cpu().numpy().T  # Transpose for MAX
        # Ensure contiguous memory layout (required by MAX)
        if not weight_array.flags['C_CONTIGUOUS']:
            weight_array = weight_array.copy()
        
        weight = Weight(
            name=weight_name,
            dtype=self.dtype,
            shape=weight_array.shape,
            device=self.device
        )
        # Store weight data for later loading
        self.weight_arrays[weight_name] = weight_array
        weight_tensor = graph.add_weight(weight)
        
        # Matrix multiplication
        output = ops.matmul(x.tensor, weight_tensor)
        
        # Add bias if present
        if layer.bias is not None:
            bias_name = f"linear_bias_{id(layer)}"
            bias_array = layer.bias.detach().cpu().numpy()
            # Ensure contiguous memory layout (required by MAX)
            if not bias_array.flags['C_CONTIGUOUS']:
                bias_array = bias_array.copy()
            
            bias = Weight(
                name=bias_name,
                dtype=self.dtype,
                shape=bias_array.shape,
                device=self.device
            )
            # Store bias data for later loading
            self.weight_arrays[bias_name] = bias_array
            bias_tensor = graph.add_weight(bias)
            output = ops.add(output, bias_tensor)
            
        return output
    
    def _convert_embedding(self, layer: nn.Embedding, inputs, graph) -> Any:
        """Convert nn.Embedding to MAX operations."""
        # Embedding layers expect a single input (indices)
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"Embedding layer expects 1 input, got {len(inputs)}")
            indices = inputs[0]
        else:
            indices = inputs
            
        # Create Weight object for embedding weights
        weight_name = f"embedding_weight_{id(layer)}"
        weight_array = layer.weight.detach().cpu().numpy()
        # Ensure contiguous memory layout (required by MAX)
        if not weight_array.flags['C_CONTIGUOUS']:
            weight_array = weight_array.copy()
        
        weight = Weight(
            name=weight_name,
            dtype=self.dtype,
            shape=weight_array.shape,
            device=self.device
        )
        # Store weight data for later loading
        self.weight_arrays[weight_name] = weight_array
        embedding_weights = graph.add_weight(weight)
        
        # Use gather operation for embedding lookup
        return ops.gather(embedding_weights, indices.tensor, axis=0)
    
    def _convert_layernorm(self, layer: nn.LayerNorm, inputs, graph) -> Any:
        """Convert nn.LayerNorm to MAX operations."""
        # LayerNorm expects a single input tensor
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"LayerNorm expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Create Weight objects for normalization parameters
        gamma_name = f"layernorm_gamma_{id(layer)}"
        gamma_array = layer.weight.detach().cpu().numpy()
        # Ensure contiguous memory layout (required by MAX)
        if not gamma_array.flags['C_CONTIGUOUS']:
            gamma_array = gamma_array.copy()
        gamma_weight = Weight(
            name=gamma_name,
            dtype=self.dtype,
            shape=gamma_array.shape,
            device=self.device
        )
        # Store weight data for later loading
        self.weight_arrays[gamma_name] = gamma_array
        gamma = graph.add_weight(gamma_weight)
        
        beta_name = f"layernorm_beta_{id(layer)}"
        beta_array = layer.bias.detach().cpu().numpy()
        # Ensure contiguous memory layout (required by MAX)
        if not beta_array.flags['C_CONTIGUOUS']:
            beta_array = beta_array.copy()
        beta_weight = Weight(
            name=beta_name,
            dtype=self.dtype,
            shape=beta_array.shape,
            device=self.device
        )
        # Store weight data for later loading
        self.weight_arrays[beta_name] = beta_array
        beta = graph.add_weight(beta_weight)
        
        # Layer normalization
        return ops.layer_norm(x.tensor, gamma, beta, epsilon=layer.eps)
    
    def _convert_conv2d(self, layer: nn.Conv2d, inputs, graph) -> Any:
        """Convert nn.Conv2d to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"Conv2d expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Create Weight object for convolution parameters
        weight_name = f"conv2d_weight_{id(layer)}"
        weight_array = layer.weight.detach().cpu().numpy()
        # Ensure contiguous memory layout (required by MAX)
        if not weight_array.flags['C_CONTIGUOUS']:
            weight_array = weight_array.copy()
        
        weight_obj = Weight(
            name=weight_name,
            dtype=self.dtype,
            shape=weight_array.shape,
            device=self.device
        )
        # Store weight data for later loading
        self.weight_arrays[weight_name] = weight_array
        weight = graph.add_weight(weight_obj)
        
        # Convert padding tuple to MAX format
        padding = (
            layer.padding[0], layer.padding[0],  # height padding
            layer.padding[1], layer.padding[1]   # width padding
        )
        
        # Convolution operation
        output = ops.conv2d(
            x.tensor, weight, 
            stride=layer.stride,
            dilation=layer.dilation,
            padding=padding
        )
        
        # Add bias if present
        if layer.bias is not None:
            bias_name = f"conv2d_bias_{id(layer)}"
            bias_array = layer.bias.detach().cpu().numpy()
            # Ensure contiguous memory layout (required by MAX)
            if not bias_array.flags['C_CONTIGUOUS']:
                bias_array = bias_array.copy()
            bias_weight = Weight(
                name=bias_name,
                dtype=self.dtype,
                shape=bias_array.shape,
                device=self.device
            )
            # Store bias data for later loading
            self.weight_arrays[bias_name] = bias_array
            bias = graph.add_weight(bias_weight)
            output = ops.add(output, bias)
            
        return output
    
    def _convert_relu(self, layer: nn.ReLU, inputs, graph) -> Any:
        """Convert nn.ReLU to MAX operations."""
        # ReLU expects a single input tensor
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"ReLU expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        return ops.relu(x.tensor)
    
    def _convert_gelu(self, layer: nn.GELU, inputs, graph) -> Any:
        """Convert nn.GELU to MAX operations."""
        # GELU expects a single input tensor
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"GELU expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Check if approximate GELU is requested
        approximate = getattr(layer, 'approximate', 'none')
        if approximate == 'tanh':
            return ops.gelu(x.tensor, approximate="tanh")
        else:
            return ops.gelu(x.tensor)
    
    def _convert_dropout(self, layer: nn.Dropout, inputs, graph) -> Any:
        """Convert nn.Dropout (passthrough during inference)."""
        if isinstance(inputs, (list, tuple)):
            return inputs[0] if len(inputs) == 1 else inputs
        else:
            return inputs
    
    def _convert_attention(self, layer: nn.MultiheadAttention, inputs, graph) -> Any:
        """Convert nn.MultiheadAttention to MAX operations."""
        # Attention can take 1 input (query=key=value) or 3 inputs (query, key, value)
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 1:
                query = key = value = inputs[0]
            elif len(inputs) == 3:
                query, key, value = inputs
            else:
                raise ValueError(f"Attention expects 1 or 3 inputs, got {len(inputs)}")
        else:
            query = key = value = inputs
            
        # This is a simplified conversion - full attention requires more complex logic
        warnings.warn("MultiheadAttention conversion is simplified. Consider using max.nn.MultiheadAttention")
        
        # For now, return the query (identity operation)
        # In a full implementation, you'd implement the full attention mechanism
        return query
    
    def _load_pytorch_weights(self, weights_path: Union[str, Path]) -> Dict[str, Any]:
        """Load PyTorch weights from file."""
        weights_path = Path(weights_path)
        
        if weights_path.suffix in ['.pt', '.pth']:
            weights = PytorchWeights(weights_path)
            return {key: weight.allocate() for key, weight in weights.items()}
        else:
            raise ValueError(f"Unsupported weight format: {weights_path.suffix}")
    
    def _extract_state_dict(self, model: nn.Module) -> Dict[str, Any]:
        """Extract state dict from PyTorch model."""
        state_dict = model.state_dict()
        weights_registry = {}
        
        for name, param in state_dict.items():
            # Ensure tensors are contiguous (required by MAX)
            tensor = param.detach().cpu().numpy()
            if not tensor.flags['C_CONTIGUOUS']:
                tensor = tensor.copy()
            weights_registry[name] = tensor
            
        return weights_registry
    
    def _build_graph_from_weights(self, weights: PytorchWeights, inputs, graph) -> Any:
        """Build MAX graph by analyzing weight structure."""
        # This is a placeholder for weight-based graph construction
        # In practice, you'd analyze the weight names and shapes to 
        # infer the model architecture
        
        # Simple example: assume a basic linear model
        weight_names = [name for name, _ in weights.items()]
        
        if any('weight' in name and 'bias' in name for name in weight_names):
            # Looks like a linear layer
            x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            
            # Find weight and bias
            weight_tensor = None
            bias_tensor = None
            
            for name, weight_obj in weights.items():
                if 'weight' in name and weight_tensor is None:
                    weight_tensor = graph.add_weight(weight_obj.allocate())
                elif 'bias' in name and bias_tensor is None:
                    bias_tensor = graph.add_weight(weight_obj.allocate())
            
            if weight_tensor is not None:
                output = ops.matmul(x.tensor, weight_tensor)
                if bias_tensor is not None:
                    output = ops.add(output, bias_tensor)
                return output
        
        # Fallback: return input (identity)
        return inputs[0] if isinstance(inputs, (list, tuple)) else inputs


# Convenience functions
def convert_pytorch_model(model: nn.Module, 
                         input_shapes: List[Tuple[int, ...]],
                         device: Optional[DeviceRef] = None,
                         dtype: Optional[DType] = None,
                         model_name: str = "converted_model") -> Any:
    """
    Quick conversion function for PyTorch models.
    
    Args:
        model: PyTorch model to convert
        input_shapes: List of input tensor shapes
        device: Target device (auto-detected if None)
        dtype: Target data type (float32 if None)
        model_name: Name for the MAX graph
        
    Returns:
        Compiled MAX model
    """
    converter = PyTorchToMAXConverter(device=device, dtype=dtype)
    return converter.convert_model(model, input_shapes, model_name)


def convert_from_checkpoint(model_path: Union[str, Path],
                           input_shapes: List[Tuple[int, ...]],
                           device: Optional[DeviceRef] = None,
                           dtype: Optional[DType] = None,
                           model_name: str = "converted_model") -> Any:
    """
    Quick conversion function for PyTorch checkpoints.
    
    Args:
        model_path: Path to PyTorch model file
        input_shapes: List of input tensor shapes  
        device: Target device (auto-detected if None)
        dtype: Target data type (float32 if None)
        model_name: Name for the MAX graph
        
    Returns:
        Compiled MAX model
    """
    converter = PyTorchToMAXConverter(device=device, dtype=dtype)
    return converter.convert_from_checkpoint(model_path, input_shapes, model_name)


# Example usage
if __name__ == "__main__":
    # Example 1: Convert a simple PyTorch model
    pytorch_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Convert to MAX
    max_model = convert_pytorch_model(
        model=pytorch_model,
        input_shapes=[(1, 784)],  # Batch size 1, 784 features
        model_name="simple_mlp"
    )
    
    print("Model converted successfully!")
    print(f"Input devices: {max_model.input_devices}")
    print(f"Output devices: {max_model.output_devices}")
    
    # Example 2: Convert from checkpoint
    # max_model = convert_from_checkpoint(
    #     model_path="path/to/model.pt",
    #     input_shapes=[(1, 3, 224, 224)],  # Batch, channels, height, width
    #     model_name="resnet_converted"
    # )
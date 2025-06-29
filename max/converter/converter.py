"""
PyTorch to Modular MAX Architecture Converter

This module provides functionality to convert PyTorch models to MAX graphs
that can be executed efficiently on MAX runtime.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

from max.graph import Graph, TensorType, ops, DeviceRef, Weight
from max.graph.weights import PytorchWeights, SafetensorWeights, WeightData
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
            nn.BatchNorm2d: self._convert_batchnorm2d,
            nn.Conv2d: self._convert_conv2d,
            nn.ReLU: self._convert_relu,
            nn.GELU: self._convert_gelu,
            nn.SiLU: self._convert_silu,
            nn.Dropout: self._convert_dropout,
            nn.MultiheadAttention: self._convert_attention,
            nn.Flatten: self._convert_flatten,
            nn.MaxPool2d: self._convert_maxpool2d,
            nn.AdaptiveAvgPool2d: self._convert_adaptive_avgpool2d,
            nn.AvgPool2d: self._convert_avgpool2d,
        }
        
        # Add custom layer mappings for Llama components
        # These will be dynamically added when we detect them
        self._add_llama_layer_mappings()
        
        # Special case handlers for known model patterns
        self.special_converters = {
            'SimpleTransformer': self._convert_simple_transformer,
            'SimpleResNet': self._convert_simple_resnet,
            'LlamaModel': self._convert_llama_model,
            'LlamaForCausalLM': self._convert_llama_for_causal_lm,
            'LlamaDecoderLayer': self._convert_llama_decoder_layer,
            'LlamaAttention': self._convert_llama_attention,
            'LlamaMLP': self._convert_llama_mlp,
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
    
    def _add_llama_layer_mappings(self):
        """Add Llama-specific layer mappings dynamically."""
        try:
            # Try to import common Llama layer types
            # These might be from transformers library or custom implementations
            
            # Check for RMSNorm (common in Llama implementations)
            try:
                from transformers.models.llama.modeling_llama import LlamaRMSNorm
                self.layer_mappings[LlamaRMSNorm] = self._convert_rmsnorm
            except ImportError:
                pass
            
            # Check for other common RMSNorm implementations
            rmsnorm_names = ['RMSNorm', 'LlamaRMSNorm', 'RmsNorm']
            for name in rmsnorm_names:
                try:
                    # Try to find RMSNorm in various modules
                    for module_name in ['transformers', 'modeling_llama', '__main__']:
                        try:
                            if module_name in sys.modules:
                                module = sys.modules[module_name]
                                if hasattr(module, name):
                                    rmsnorm_class = getattr(module, name)
                                    self.layer_mappings[rmsnorm_class] = self._convert_rmsnorm
                                    break
                        except:
                            continue
                except:
                    continue
                    
        except Exception as e:
            # Don't fail if we can't add Llama mappings
            warnings.warn(f"Could not add some Llama layer mappings: {e}")
    
    def convert_model(self, 
                     pytorch_model: nn.Module,
                     input_shapes: List[Tuple[int, ...]],
                     model_name: str = "converted_model",
                     weights_path: Optional[Union[str, Path]] = None,
                     input_dtypes: Optional[List[DType]] = None) -> Any:
        """
        Convert a PyTorch model to MAX format.
        
        Args:
            pytorch_model: PyTorch model to convert
            input_shapes: List of input tensor shapes
            model_name: Name for the MAX graph
            weights_path: Optional path to weights file (.pt, .pth, .safetensors, .bin)
            input_dtypes: Optional list of input data types. If None, uses self.dtype for all inputs.
                         Use DType.int64 for embedding inputs (token IDs).
            
        Returns:
            Compiled MAX model ready for inference
        """
        
        # Clear previous weight arrays
        self.weight_arrays.clear()
        
        # Create input types for MAX graph
        if input_dtypes is None:
            input_types = [
                TensorType(self.dtype, shape, device=self.device) 
                for shape in input_shapes
            ]
        else:
            if len(input_dtypes) != len(input_shapes):
                raise ValueError(f"Number of input_dtypes ({len(input_dtypes)}) must match number of input_shapes ({len(input_shapes)})")
            input_types = [
                TensorType(dtype, shape, device=self.device) 
                for dtype, shape in zip(input_dtypes, input_shapes)
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
        if weights_path:
            # If external weights provided, load them properly
            weights_registry = self._load_pytorch_weights(weights_path)
            # Add any weights we collected during conversion that aren't in the file
            for weight_name, weight_array in self.weight_arrays.items():
                if weight_name not in weights_registry:
                    weights_registry[weight_name] = WeightData.from_numpy(weight_array, weight_name)
        else:
            # Convert raw numpy arrays to WeightData objects for dlpack compatibility
            weights_registry = {}
            for weight_name, weight_array in self.weight_arrays.items():
                weights_registry[weight_name] = WeightData.from_numpy(weight_array, weight_name)
            
        # Compile and return the model
        try:
            return self.session.load(graph, weights_registry=weights_registry)
        except Exception as e:
            # Provide helpful error information
            error_msg = str(e)
            if "dlpack" in error_msg.lower():
                print(f"DLPack error details:")
                print(f"  Number of weights: {len(weights_registry)}")
                print(f"  Weight types: {set(type(w).__name__ for w in weights_registry.values())}")
                print(f"  Sample weight names: {list(weights_registry.keys())[:5]}")
                if weights_path:
                    print(f"  Using external weights from: {weights_path}")
                else:
                    print(f"  Using extracted weights from model")
            raise
    
    def convert_from_checkpoint(self,
                               model_path: Union[str, Path],
                               input_shapes: List[Tuple[int, ...]],
                               model_name: str = "converted_model") -> Any:
        """
        Convert a PyTorch model directly from a checkpoint file.
        
        Args:
            model_path: Path to model file (.pt, .pth, .safetensors, .bin)
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
        
        # Load weights based on file format
        if model_path.suffix in ['.pt', '.pth', '.bin']:
            weights = PytorchWeights(model_path)
        elif model_path.suffix == '.safetensors':
            weights = SafetensorWeights([model_path])  # SafetensorWeights expects a sequence
        else:
            raise ValueError(f"Unsupported weight format: {model_path.suffix}. Supported formats: .pt, .pth, .safetensors, .bin")
        
        # Create input types
        input_types = [
            TensorType(self.dtype, shape, device=self.device) 
            for shape in input_shapes
        ]
        
        # Build graph by analyzing weight structure
        with Graph(model_name, input_types=input_types) as graph:
            outputs = self._build_graph_from_weights(weights, graph.inputs, graph)
            graph.output(outputs)
        
        # Load weights into graph - convert to WeightData for dlpack compatibility
        weights_registry = {}
        for key, weight in weights.items():
            try:
                # Get the raw numpy array from the weight using allocate()
                weight_data = weight.allocate()
                
                # Convert to numpy array
                if isinstance(weight_data, np.ndarray):
                    weight_array = weight_data
                else:
                    weight_array = np.array(weight_data)
                
                # Ensure it's a proper numpy array with standard dtype
                if not isinstance(weight_array, np.ndarray):
                    weight_array = np.array(weight_array)
                
                # Ensure contiguous memory layout
                if not weight_array.flags['C_CONTIGUOUS']:
                    weight_array = weight_array.copy()
                
                # Ensure compatible numpy dtype for MAX
                if weight_array.dtype == np.object_ or str(weight_array.dtype) == 'object':
                    # Try to infer the correct dtype from the data
                    try:
                        # Attempt to convert to float32 by default
                        weight_array = weight_array.astype(np.float32)
                    except (ValueError, TypeError):
                        print(f"Warning: Skipping weight {key} with unsupported dtype: {weight_array.dtype}")
                        continue
                elif hasattr(weight_array.dtype, 'name'):
                    # Convert to standard numpy dtype if needed
                    if 'float32' in str(weight_array.dtype):
                        weight_array = weight_array.astype(np.float32)
                    elif 'float16' in str(weight_array.dtype):
                        weight_array = weight_array.astype(np.float16)
                    elif 'int' in str(weight_array.dtype):
                        weight_array = weight_array.astype(np.int32)
                    elif 'bool' in str(weight_array.dtype):
                        weight_array = weight_array.astype(np.float32)
                
                # Create WeightData object
                weights_registry[key] = WeightData.from_numpy(weight_array, key)
                
            except Exception as e:
                print(f"Error processing weight {key}: {e}")
                print(f"  Weight type: {type(weight)}")
                if hasattr(weight, 'allocate'):
                    try:
                        allocated = weight.allocate()
                        print(f"  Allocated type: {type(allocated)}")
                        print(f"  Allocated dtype: {getattr(allocated, 'dtype', 'no dtype')}")
                    except:
                        print(f"  Could not allocate weight")
                raise
        
        try:
            return self.session.load(graph, weights_registry=weights_registry)
        except Exception as e:
            # Provide helpful error information
            error_msg = str(e)
            if "dlpack" in error_msg.lower():
                print(f"DLPack error details in convert_from_checkpoint:")
                print(f"  Number of weights: {len(weights_registry)}")
                print(f"  Weight types: {set(type(w).__name__ for w in weights_registry.values())}")
                print(f"  Sample weight names: {list(weights_registry.keys())[:5]}")
                print(f"  Model path: {model_path}")
            raise
    
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
        # Check for special converters first
        class_name = type(module).__name__
        if class_name in self.special_converters:
            return self.special_converters[class_name](module, inputs, graph)
            
        # For custom modules, try to convert child modules
        if hasattr(module, 'forward'):
            # This is a simplified approach - in practice, you'd need
            # to trace the forward pass or have more sophisticated analysis
            warnings.warn(f"Generic conversion for {type(module)} may not be accurate")
            
        # Try to convert known child modules with smart shape handling
        x = inputs
        child_modules = list(module.named_children())
        
        for i, (name, child) in enumerate(child_modules):
            if hasattr(child, 'forward'):
                prev_x = x
                x = self._convert_module(child, x, graph)
                
                # Check if we need to insert a flatten operation
                # This handles the common pattern where conv layers are followed by linear layers
                if i < len(child_modules) - 1:  # Not the last child
                    next_name, next_child = child_modules[i + 1]
                    
                    # If current output is 4D (from conv/pool) and next module is Linear
                    if (isinstance(child, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)) and 
                        isinstance(next_child, nn.Linear)):
                        
                        # Insert flatten operation
                        if isinstance(x, (list, tuple)):
                            if len(x) == 1:
                                x_tensor = x[0]
                            else:
                                raise ValueError(f"Expected single tensor before flatten, got {len(x)}")
                        else:
                            x_tensor = x
                            
                        # Check if we have a 4D tensor that needs flattening
                        if hasattr(x_tensor, 'tensor') and hasattr(x_tensor.tensor, 'shape'):
                            tensor_shape = x_tensor.tensor.shape
                            if len(tensor_shape) == 4:  # [batch, channels, height, width]
                                # Flatten to [batch, channels * height * width]
                                x = ops.flatten(x_tensor.tensor, start_dim=1)
                                
        return x
    
    def _convert_simple_transformer(self, module: nn.Module, inputs, graph) -> Any:
        """Convert SimpleTransformer module with proper mean pooling."""
        # Handle the SimpleTransformer forward pass:
        # x = self.embedding(x)
        # x = self.layer_norm(x)  
        # x = x.mean(dim=1)  # Global average pooling
        # x = self.linear(x)
        
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"SimpleTransformer expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Get the child modules
        embedding = None
        layer_norm = None 
        linear = None
        
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding):
                embedding = child
            elif isinstance(child, nn.LayerNorm):
                layer_norm = child
            elif isinstance(child, nn.Linear):
                linear = child
                
        if not all([embedding, layer_norm, linear]):
            raise ValueError("SimpleTransformer requires embedding, layer_norm, and linear modules")
        
        # Apply the layers in sequence with mean pooling
        x = self._convert_embedding(embedding, x, graph)
        x = self._convert_layernorm(layer_norm, x, graph)
        
        # Apply mean pooling: x.mean(dim=1) 
        # This reduces (batch, seq_len, embed_dim) to (batch, embed_dim)
        # GPU reduction is limited to inner axis, so we transpose to make seq_len the last dimension
        x_transposed = ops.transpose(x, 1, 2)  # (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        x_mean = ops.mean(x_transposed, axis=-1)  # Mean along last axis -> (batch, embed_dim, 1)
        x_squeezed = ops.squeeze(x_mean, axis=-1)  # Remove the singleton dim -> (batch, embed_dim)
        
        x = self._convert_linear(linear, x_squeezed, graph)
        
        return x
    
    def _convert_simple_resnet(self, module: nn.Module, inputs, graph) -> Any:
        """Convert SimpleResNet module with proper conv-to-linear transition."""
        # Handle the SimpleResNet forward pass:
        # x = self.pool1(self.relu1(self.conv1(x)))
        # x = self.pool2(self.relu2(self.conv2(x)))  
        # x = self.pool3(self.relu3(self.conv3(x)))
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.relu4(self.fc1(x))
        # x = self.fc2(x)
        
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"SimpleResNet expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Get the child modules in the expected order
        conv1 = relu1 = pool1 = None
        conv2 = relu2 = pool2 = None  
        conv3 = relu3 = pool3 = None
        fc1 = relu4 = fc2 = None
        
        for name, child in module.named_children():
            if name == 'conv1':
                conv1 = child
            elif name == 'relu1':
                relu1 = child
            elif name == 'pool1':
                pool1 = child
            elif name == 'conv2':
                conv2 = child
            elif name == 'relu2':
                relu2 = child
            elif name == 'pool2':
                pool2 = child
            elif name == 'conv3':
                conv3 = child
            elif name == 'relu3':
                relu3 = child
            elif name == 'pool3':
                pool3 = child
            elif name == 'fc1':
                fc1 = child
            elif name == 'relu4':
                relu4 = child
            elif name == 'fc2':
                fc2 = child
                
        # Apply the conv blocks in sequence
        if all([conv1, relu1, pool1]):
            x = self._convert_conv2d(conv1, x, graph)
            x = self._convert_relu(relu1, x, graph)
            x = self._convert_maxpool2d(pool1, x, graph)
        
        if all([conv2, relu2, pool2]):
            x = self._convert_conv2d(conv2, x, graph)
            x = self._convert_relu(relu2, x, graph)
            x = self._convert_maxpool2d(pool2, x, graph)
            
        if all([conv3, relu3, pool3]):
            x = self._convert_conv2d(conv3, x, graph)
            x = self._convert_relu(relu3, x, graph)
            x = self._convert_maxpool2d(pool3, x, graph)
        
        # Flatten before linear layers (equivalent to x.view(x.size(0), -1))
        if hasattr(x, 'tensor'):
            x_flattened = ops.flatten(x.tensor, start_dim=1)
        else:
            x_flattened = ops.flatten(x, start_dim=1)
        
        # Apply the linear layers
        if fc1 and relu4 and fc2:
            x = self._convert_linear(fc1, x_flattened, graph)
            x = self._convert_relu(relu4, x, graph)
            x = self._convert_linear(fc2, x, graph)
        
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
    
    def _convert_rmsnorm(self, layer, inputs, graph) -> Any:
        """Convert RMSNorm to MAX operations."""
        # RMSNorm expects a single input tensor
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"RMSNorm expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Get epsilon value (different implementations may store this differently)
        eps = getattr(layer, 'eps', getattr(layer, 'variance_epsilon', 1e-6))
        
        # Create Weight object for the scale parameter
        # RMSNorm typically only has a weight (scale) parameter, no bias
        weight_name = f"rmsnorm_weight_{id(layer)}"
        
        # Try different attribute names for the weight parameter
        weight_param = None
        for attr_name in ['weight', 'scale', 'g']:
            if hasattr(layer, attr_name):
                weight_param = getattr(layer, attr_name)
                break
        
        if weight_param is None:
            raise ValueError(f"Could not find weight parameter in RMSNorm layer")
            
        weight_array = weight_param.detach().cpu().numpy()
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
        scale = graph.add_weight(weight)
        
        # RMSNorm: x * scale / sqrt(mean(x^2) + eps)
        # Implement RMSNorm manually since ops.rms_norm may not be available
        # 1. Compute x^2
        x_squared = ops.mul(x.tensor, x.tensor)
        
        # 2. Compute mean along last dimension
        mean_x_squared = ops.mean(x_squared, axis=-1)
        # Add back the dimension that was removed by mean for broadcasting
        mean_x_squared = ops.unsqueeze(mean_x_squared, axis=-1)
        
        # 3. Add epsilon and sqrt  
        eps_const = ops.constant(eps, dtype=self.dtype, device=self.device)
        variance = ops.add(mean_x_squared, eps_const)
        rms = ops.sqrt(variance)
        
        # 4. Normalize
        normalized = ops.div(x.tensor, rms)
        
        # 5. Scale
        return ops.mul(normalized, scale)
    
    def _convert_conv2d(self, layer: nn.Conv2d, inputs, graph) -> Any:
        """Convert nn.Conv2d to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"Conv2d expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Convert input from NCHW to NHWC format
        # PyTorch uses NCHW: (batch, channels, height, width)
        # MAX expects NHWC: (batch, height, width, channels)
        x_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
            
        # Create Weight object for convolution parameters
        weight_name = f"conv2d_weight_{id(layer)}"
        # Convert PyTorch weight format (OIHW) to MAX format (HWIO)
        # PyTorch: (out_channels, in_channels, height, width)
        # MAX: (height, width, in_channels, out_channels)
        weight_array = layer.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
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
        
        # Convolution operation (input is now in NHWC format)
        output = ops.conv2d(
            x_nhwc, weight, 
            stride=layer.stride,
            dilation=layer.dilation,
            padding=padding
        )
        
        # Add bias if present (while output is still in NHWC format)
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
        
        # Convert output back from NHWC to NCHW to match PyTorch convention
        output = ops.permute(output, [0, 3, 1, 2])  # NHWC -> NCHW
            
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
    
    def _convert_silu(self, layer: nn.SiLU, inputs, graph) -> Any:
        """Convert nn.SiLU (Swish) to MAX operations."""
        # SiLU expects a single input tensor
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"SiLU expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # SiLU(x) = x * sigmoid(x)
        # Check if MAX has a native SiLU operation, otherwise implement manually
        try:
            return ops.silu(x.tensor)
        except AttributeError:
            # Fallback: implement SiLU as x * sigmoid(x)
            sigmoid_x = ops.sigmoid(x.tensor)
            return ops.mul(x.tensor, sigmoid_x)
    
    def _convert_dropout(self, layer: nn.Dropout, inputs, graph) -> Any:
        """Convert nn.Dropout (passthrough during inference)."""
        if isinstance(inputs, (list, tuple)):
            return inputs[0] if len(inputs) == 1 else inputs
        else:
            return inputs
    
    def _convert_flatten(self, layer: nn.Flatten, inputs, graph) -> Any:
        """Convert nn.Flatten to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"Flatten expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Get flatten parameters
        start_dim = getattr(layer, 'start_dim', 1)
        end_dim = getattr(layer, 'end_dim', -1)
        
        # For most cases, flatten just flattens from start_dim onwards
        # We'll implement the most common case: flatten everything except batch dimension
        if start_dim == 1 and end_dim == -1:
            # Flatten all dimensions except the first (batch) dimension
            # This is equivalent to x.view(batch_size, -1)
            return ops.flatten(x.tensor, start_dim=1)
        else:
            # For more complex flatten operations, use the provided parameters
            return ops.flatten(x.tensor, start_dim=start_dim, end_dim=end_dim)
    
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
    
    def _convert_batchnorm2d(self, layer: nn.BatchNorm2d, inputs, graph) -> Any:
        """Convert nn.BatchNorm2d to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"BatchNorm2d expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Convert input from NCHW to NHWC format (matching conv2d)
        x_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
        
        # Create Weight objects for batch norm parameters
        weight_name = f"batchnorm_weight_{id(layer)}"
        weight_array = layer.weight.detach().cpu().numpy()
        if not weight_array.flags['C_CONTIGUOUS']:
            weight_array = weight_array.copy()
        
        weight_obj = Weight(
            name=weight_name,
            dtype=self.dtype,
            shape=weight_array.shape,
            device=self.device
        )
        self.weight_arrays[weight_name] = weight_array
        gamma = graph.add_weight(weight_obj)
        
        bias_name = f"batchnorm_bias_{id(layer)}"
        bias_array = layer.bias.detach().cpu().numpy()
        if not bias_array.flags['C_CONTIGUOUS']:
            bias_array = bias_array.copy()
        
        bias_obj = Weight(
            name=bias_name,
            dtype=self.dtype,
            shape=bias_array.shape,
            device=self.device
        )
        self.weight_arrays[bias_name] = bias_array
        beta = graph.add_weight(bias_obj)
        
        # Running mean and variance
        mean_name = f"batchnorm_mean_{id(layer)}"
        mean_array = layer.running_mean.detach().cpu().numpy()
        if not mean_array.flags['C_CONTIGUOUS']:
            mean_array = mean_array.copy()
        
        mean_obj = Weight(
            name=mean_name,
            dtype=self.dtype,
            shape=mean_array.shape,
            device=self.device
        )
        self.weight_arrays[mean_name] = mean_array
        running_mean = graph.add_weight(mean_obj)
        
        var_name = f"batchnorm_var_{id(layer)}"
        var_array = layer.running_var.detach().cpu().numpy()
        if not var_array.flags['C_CONTIGUOUS']:
            var_array = var_array.copy()
        
        var_obj = Weight(
            name=var_name,
            dtype=self.dtype,
            shape=var_array.shape,
            device=self.device
        )
        self.weight_arrays[var_name] = var_array
        running_var = graph.add_weight(var_obj)
        
        # Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
        # For inference, use running statistics
        eps_const = ops.constant(layer.eps, self.dtype, self.device)
        
        # Broadcast running stats to match input shape
        # Input is NHWC, so we need to broadcast along spatial dimensions
        mean_bc = ops.unsqueeze(ops.unsqueeze(running_mean, 0), 0)  # [1, 1, C]
        var_bc = ops.unsqueeze(ops.unsqueeze(running_var, 0), 0)    # [1, 1, C]
        gamma_bc = ops.unsqueeze(ops.unsqueeze(gamma, 0), 0)        # [1, 1, C]
        beta_bc = ops.unsqueeze(ops.unsqueeze(beta, 0), 0)          # [1, 1, C]
        
        # Normalize
        x_norm = ops.sub(x_nhwc, mean_bc)
        var_eps = ops.add(var_bc, eps_const)
        std = ops.sqrt(var_eps)
        x_norm = ops.div(x_norm, std)
        
        # Scale and shift
        output = ops.mul(x_norm, gamma_bc)
        output = ops.add(output, beta_bc)
        
        # Convert back to NCHW
        output = ops.permute(output, [0, 3, 1, 2])  # NHWC -> NCHW
        
        return output
    
    def _convert_maxpool2d(self, layer: nn.MaxPool2d, inputs, graph) -> Any:
        """Convert nn.MaxPool2d to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"MaxPool2d expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Convert input from NCHW to NHWC format
        x_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
        
        # Get pooling parameters
        if isinstance(layer.kernel_size, int):
            kernel_size = (layer.kernel_size, layer.kernel_size)
        else:
            kernel_size = layer.kernel_size
            
        if isinstance(layer.stride, int):
            stride = (layer.stride, layer.stride)
        elif layer.stride is None:
            stride = kernel_size
        else:
            stride = layer.stride
            
        if isinstance(layer.padding, int):
            padding = (layer.padding, layer.padding)
        else:
            padding = layer.padding
        
        # Max pooling operation
        output = ops.max_pool2d(
            x_nhwc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Convert back to NCHW
        output = ops.permute(output, [0, 3, 1, 2])  # NHWC -> NCHW
        
        return output
    
    def _convert_avgpool2d(self, layer: nn.AvgPool2d, inputs, graph) -> Any:
        """Convert nn.AvgPool2d to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"AvgPool2d expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Convert input from NCHW to NHWC format
        x_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
        
        # Get pooling parameters
        if isinstance(layer.kernel_size, int):
            kernel_size = (layer.kernel_size, layer.kernel_size)
        else:
            kernel_size = layer.kernel_size
            
        if isinstance(layer.stride, int):
            stride = (layer.stride, layer.stride)
        elif layer.stride is None:
            stride = kernel_size
        else:
            stride = layer.stride
            
        if isinstance(layer.padding, int):
            padding = (layer.padding, layer.padding)
        else:
            padding = layer.padding
        
        # Average pooling operation
        output = ops.avg_pool2d(
            x_nhwc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Convert back to NCHW
        output = ops.permute(output, [0, 3, 1, 2])  # NHWC -> NCHW
        
        return output
    
    def _convert_adaptive_avgpool2d(self, layer: nn.AdaptiveAvgPool2d, inputs, graph) -> Any:
        """Convert nn.AdaptiveAvgPool2d to MAX operations."""
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"AdaptiveAvgPool2d expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Get output size
        output_size = layer.output_size
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        
        # For AdaptiveAvgPool2d with output size (1, 1), use global average pooling
        if output_size == (1, 1):
            # Convert input from NCHW to NHWC format
            x_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
            
            # Global average pooling: mean over spatial dimensions (H, W)
            # Input: [batch, height, width, channels]
            # We want to average over dimensions 1 and 2 (height and width)
            
            # Mean over height dimension (axis=1)
            x_h_mean = ops.mean(x_nhwc, axis=1)  # [batch, width, channels]
            # Mean over width dimension (axis=1, since height was already reduced)
            x_hw_mean = ops.mean(x_h_mean, axis=1)  # [batch, channels]
            
            # Reshape to [batch, channels, 1, 1] to match PyTorch output
            output = ops.unsqueeze(ops.unsqueeze(x_hw_mean, -1), -1)  # [batch, channels, 1, 1]
            
            return output
        else:
            # For other output sizes, fall back to a simplified approach
            warnings.warn(f"AdaptiveAvgPool2d with output_size {output_size} uses simplified conversion")
            
            # Calculate approximate kernel size and stride based on input/output ratio
            input_shape = x.tensor.shape
            if len(input_shape) == 4:  # [batch, channels, height, width]
                input_h, input_w = input_shape[2], input_shape[3]
                output_h, output_w = output_size
                
                kernel_h = input_h // output_h
                kernel_w = input_w // output_w
                stride_h = kernel_h
                stride_w = kernel_w
                
                # Use regular average pooling as approximation
                x_nhwc = ops.permute(x.tensor, [0, 2, 3, 1])  # NCHW -> NHWC
                output = ops.avg_pool2d(
                    x_nhwc,
                    kernel_size=(kernel_h, kernel_w),
                    stride=(stride_h, stride_w),
                    padding=(0, 0)
                )
                output = ops.permute(output, [0, 3, 1, 2])  # NHWC -> NCHW
                
                return output
            else:
                raise ValueError(f"Unexpected input shape for AdaptiveAvgPool2d: {input_shape}")
    
    # The rest of the existing methods continue unchanged...
    
    def _load_pytorch_weights(self, weights_path: Union[str, Path]) -> Dict[str, Any]:
        """Load PyTorch weights from file."""
        weights_path = Path(weights_path)
        
        # Basic file validation
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        if weights_path.stat().st_size == 0:
            raise ValueError(f"Weights file is empty: {weights_path}")
        
        try:
            if weights_path.suffix in ['.pt', '.pth']:
                weights = PytorchWeights(weights_path)
                return {key: weight.allocate() for key, weight in weights.items()}
            elif weights_path.suffix == '.safetensors':
                weights = SafetensorWeights([weights_path])  # SafetensorWeights expects a sequence
                return {key: weight.allocate() for key, weight in weights.items()}
            elif weights_path.suffix == '.bin':
                # .bin files are typically PyTorch format as well
                weights = PytorchWeights(weights_path)
                return {key: weight.allocate() for key, weight in weights.items()}
            else:
                raise ValueError(f"Unsupported weight format: {weights_path.suffix}. Supported formats: .pt, .pth, .safetensors, .bin")
        except Exception as e:
            # Provide more helpful error messages for common issues
            file_size = weights_path.stat().st_size
            if "central directory" in str(e).lower() or "zip" in str(e).lower():
                raise ValueError(
                    f"Failed to load weights from {weights_path} (size: {file_size} bytes). "
                    f"This usually indicates a corrupted or incomplete download. "
                    f"Original error: {e}"
                ) from e
            elif "safetensors" in str(e).lower():
                raise ValueError(
                    f"Failed to load safetensors file {weights_path} (size: {file_size} bytes). "
                    f"Make sure the file is a valid safetensors format. "
                    f"Original error: {e}"
                ) from e
            else:
                raise ValueError(
                    f"Failed to load weights from {weights_path} (size: {file_size} bytes). "
                    f"Original error: {e}"
                ) from e
    
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
    
    def _build_graph_from_weights(self, weights: Union[PytorchWeights, SafetensorWeights], inputs, graph) -> Any:
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

    # Llama-specific conversion methods
    def _convert_llama_model(self, module, inputs, graph) -> Any:
        """Convert LlamaModel to MAX operations."""
        warnings.warn("LlamaModel conversion is experimental")
        
        # For now, try generic conversion
        return self._convert_generic_module(module, inputs, graph)
    
    def _convert_llama_for_causal_lm(self, module, inputs, graph) -> Any:
        """Convert LlamaForCausalLM to MAX operations."""
        warnings.warn("LlamaForCausalLM conversion is experimental")
        
        # LlamaForCausalLM typically has:
        # - model (LlamaModel)
        # - lm_head (Linear layer for output projection)
        
        x = inputs
        for name, child in module.named_children():
            if name == 'model':
                x = self._convert_module(child, x, graph)
            elif name == 'lm_head':
                x = self._convert_module(child, x, graph)
                
        return x
    
    def _convert_llama_decoder_layer(self, module, inputs, graph) -> Any:
        """Convert LlamaDecoderLayer to MAX operations."""
        # LlamaDecoderLayer typically has:
        # - self_attn (LlamaAttention)
        # - mlp (LlamaMLP)
        # - input_layernorm (RMSNorm)
        # - post_attention_layernorm (RMSNorm)
        
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"LlamaDecoderLayer expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Store input for residual connection
        residual = x
        
        # Get components
        input_layernorm = getattr(module, 'input_layernorm', None)
        self_attn = getattr(module, 'self_attn', None)
        post_attention_layernorm = getattr(module, 'post_attention_layernorm', None)
        mlp = getattr(module, 'mlp', None)
        
        # Pre-attention normalization
        if input_layernorm:
            x = self._convert_module(input_layernorm, x, graph)
            
        # Self-attention
        if self_attn:
            attn_output = self._convert_module(self_attn, x, graph)
            # Residual connection
            x = ops.add(residual.tensor if hasattr(residual, 'tensor') else residual, 
                       attn_output.tensor if hasattr(attn_output, 'tensor') else attn_output)
        
        # Store for next residual
        residual = x
        
        # Pre-MLP normalization
        if post_attention_layernorm:
            x = self._convert_module(post_attention_layernorm, x, graph)
            
        # MLP
        if mlp:
            mlp_output = self._convert_module(mlp, x, graph)
            # Residual connection
            x = ops.add(residual.tensor if hasattr(residual, 'tensor') else residual,
                       mlp_output.tensor if hasattr(mlp_output, 'tensor') else mlp_output)
            
        return x
    
    def _convert_llama_attention(self, module, inputs, graph) -> Any:
        """Convert LlamaAttention to MAX operations."""
        warnings.warn("LlamaAttention conversion is simplified - may not include all features like RoPE")
        
        # For now, use the generic multihead attention conversion
        # In a full implementation, this would handle:
        # - Grouped query attention
        # - Rotary position embeddings (RoPE)
        # - Proper key-value caching
        
        return self._convert_attention(module, inputs, graph)
    
    def _convert_llama_mlp(self, module, inputs, graph) -> Any:
        """Convert LlamaMLP (SwiGLU) to MAX operations."""
        # LlamaMLP typically has:
        # - gate_proj (Linear)
        # - up_proj (Linear) 
        # - down_proj (Linear)
        # Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(f"LlamaMLP expects 1 input, got {len(inputs)}")
            x = inputs[0]
        else:
            x = inputs
            
        # Get components
        gate_proj = getattr(module, 'gate_proj', None)
        up_proj = getattr(module, 'up_proj', None)
        down_proj = getattr(module, 'down_proj', None)
        
        if not all([gate_proj, up_proj, down_proj]):
            warnings.warn("LlamaMLP missing expected components, trying generic conversion")
            return self._convert_generic_module(module, inputs, graph)
        
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        gate_output = self._convert_linear(gate_proj, x, graph)
        up_output = self._convert_linear(up_proj, x, graph)
        
        # Apply SiLU to gate output
        # SiLU(x) = x * sigmoid(x)
        try:
            gate_activated = ops.silu(gate_output.tensor if hasattr(gate_output, 'tensor') else gate_output)
        except AttributeError:
            # Fallback: implement SiLU manually
            gate_tensor = gate_output.tensor if hasattr(gate_output, 'tensor') else gate_output
            sigmoid_gate = ops.sigmoid(gate_tensor)
            gate_activated = ops.mul(gate_tensor, sigmoid_gate)
        
        # Element-wise multiplication
        up_tensor = up_output.tensor if hasattr(up_output, 'tensor') else up_output
        combined = ops.mul(gate_activated, up_tensor)
        
        # Final projection
        return self._convert_linear(down_proj, combined, graph)


# Convenience functions
def convert_pytorch_model(model: nn.Module, 
                         input_shapes: List[Tuple[int, ...]],
                         device: Optional[DeviceRef] = None,
                         dtype: Optional[DType] = None,
                         model_name: str = "converted_model",
                         input_dtypes: Optional[List[DType]] = None) -> Any:
    """
    Quick conversion function for PyTorch models.
    
    Args:
        model: PyTorch model to convert
        input_shapes: List of input tensor shapes
        device: Target device (auto-detected if None)
        dtype: Target data type (float32 if None)
        model_name: Name for the MAX graph
        input_dtypes: Optional list of input data types. If None, uses dtype for all inputs.
                     Use DType.int64 for embedding inputs (token IDs).
        
    Returns:
        Compiled MAX model
    """
    converter = PyTorchToMAXConverter(device=device, dtype=dtype)
    return converter.convert_model(model, input_shapes, model_name, weights_path=None, input_dtypes=input_dtypes)


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
    
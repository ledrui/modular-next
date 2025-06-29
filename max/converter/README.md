# PyTorch to MAX Converter

A Python library for converting PyTorch models to Modular MAX format for optimized inference.

## Features

- **Automatic Layer Mapping**: Supports common PyTorch layers (Linear, Conv2d, ReLU, GELU, LayerNorm, Embedding, etc.)
- **Weight Management**: Handles weight extraction and proper memory layout for MAX
- **Device Support**: Works with both CPU and GPU devices
- **Easy to Use**: Simple API for quick model conversion

## Quick Start

### Basic Usage

```python
import torch.nn as nn
from converter import convert_pytorch_model

# Create your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Convert to MAX
max_model = convert_pytorch_model(
    model=model,
    input_shapes=[(1, 784)],  # Specify input shapes
    model_name="my_model"
)

# Run inference
import numpy as np
from max.driver import Tensor

test_input = np.random.randn(1, 784).astype(np.float32)
max_input = Tensor.from_numpy(test_input)
result = max_model.execute(max_input)
```

### Advanced Usage

```python
from converter import PyTorchToMAXConverter
from max.dtype import DType
from max.graph import DeviceRef

# Create converter with specific settings
converter = PyTorchToMAXConverter(
    device=DeviceRef.GPU(),  # Force GPU usage
    dtype=DType.float32      # Specify data type
)

# Convert model
max_model = converter.convert_model(
    pytorch_model=model,
    input_shapes=[(32, 784)],  # Batch size 32
    model_name="gpu_model"
)
```

## Test Files

### 1. `quick_test.py` - Simple Quick Test

Run a basic test to verify the converter works:

```bash
pixi run python quick_test.py
```

This creates a simple MLP, converts it to MAX, and compares inference results.

### 2. `test_converter.py` - Comprehensive Test Suite

Run comprehensive tests with multiple model types:

```bash
pixi run python test_converter.py
```

Tests include:
- MLP models
- CNN models  
- Transformer-like models
- Different data types
- Different device placements
- Accuracy verification
- Performance comparison

### 3. `example_usage.py` - Practical Examples

See real-world usage scenarios:

```bash
pixi run python example_usage.py
```

Examples include:
- Simple model conversion
- Batch inference
- Performance benchmarking
- Different model architectures

## Supported PyTorch Layers

| PyTorch Layer | MAX Operation | Status |
|---------------|---------------|--------|
| `nn.Linear` | `ops.matmul` + `ops.add` | ✅ |
| `nn.ReLU` | `ops.relu` | ✅ |
| `nn.GELU` | `ops.gelu` | ✅ |
| `nn.LayerNorm` | `ops.layer_norm` | ✅ |
| `nn.Embedding` | `ops.gather` | ✅ |
| `nn.Conv2d` | `ops.conv2d` | ✅ |
| `nn.Dropout` | Identity (inference) | ✅ |
| `nn.MultiheadAttention` | Simplified | 🚧 |

## Requirements

- Python 3.8+
- PyTorch
- Modular MAX Python SDK
- NumPy

## Architecture

The converter works in three main steps:

1. **Graph Construction**: Converts PyTorch modules to MAX operations
2. **Weight Extraction**: Extracts weights and ensures contiguous memory layout
3. **Compilation**: Compiles the MAX graph and loads weights

### Key Components

- **`PyTorchToMAXConverter`**: Main converter class
- **Layer Mappings**: Dictionary mapping PyTorch layers to conversion functions
- **Weight Management**: Handles weight extraction and memory layout
- **Device Management**: Manages CPU/GPU device placement

## Limitations

- **Limited Layer Support**: Not all PyTorch layers are supported yet
- **Dynamic Shapes**: Input shapes must be known at conversion time
- **Control Flow**: Complex control flow (if/while) not supported
- **Custom Layers**: Custom PyTorch layers need manual mapping

## Error Handling

Common issues and solutions:

### "Weight is not contiguous"
The converter automatically handles this by copying non-contiguous tensors.

### "Device not set up in InferenceSession"
The converter automatically detects and configures available devices.

### "Layer not supported"
Add custom layer mapping or use supported equivalent layers.

## Contributing

To add support for new PyTorch layers:

1. Add the layer type to `layer_mappings` in `__init__`
2. Implement the conversion function (e.g., `_convert_my_layer`)
3. Handle weight extraction and device placement
4. Add tests in the test suite

Example:

```python
def _convert_my_layer(self, layer: nn.MyLayer, inputs, graph) -> Any:
    # Extract weights
    weight_name = f"my_layer_weight_{id(layer)}"
    weight_array = layer.weight.detach().cpu().numpy()
    
    # Ensure contiguous
    if not weight_array.flags['C_CONTIGUOUS']:
        weight_array = weight_array.copy()
    
    # Create MAX weight
    weight = Weight(
        name=weight_name,
        dtype=self.dtype,
        shape=weight_array.shape,
        device=self.device
    )
    
    # Store for later loading
    self.weight_arrays[weight_name] = weight_array
    weight_tensor = graph.add_weight(weight)
    
    # Implement the operation
    return ops.my_operation(inputs[0].tensor, weight_tensor)
```

## Performance Tips

1. **Use GPU**: GPU inference is typically much faster
2. **Batch Inference**: Process multiple samples together
3. **Contiguous Memory**: The converter handles this automatically
4. **Warm-up**: Run a few inference calls before benchmarking

## License

This converter is part of the Modular MAX ecosystem. See Modular's license terms. 
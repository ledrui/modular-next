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
pixi run python simple_llama_test.py
pixi pixi run  python cli.py convert simple_llama_test.pt --input-shapes "1,8" --output converted/ --verbose
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
| `nn.SiLU` | `ops.silu` (with fallback) | ✅ |
| `nn.BatchNorm2d` | `ops.batch_norm` | ✅ |
| `nn.MaxPool2d` | `ops.max_pool2d` | ✅ |
| `nn.AvgPool2d` | `ops.avg_pool2d` | ✅ |
| `nn.AdaptiveAvgPool2d` | `ops.avg_pool2d` | ✅ |

## Llama Model Support 🦙

The converter now includes **experimental support for Llama models** with native MAX operations:

### Supported Llama Components

| Component | Implementation | Status |
|-----------|----------------|--------|
| **RMSNorm** | Manual implementation with MAX ops | ✅ |
| **SiLU Activation** | `ops.silu` with fallback | ✅ |
| **SwiGLU MLP** | `down_proj(SiLU(gate_proj(x)) * up_proj(x))` | ✅ |
| **LlamaAttention** | Simplified (no RoPE/GQA) | 🚧 |
| **LlamaDecoderLayer** | Full residual connections | ✅ |
| **LlamaForCausalLM** | Complete architecture | ✅ |

### Testing with Sample Models

We provide sample Llama models for testing the converter:

#### Option 1: Full Llama Architecture
```bash
# Create a complete Llama model with proper components
pixi run python create_sample_llama.py

# Convert to MAX
python cli.py convert sample_llama.pt --input-shapes "1,10" --output converted/ --verbose
```

#### Option 2: Simple Test Model
```bash
# Create a simplified Llama-like model for quick testing
pixi run python simple_llama_test.py

# Convert to MAX
python cli.py convert simple_llama_test.pt --input-shapes "1,8" --output converted/ --verbose
```

### What's Included in Sample Models

**Full Llama Model (`create_sample_llama.py`):**
- ✅ Custom `RMSNorm` implementation
- ✅ `LlamaMLP` with SwiGLU activation
- ✅ `LlamaAttention` (simplified, no RoPE)
- ✅ `LlamaDecoderLayer` with residual connections
- ✅ Complete `LlamaForCausalLM` architecture
- 📊 Small size: ~1K vocab, 256 hidden, 2 layers

**Simple Test Model (`simple_llama_test.py`):**
- ✅ Basic embedding + linear layers
- ✅ SiLU activation
- ✅ LayerNorm (as RMSNorm substitute)
- ✅ Residual connections
- 📊 Tiny size: 100 vocab, 64 hidden, 1 layer

### Real Llama Model Conversion

To convert actual Llama models from Hugging Face:

```bash
# Download Llama model
python cli.py download meta-llama/Llama-2-7b-hf --output downloads/

# Convert to MAX (experimental)
python cli.py convert downloads/llama_2_7b_hf.safetensors --input-shapes "1,512" --output converted/ --verbose
```

### Current Limitations

- **RoPE (Rotary Position Embedding)**: Not implemented - uses simplified attention
- **Grouped Query Attention**: Falls back to standard multi-head attention
- **KV Caching**: Not implemented - affects inference efficiency
- **Attention Masking**: Basic implementation

### Expected Results

The converter should successfully handle:
- ✅ Basic text generation tasks
- ✅ Model architecture conversion
- ✅ Weight loading and format conversion
- ⚠️ May have reduced accuracy for complex tasks due to simplified attention

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
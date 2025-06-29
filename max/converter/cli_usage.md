# CLI Usage Guide

## Overview

The CLI now uses subcommands for better separation of concerns:
- `convert`: Convert local PyTorch models to MAX format
- `download`: Download models from Hugging Face Hub

## Basic Usage

### Convert a local PyTorch model to MAX format:
```bash
python cli.py convert /path/to/model.pt --input-shapes "1,784"
```

### Download a Hugging Face model:
```bash
python cli.py download microsoft/DialoGPT-medium --output downloads/
```

### Two-step workflow (recommended):
```bash
# Step 1: Download
python cli.py download bert-base-uncased --output downloads/

# Step 2: Convert  
python cli.py convert downloads/model.safetensors --input-shapes "1,512" --output converted/
```

## Commands

### `convert` - Convert PyTorch model to MAX format

**Required Arguments:**
- `model_path`: Path to PyTorch model file (.pt, .pth, .safetensors, .bin)
- `--input-shapes`: Input tensor shapes in format 'batch,dim1,dim2,...' or 'batch,dim1;batch,dim2' for multiple inputs

**Optional Arguments:**
- `--output, -o`: Output directory (default: models)
- `--model-name`: Name for the converted model (default: derived from filename)
- `--device`: Target device - cpu, gpu, or auto (default: auto)
- `--dtype`: Target data type - float32, float16, int8, int32, int64 (default: float32)
- `--verbose, -v`: Enable verbose output

### `download` - Download model from Hugging Face

**Required Arguments:**
- `model_path`: Hugging Face model URL/ID (e.g., microsoft/DialoGPT-medium)

**Optional Arguments:**
- `--output, -o`: Output directory for downloaded model
- `--hf-cache-dir`: Cache directory for Hugging Face models (default: system temp directory)
- `--verbose, -v`: Enable verbose output

## Examples

### Convert Local Models

#### Simple Linear Model
```bash
python cli.py convert model.pt --input-shapes "1,784" --output models/
```

#### CNN Model with GPU
```bash
python cli.py convert resnet.pth --input-shapes "1,3,224,224" --device gpu --output converted/
```

#### Multiple Inputs
```bash
python cli.py convert multimodal.pt --input-shapes "1,768;1,10" --verbose
```

#### Different Data Types
```bash
python cli.py convert quantized.pt --input-shapes "32,512" --dtype float16 --model-name fast_model
```

### Download Hugging Face Models

#### Download from model ID
```bash
python cli.py download microsoft/DialoGPT-medium --output downloads/
```

#### Download from full URL
```bash
python cli.py download https://huggingface.co/bert-base-uncased --output downloads/ --verbose
```

#### With custom cache directory
```bash
python cli.py download openai/gpt-2 --hf-cache-dir ./hf_cache --output downloads/
```

#### Download large model
```bash
python cli.py download facebook/opt-1.3b --verbose
```

### Complete Workflows

#### Download then convert workflow
```bash
# Step 1: Download
python cli.py download microsoft/DialoGPT-medium --output downloads/

# Step 2: Convert (the CLI will show you the exact command)
python cli.py convert downloads/pytorch_model.bin --input-shapes "1,512" --output converted/
```

#### Direct conversion with specific settings
```bash
python cli.py convert model.safetensors --input-shapes "32,1024" --device gpu --dtype float16 --verbose
```

## Output

The CLI will:
1. Convert the PyTorch model to MAX format
2. Save model information to `{model_name}_info.txt` in the output directory
3. Print conversion status and device information

## Error Handling

Common errors and solutions:

- **File not found**: Check that the model path is correct (for local files)
- **Unsupported format**: Only .pt, .pth, and .safetensors are supported
- **Invalid input shapes**: Use format 'batch,dim1,dim2,...' for single input or 'batch,dim1;batch,dim2' for multiple inputs
- **Module not found**: Ensure PyTorch and MAX are installed in your environment
- **HuggingFace Hub error**: Install huggingface_hub with `pip install huggingface_hub`
- **Model not found on HF**: Check that the model ID is correct and the model exists
- **Download failed**: Check internet connection and ensure you have access to the model (some models require authentication)

## Integration with Pixi

If using Pixi for environment management:

```bash
pixi run python cli.py model.pt --input-shapes "1,784"
```

## Integration with Bazel

If using Bazel build system:

```bash
./bazelw run //max/converter:cli -- model.pt --input-shapes "1,784"
```
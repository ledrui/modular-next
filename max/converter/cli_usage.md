# CLI Usage Guide

## Basic Usage

### Convert a local PyTorch model to MAX format:
```bash
python cli.py /path/to/model.pt --input-shapes "1,784"
```

### Download and convert a Hugging Face model:
```bash
python cli.py microsoft/DialoGPT-medium --input-shapes "1,512"
```

### Download only (without conversion):
```bash
python cli.py microsoft/DialoGPT-medium --download-only
```

## Command Line Arguments

### Required Arguments

- `model_path`: Path to PyTorch model file (.pt, .pth, .safetensors) or Hugging Face model URL/ID

### Conditionally Required Arguments

- `--input-shapes`: Input tensor shapes in format 'batch,dim1,dim2,...' (required for conversion, not needed with --download-only)

### Optional Arguments

- `--output, -o`: Output directory (default: models)
- `--model-name`: Name for the converted model (default: derived from filename or HF model ID)
- `--device`: Target device - cpu, gpu, or auto (default: auto)
- `--dtype`: Target data type - float32, float16, int8, int32, int64 (default: float32)
- `--verbose, -v`: Enable verbose output
- `--hf-cache-dir`: Cache directory for Hugging Face models (default: system temp directory)
- `--download-only`: Only download the model without converting (HF models only)

## Examples

### Simple Linear Model
```bash
python cli.py model.pt --input-shapes "1,784" --output models/
```

### CNN Model with GPU
```bash
python cli.py resnet.pth --input-shapes "1,3,224,224" --device gpu --output converted/
```

### Multiple Inputs
```bash
python cli.py multimodal.pt --input-shapes "1,768;1,10" --verbose
```

### Different Data Types
```bash
python cli.py quantized.pt --input-shapes "32,512" --dtype float16 --model-name fast_model
```

### Hugging Face Models

#### Download and convert from model ID
```bash
python cli.py microsoft/DialoGPT-medium --input-shapes "1,512" --output models/
```

#### Download from full URL
```bash
python cli.py https://huggingface.co/bert-base-uncased --input-shapes "1,512" --verbose
```

#### With custom cache directory
```bash
python cli.py openai/gpt-2 --input-shapes "1,1024" --hf-cache-dir ./hf_cache --output converted/
```

#### Large language model
```bash
python cli.py microsoft/DialoGPT-large --input-shapes "1,1024" --device gpu --dtype float16
```

### Download Only (No Conversion)

#### Download model to inspect before conversion
```bash
python cli.py microsoft/DialoGPT-medium --download-only --output downloads/
```

#### Download with custom cache
```bash
python cli.py bert-base-uncased --download-only --hf-cache-dir ./my_cache/
```

#### Download large model to verify it works
```bash
python cli.py facebook/opt-1.3b --download-only --verbose
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
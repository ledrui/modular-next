# CLI Usage Guide

## Basic Usage

Convert a PyTorch model to MAX format:

```bash
python cli.py /path/to/model.pt --input-shapes "1,784"
```

## Command Line Arguments

### Required Arguments

- `model_path`: Path to PyTorch model file (.pt, .pth, .safetensors)
- `--input-shapes`: Input tensor shapes in format 'batch,dim1,dim2,...'

### Optional Arguments

- `--output, -o`: Output directory (default: models)
- `--model-name`: Name for the converted model (default: derived from filename)
- `--device`: Target device - cpu, gpu, or auto (default: auto)
- `--dtype`: Target data type - float32, float16, int8, int32, int64 (default: float32)
- `--verbose, -v`: Enable verbose output

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

## Output

The CLI will:
1. Convert the PyTorch model to MAX format
2. Save model information to `{model_name}_info.txt` in the output directory
3. Print conversion status and device information

## Error Handling

Common errors and solutions:

- **File not found**: Check that the model path is correct
- **Unsupported format**: Only .pt, .pth, and .safetensors are supported
- **Invalid input shapes**: Use format 'batch,dim1,dim2,...' for single input or 'batch,dim1;batch,dim2' for multiple inputs
- **Module not found**: Ensure PyTorch and MAX are installed in your environment

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
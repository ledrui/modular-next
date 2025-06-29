#!/usr/bin/env python3
"""
Command-line interface for PyTorch to MAX converter.

Usage:
    python cli.py /path/to/model.pt --output /path/to/output/
    python cli.py /path/to/model.pth --output models/
    python cli.py microsoft/DialoGPT-medium --input-shapes "1,512" --output models/
    python cli.py https://huggingface.co/bert-base-uncased --input-shapes "1,512" --output models/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from converter import convert_from_checkpoint
from huggingface_utils import is_huggingface_url, extract_model_id_from_url, download_hf_model, get_model_info
from max.dtype import DType
from max.graph import DeviceRef


def parse_input_shapes(shapes_str: str) -> List[Tuple[int, ...]]:
    """
    Parse input shapes from string format.
    
    Examples:
        "1,784" -> [(1, 784)]
        "1,3,224,224" -> [(1, 3, 224, 224)]
        "1,784;1,10" -> [(1, 784), (1, 10)]
    """
    shapes = []
    for shape_str in shapes_str.split(';'):
        shape = tuple(int(x.strip()) for x in shape_str.split(','))
        shapes.append(shape)
    return shapes


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to MAX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py model.pt --input-shapes "1,784" --output models/
  python cli.py model.pth --input-shapes "1,3,224,224" --output converted/
  python cli.py checkpoint.pt --input-shapes "32,512" --device gpu --dtype float16
  python cli.py microsoft/DialoGPT-medium --input-shapes "1,512" --output models/
  python cli.py https://huggingface.co/bert-base-uncased --input-shapes "1,512" --output models/
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to PyTorch model file (.pt, .pth, .safetensors) or Hugging Face model URL/ID"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Output directory (default: models)"
    )
    
    parser.add_argument(
        "--input-shapes",
        type=str,
        required=True,
        help="Input tensor shapes in format 'batch,dim1,dim2,...' or 'batch,dim1;batch,dim2' for multiple inputs"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the converted model (default: derived from input filename)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Target device (default: auto)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "int8", "int32", "int64"],
        default="float32",
        help="Target data type (default: float32)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        help="Cache directory for Hugging Face models (default: system temp directory)"
    )
    
    args = parser.parse_args()
    
    # Handle Hugging Face URLs/IDs vs local files
    if is_huggingface_url(args.model_path):
        # Handle Hugging Face model
        model_id = extract_model_id_from_url(args.model_path)
        
        if args.verbose:
            print(f"Detected Hugging Face model: {model_id}")
            # Get model info if possible
            model_info = get_model_info(model_id)
            if "error" not in model_info:
                print(f"Model info: {model_info}")
            print()
        
        try:
            # Download the model
            print(f"Downloading model from Hugging Face: {model_id}")
            model_path = download_hf_model(model_id, cache_dir=args.hf_cache_dir)
            print(f"Model downloaded to: {model_path}")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            sys.exit(1)
    
    else:
        # Handle local file
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model file not found: {model_path}")
            sys.exit(1)
        
        if model_path.suffix not in ['.pt', '.pth', '.safetensors']:
            print(f"Error: Unsupported file format: {model_path.suffix}")
            print("Supported formats: .pt, .pth, .safetensors")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse input shapes
    try:
        input_shapes = parse_input_shapes(args.input_shapes)
    except ValueError as e:
        print(f"Error parsing input shapes: {e}")
        print("Format: 'batch,dim1,dim2,...' or 'batch,dim1;batch,dim2' for multiple inputs")
        sys.exit(1)
    
    # Set up device
    device = None
    if args.device == "cpu":
        device = DeviceRef.CPU()
    elif args.device == "gpu":
        device = DeviceRef.GPU()
    # auto will use the converter's auto-detection
    
    # Set up dtype
    dtype_map = {
        "float32": DType.float32,
        "float16": DType.float16,
        "int8": DType.int8,
        "int32": DType.int32,
        "int64": DType.int64,
    }
    dtype = dtype_map[args.dtype]
    
    # Generate model name
    if args.model_name:
        model_name = args.model_name
    elif is_huggingface_url(args.model_path):
        # For HF models, use the model ID as base name
        model_id = extract_model_id_from_url(args.model_path)
        model_name = model_id.replace('/', '_').replace('-', '_')
    else:
        model_name = model_path.stem
    
    if args.verbose:
        print(f"Converting model: {model_path}")
        print(f"Input shapes: {input_shapes}")
        print(f"Output directory: {output_dir}")
        print(f"Model name: {model_name}")
        print(f"Device: {args.device}")
        print(f"Data type: {args.dtype}")
        print()
    
    try:
        # Convert the model
        if args.verbose:
            print("Starting conversion...")
        
        max_model = convert_from_checkpoint(
            model_path=model_path,
            input_shapes=input_shapes,
            device=device,
            dtype=dtype,
            model_name=model_name
        )
        
        if args.verbose:
            print("✅ Model converted successfully!")
            print(f"Input devices: {max_model.input_devices}")
            print(f"Output devices: {max_model.output_devices}")
        
        # Save model info
        model_info_path = output_dir / f"{model_name}_info.txt"
        with open(model_info_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            if is_huggingface_url(args.model_path):
                model_id = extract_model_id_from_url(args.model_path)
                f.write(f"Hugging Face ID: {model_id}\n")
                f.write(f"Original Input: {args.model_path}\n")
                f.write(f"Downloaded to: {model_path}\n")
            else:
                f.write(f"Source: {model_path}\n")
            f.write(f"Input shapes: {input_shapes}\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Data type: {args.dtype}\n")
            f.write(f"Input devices: {max_model.input_devices}\n")
            f.write(f"Output devices: {max_model.output_devices}\n")
        
        print(f"✅ Model converted and saved to: {output_dir}")
        print(f"Model info saved to: {model_info_path}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
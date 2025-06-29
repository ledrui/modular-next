#!/usr/bin/env python3
"""
Command-line interface for PyTorch to MAX converter.

Usage:
    python cli.py convert /path/to/model.pt --input-shapes "1,784" --output models/
    python cli.py download microsoft/DialoGPT-medium --output downloads/
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


def cmd_convert(args):
    """Handle the convert subcommand."""
    # Validate local file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if model_path.suffix not in ['.pt', '.pth', '.safetensors', '.bin']:
        print(f"Error: Unsupported file format: {model_path.suffix}")
        print("Supported formats: .pt, .pth, .safetensors, .bin")
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
    model_name = args.model_name or model_path.stem
    
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


def cmd_download(args):
    """Handle the download subcommand."""
    # Validate it's a HF model
    if not is_huggingface_url(args.model_path):
        print("Error: download command only works with Hugging Face models")
        print("For local files, use the convert command directly")
        sys.exit(1)
    
    # Handle Hugging Face model
    model_id = extract_model_id_from_url(args.model_path)
    
    if args.verbose:
        print(f"Downloading Hugging Face model: {model_id}")
        # Get model info if possible
        model_info = get_model_info(model_id)
        if "error" not in model_info:
            print(f"Model info: {model_info}")
        print()
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download the model
        print(f"Downloading model from Hugging Face: {model_id}")
        model_path = download_hf_model(model_id, cache_dir=args.hf_cache_dir)
        print(f"✅ Model downloaded to: {model_path}")
        
        # If output directory specified, copy the model there
        if args.output:
            import shutil
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a clean filename from model_id
            clean_name = model_id.replace('/', '_').replace('-', '_')
            output_file = output_dir / f"{clean_name}{model_path.suffix}"
            
            print(f"Copying model to output directory: {output_file}")
            shutil.copy2(model_path, output_file)
            
            print(f"\nTo convert this model, run:")
            print(f"python cli.py convert {output_file} --input-shapes \"<your_shapes>\" --output converted/")
        else:
            print(f"\nTo convert this model, run:")
            print(f"python cli.py convert {model_path} --input-shapes \"<your_shapes>\" --output converted/")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch to MAX converter and Hugging Face downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert local model
  python cli.py convert model.pt --input-shapes "1,784" --output models/
  
  # Download HF model
  python cli.py download microsoft/DialoGPT-medium --output downloads/
  
  # Two-step workflow
  python cli.py download bert-base-uncased --output downloads/
  python cli.py convert downloads/model.safetensors --input-shapes "1,512" --output converted/
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True
    
    # Convert subcommand
    convert_parser = subparsers.add_parser('convert', help='Convert PyTorch model to MAX format')
    convert_parser.add_argument(
        "model_path",
        type=str,
        help="Path to PyTorch model file (.pt, .pth, .safetensors, .bin)"
    )
    convert_parser.add_argument(
        "--input-shapes",
        type=str,
        required=True,
        help="Input tensor shapes in format 'batch,dim1,dim2,...' or 'batch,dim1;batch,dim2' for multiple inputs"
    )
    convert_parser.add_argument(
        "--output", "-o",
        type=str,
        default="models",
        help="Output directory (default: models)"
    )
    convert_parser.add_argument(
        "--model-name",
        type=str,
        help="Name for the converted model (default: derived from filename)"
    )
    convert_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Target device (default: auto)"
    )
    convert_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "int8", "int32", "int64"],
        default="float32",
        help="Target data type (default: float32)"
    )
    convert_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    convert_parser.set_defaults(func=cmd_convert)
    
    # Download subcommand
    download_parser = subparsers.add_parser('download', help='Download model from Hugging Face')
    download_parser.add_argument(
        "model_path",
        type=str,
        help="Hugging Face model URL/ID (e.g., microsoft/DialoGPT-medium)"
    )
    download_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for downloaded model"
    )
    download_parser.add_argument(
        "--hf-cache-dir",
        type=str,
        help="Cache directory for Hugging Face models (default: system temp directory)"
    )
    download_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    download_parser.set_defaults(func=cmd_download)
    
    # Parse arguments and call appropriate function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Hugging Face integration utilities for PyTorch to MAX converter.

This module provides functionality to download and cache models from Hugging Face Hub.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse


def is_huggingface_url(model_path: str) -> bool:
    """
    Check if the model path is a Hugging Face URL or model identifier.
    
    Supports:
    - https://huggingface.co/model_name
    - huggingface.co/model_name  
    - model_name (simple identifier like "bert-base-uncased")
    - organization/model_name
    
    Args:
        model_path: The model path or URL to check
        
    Returns:
        True if it appears to be a Hugging Face reference
    """
    if not isinstance(model_path, str):
        return False
    
    # Check for explicit HF URLs
    if "huggingface.co" in model_path.lower():
        return True
    
    # Check for model identifiers (no file extension and contains forward slash or is a known pattern)
    path_obj = Path(model_path)
    
    # If it has a file extension, it's likely a local file
    if path_obj.suffix in ['.pt', '.pth', '.safetensors', '.bin']:
        return False
    
    # If it's an absolute or relative path to existing file, it's local
    if path_obj.exists() or path_obj.is_absolute() or str(path_obj).startswith('./'):
        return False
    
    # Check for HF model identifier patterns
    # Simple model names (e.g., "bert-base-uncased") or org/model (e.g., "microsoft/DialoGPT-medium")
    hf_pattern = re.compile(r'^[a-zA-Z0-9][\w\-\.]*(/[\w\-\.]+)*$')
    if hf_pattern.match(model_path) and ('/' in model_path or '-' in model_path):
        return True
    
    return False


def extract_model_id_from_url(url_or_id: str) -> str:
    """
    Extract the model ID from a Hugging Face URL or return the ID as-is.
    
    Args:
        url_or_id: HF URL or model ID
        
    Returns:
        Clean model ID (e.g., "microsoft/DialoGPT-medium")
        
    Examples:
        "https://huggingface.co/microsoft/DialoGPT-medium" -> "microsoft/DialoGPT-medium"
        "microsoft/DialoGPT-medium" -> "microsoft/DialoGPT-medium"
    """
    if "huggingface.co" in url_or_id:
        # Parse URL and extract model ID from path
        parsed = urlparse(url_or_id)
        path_parts = [p for p in parsed.path.split('/') if p]
        if len(path_parts) >= 2:
            return '/'.join(path_parts[-2:])  # org/model
        elif len(path_parts) == 1:
            return path_parts[0]  # simple model name
    
    return url_or_id


def download_hf_model(model_id: str, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model identifier (e.g., "microsoft/DialoGPT-medium")
        cache_dir: Optional cache directory. If None, uses system temp directory.
        
    Returns:
        Path to the downloaded model file
        
    Raises:
        ImportError: If huggingface_hub is not installed
        Exception: If download fails
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download models from Hugging Face. "
            "Install it with: pip install huggingface_hub"
        )
    
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "hf_models_cache"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # List available files in the repository
    try:
        repo_files = list_repo_files(model_id)
    except Exception as e:
        raise Exception(f"Failed to list files for model '{model_id}': {e}")
    
    # Look for PyTorch model files (in order of preference)
    model_file_candidates = [
        "pytorch_model.bin",
        "model.safetensors", 
        "pytorch_model.safetensors",
        "model.pt",
        "model.pth"
    ]
    
    model_file = None
    for candidate in model_file_candidates:
        if candidate in repo_files:
            model_file = candidate
            break
    
    if model_file is None:
        # Look for any .bin, .pt, .pth, or .safetensors file
        pytorch_files = [f for f in repo_files if f.endswith(('.bin', '.pt', '.pth', '.safetensors'))]
        if pytorch_files:
            model_file = pytorch_files[0]  # Take the first one
        else:
            raise Exception(f"No PyTorch model files found in repository '{model_id}'")
    
    print(f"Downloading {model_file} from {model_id}...")
    
    try:
        # Download the model file
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=model_file,
            cache_dir=str(cache_dir)
        )
        
        print(f"✅ Model downloaded to: {downloaded_path}")
        return Path(downloaded_path)
        
    except Exception as e:
        raise Exception(f"Failed to download model '{model_id}': {e}")


def get_model_info(model_id: str) -> dict:
    """
    Get information about a Hugging Face model.
    
    Args:
        model_id: Hugging Face model identifier
        
    Returns:
        Dictionary with model information
    """
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        return {
            "model_id": model_id,
            "tags": info.tags if hasattr(info, 'tags') else [],
            "pipeline_tag": info.pipeline_tag if hasattr(info, 'pipeline_tag') else None,
            "library_name": info.library_name if hasattr(info, 'library_name') else None,
        }
    except ImportError:
        return {"model_id": model_id, "error": "huggingface_hub not installed"}
    except Exception as e:
        return {"model_id": model_id, "error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Test URL/ID detection
    test_cases = [
        "https://huggingface.co/microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-medium", 
        "bert-base-uncased",
        "/path/to/model.pt",
        "./model.pth",
        "model.safetensors"
    ]
    
    print("Testing HF URL detection:")
    for case in test_cases:
        is_hf = is_huggingface_url(case)
        if is_hf:
            model_id = extract_model_id_from_url(case)
            print(f"  {case} -> HF: {is_hf}, ID: {model_id}")
        else:
            print(f"  {case} -> HF: {is_hf}")
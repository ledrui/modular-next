#!/usr/bin/env python3
"""
Quick test for PyTorch to MAX converter.
A simple script to quickly test the converter with a basic model.
"""

import numpy as np
import torch
import torch.nn as nn

from converter import PyTorchToMAXConverter
from max.driver import Tensor
from max.dtype import DType


def quick_test():
    """Quick test with a simple MLP model."""
    print("Quick PyTorch to MAX Converter Test")
    print("=" * 40)
    
    # Create a simple PyTorch model
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, 4)
    print(f"Input shape: {test_input.shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Convert to MAX
    print("\nConverting to MAX...")
    converter = PyTorchToMAXConverter()
    max_model = converter.convert_model(
        model, 
        input_shapes=[(1, 4)], 
        model_name="quick_test"
    )
    print("✅ Conversion successful!")
    
    # Run PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(test_input)
    pytorch_result = pytorch_output.numpy()
    print(f"PyTorch result: {pytorch_result.flatten()}")
    
    # Run MAX inference
    print("\nRunning MAX inference...")
    max_input = Tensor.from_numpy(test_input.numpy().astype(np.float32))
    
    # Move to correct device if needed
    if len(max_model.input_devices) > 0:
        max_input = max_input.to(max_model.input_devices[0])
    
    max_output = max_model.execute(max_input)
    max_result = max_output[0].to_numpy()
    print(f"MAX result: {max_result.flatten()}")
    
    # Compare results
    print("\nComparing results...")
    diff = np.abs(pytorch_result - max_result)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    tolerance = 1e-4
    if np.allclose(pytorch_result, max_result, rtol=tolerance, atol=tolerance):
        print(f"✅ Results match within tolerance ({tolerance})")
        print("🎉 Quick test PASSED!")
        return True
    else:
        print(f"❌ Results differ beyond tolerance ({tolerance})")
        print("⚠️ Quick test FAILED!")
        return False


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1) 
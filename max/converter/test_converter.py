#!/usr/bin/env python3
"""
Test script for PyTorch to MAX converter.

This script demonstrates how to:
1. Convert PyTorch models to MAX format
2. Run inference on both PyTorch and MAX models
3. Compare results for accuracy verification
4. Benchmark performance differences
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from converter import convert_pytorch_model, PyTorchToMAXConverter
from max.driver import Tensor, CPU
from max.dtype import DType
from max.graph import DeviceRef


def create_test_models():
    """Create various PyTorch models for testing."""
    
    # Simple MLP
    mlp = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Small CNN
    cnn = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Transformer-like model
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=128, num_heads=8, seq_len=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.layer_norm = nn.LayerNorm(embed_dim)
            self.linear = nn.Linear(embed_dim, vocab_size)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.layer_norm(x)
            # Simple pooling instead of full attention for this test
            x = x.mean(dim=1)  # Global average pooling
            x = self.linear(x)
            return x
    
    transformer = SimpleTransformer()
    
    # Simplified ResNet model
    class SimpleResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            # Initial convolution
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Simplified residual block (without skip connections for now)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(128)
            
            # Global average pooling and classifier
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)
            
        def forward(self, x):
            # Initial conv block
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            # Simplified residual blocks (without skip connections)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            
            # Global average pooling and classifier
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return x
    
    resnet = SimpleResNet()
    
    return {
        'mlp': (mlp, [(1, 784)]),
        'cnn': (cnn, [(1, 1, 28, 28)]),
        'transformer': (transformer, [(1, 32)]),  # batch_size=1, seq_len=32
        'resnet': (resnet, [(1, 3, 224, 224)]),  # batch_size=1, 3 channels, 224x224 image
    }


def test_model_conversion(model_name, pytorch_model, input_shapes, device=None, dtype=None):
    """Test conversion and inference for a single model."""
    
    print(f"\n{'='*50}")
    print(f"Testing {model_name.upper()} Model")
    print(f"{'='*50}")
    
    # Set model to evaluation mode
    pytorch_model.eval()
    
    # Create test input
    if model_name == 'transformer':
        # For transformer, create integer token inputs
        test_input = torch.randint(0, 1000, input_shapes[0])
        test_input_np = test_input.numpy()
    else:
        # For other models, create float inputs
        test_input = torch.randn(*input_shapes[0])
        test_input_np = test_input.numpy().astype(np.float32)
    
    print(f"Input shape: {input_shapes[0]}")
    print(f"Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    
    # Convert model to MAX
    try:
        print("\nConverting PyTorch model to MAX...")
        start_time = time.time()
        
        # Determine input dtypes - use int64 for transformer models with embedding
        input_dtypes = None
        if model_name == 'transformer':
            input_dtypes = [DType.int64]  # Transformer expects integer token IDs
        
        converter = PyTorchToMAXConverter(device=device, dtype=dtype)
        max_model = converter.convert_model(
            pytorch_model, 
            input_shapes, 
            model_name=f"{model_name}_converted",
            input_dtypes=input_dtypes
        )
        
        conversion_time = time.time() - start_time
        print(f"Conversion completed in {conversion_time:.3f}s")
        print(f"MAX model devices: {max_model.devices}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False
    
    # Run PyTorch inference
    try:
        print("\nRunning PyTorch inference...")
        with torch.no_grad():
            start_time = time.time()
            pytorch_output = pytorch_model(test_input)
            pytorch_time = time.time() - start_time
            
        pytorch_result = pytorch_output.numpy()
        print(f"PyTorch inference completed in {pytorch_time:.4f}s")
        print(f"PyTorch output shape: {pytorch_result.shape}")
        print(f"PyTorch output range: [{pytorch_result.min():.4f}, {pytorch_result.max():.4f}]")
        
    except Exception as e:
        print(f"❌ PyTorch inference failed: {e}")
        return False
    
    # Run MAX inference
    try:
        print("\nRunning MAX inference...")
        
        # Convert input to MAX tensor 
        if model_name == 'transformer':
            # For transformer, ensure input is int64
            test_input_for_max = test_input_np.astype(np.int64)
        else:
            test_input_for_max = test_input_np.astype(np.float32)
            
        if len(max_model.input_devices) > 0:
            target_device = max_model.input_devices[0]
            max_input = Tensor.from_numpy(test_input_for_max).to(target_device)
        else:
            max_input = Tensor.from_numpy(test_input_for_max)
        
        start_time = time.time()
        max_output = max_model.execute(max_input)
        max_time = time.time() - start_time
        
        # Extract result
        if isinstance(max_output, (list, tuple)):
            max_result = max_output[0].to_numpy()
        else:
            max_result = max_output.to_numpy()
            
        print(f"MAX inference completed in {max_time:.4f}s")
        print(f"MAX output shape: {max_result.shape}")
        print(f"MAX output range: [{max_result.min():.4f}, {max_result.max():.4f}]")
        
    except Exception as e:
        print(f"❌ MAX inference failed: {e}")
        return False
    
    # Compare results
    print("\nComparing results...")
    try:
        # Check shapes match
        if pytorch_result.shape != max_result.shape:
            print(f"❌ Shape mismatch: PyTorch {pytorch_result.shape} vs MAX {max_result.shape}")
            return False
        
        # Calculate differences
        abs_diff = np.abs(pytorch_result - max_result)
        rel_diff = abs_diff / (np.abs(pytorch_result) + 1e-8)
        
        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()
        
        print(f"Max absolute difference: {max_abs_diff:.6f}")
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"Max relative difference: {max_rel_diff:.6f}")
        print(f"Mean relative difference: {mean_rel_diff:.6f}")
        
        # Check if results are close
        tolerance = 1e-4
        if np.allclose(pytorch_result, max_result, rtol=tolerance, atol=tolerance):
            print(f"✅ Results match within tolerance ({tolerance})")
            success = True
        else:
            print(f"❌ Results differ beyond tolerance ({tolerance})")
            success = False
            
        # Performance comparison
        if pytorch_time > 0 and max_time > 0:
            speedup = pytorch_time / max_time
            print(f"\nPerformance comparison:")
            print(f"PyTorch: {pytorch_time*1000:.2f}ms")
            print(f"MAX: {max_time*1000:.2f}ms")
            print(f"Speedup: {speedup:.2f}x")
        
        return success
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False


def test_different_dtypes():
    """Test conversion with different data types."""
    print(f"\n{'='*50}")
    print("Testing Different Data Types")
    print(f"{'='*50}")
    
    # Simple model for dtype testing
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    model.eval()
    
    input_shapes = [(1, 10)]
    test_input = torch.randn(*input_shapes[0])
    
    dtypes_to_test = [
        (DType.float32, "float32"),
        # Add other dtypes as supported
    ]
    
    for dtype, dtype_name in dtypes_to_test:
        print(f"\nTesting with {dtype_name}...")
        try:
            converter = PyTorchToMAXConverter(dtype=dtype)
            max_model = converter.convert_model(
                model, 
                input_shapes, 
                model_name=f"dtype_test_{dtype_name}"
            )
            
            # Quick inference test
            max_input = Tensor.from_numpy(test_input.numpy().astype(np.float32))
            if len(max_model.input_devices) > 0:
                max_input = max_input.to(max_model.input_devices[0])
                
            result = max_model.execute(max_input)
            print(f"✅ {dtype_name} conversion and inference successful")
            
        except Exception as e:
            print(f"❌ {dtype_name} failed: {e}")


def test_device_placement():
    """Test conversion with different device placements."""
    print(f"\n{'='*50}")
    print("Testing Device Placement")
    print(f"{'='*50}")
    
    # Simple model for device testing
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    model.eval()
    
    input_shapes = [(1, 10)]
    test_input = torch.randn(*input_shapes[0])
    
    devices_to_test = [
        (DeviceRef.CPU(), "CPU"),
    ]
    
    # Add GPU if available
    try:
        from max.driver import accelerator_count
        if accelerator_count() > 0:
            devices_to_test.append((DeviceRef.GPU(), "GPU"))
    except:
        pass
    
    for device, device_name in devices_to_test:
        print(f"\nTesting with {device_name}...")
        try:
            converter = PyTorchToMAXConverter(device=device)
            max_model = converter.convert_model(
                model, 
                input_shapes, 
                model_name=f"device_test_{device_name}"
            )
            
            # Quick inference test
            max_input = Tensor.from_numpy(test_input.numpy().astype(np.float32))
            if len(max_model.input_devices) > 0:
                max_input = max_input.to(max_model.input_devices[0])
                
            result = max_model.execute(max_input)
            print(f"✅ {device_name} conversion and inference successful")
            print(f"   Model devices: {max_model.devices}")
            
        except Exception as e:
            print(f"❌ {device_name} failed: {e}")


def main():
    """Main test function."""
    print("PyTorch to MAX Converter Test Suite")
    print("==================================")
    
    # Get test models
    test_models = create_test_models()
    
    # Track results
    results = {}
    
    # Test each model
    for model_name, (pytorch_model, input_shapes) in test_models.items():
        success = test_model_conversion(
            model_name, 
            pytorch_model, 
            input_shapes,
            device=None,  # Auto-detect
            dtype=DType.float32
        )
        results[model_name] = success
    
    # Test different configurations
    test_different_dtypes()
    test_device_placement()
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name.upper():<15}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The converter is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Example usage scenarios for PyTorch to MAX converter.

This script demonstrates practical use cases:
1. Converting a pre-trained model
2. Saving and loading converted models
3. Batch inference
4. Performance benchmarking
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from converter import PyTorchToMAXConverter, convert_pytorch_model
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef


def example_1_simple_conversion():
    """Example 1: Simple model conversion and inference."""
    print("\n" + "="*60)
    print("Example 1: Simple Model Conversion")
    print("="*60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    model.eval()
    
    # Convert using the convenience function
    max_model = convert_pytorch_model(
        model=model,
        input_shapes=[(1, 10)],
        model_name="simple_example"
    )
    
    # Create test data
    test_input = np.random.randn(1, 10).astype(np.float32)
    
    # Run inference
    max_input = Tensor.from_numpy(test_input)
    if len(max_model.input_devices) > 0:
        max_input = max_input.to(max_model.input_devices[0])
    
    result = max_model.execute(max_input)
    print(f"✅ Inference successful! Output shape: {result[0].shape}")
    print(f"   Output: {result[0].to_numpy().flatten()[:5]}...")  # Show first 5 values


def example_2_batch_inference():
    """Example 2: Batch inference with multiple inputs."""
    print("\n" + "="*60)
    print("Example 2: Batch Inference")
    print("="*60)
    
    # Create a classification model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    model.eval()
    
    # Convert model
    converter = PyTorchToMAXConverter()
    max_model = converter.convert_model(
        model, 
        input_shapes=[(32, 784)],  # Batch size 32
        model_name="batch_classifier"
    )
    
    # Create batch of test data
    batch_size = 32
    test_batch = np.random.randn(batch_size, 784).astype(np.float32)
    
    print(f"Processing batch of {batch_size} samples...")
    
    # Run batch inference
    max_input = Tensor.from_numpy(test_batch)
    if len(max_model.input_devices) > 0:
        max_input = max_input.to(max_model.input_devices[0])
    
    start_time = time.time()
    result = max_model.execute(max_input)
    inference_time = time.time() - start_time
    
    predictions = result[0].to_numpy()
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"✅ Batch inference completed in {inference_time*1000:.2f}ms")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Predicted classes: {predicted_classes[:10]}...")  # Show first 10
    print(f"   Throughput: {batch_size/inference_time:.1f} samples/second")


def example_3_performance_comparison():
    """Example 3: Performance comparison between PyTorch and MAX."""
    print("\n" + "="*60)
    print("Example 3: Performance Comparison")
    print("="*60)
    
    # Create a moderately complex model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    model.eval()
    
    # Convert to MAX
    converter = PyTorchToMAXConverter()
    max_model = converter.convert_model(
        model,
        input_shapes=[(100, 512)],  # Batch of 100
        model_name="performance_test"
    )
    
    # Prepare test data
    test_data = torch.randn(100, 512)
    test_data_np = test_data.numpy().astype(np.float32)
    
    # Warm-up runs
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(test_data)
        
        max_input = Tensor.from_numpy(test_data_np)
        if len(max_model.input_devices) > 0:
            max_input = max_input.to(max_model.input_devices[0])
        _ = max_model.execute(max_input)
    
    # Benchmark PyTorch
    print("Benchmarking PyTorch...")
    pytorch_times = []
    for _ in range(10):
        start_time = time.time()
        with torch.no_grad():
            pytorch_result = model(test_data)
        pytorch_times.append(time.time() - start_time)
    
    pytorch_avg_time = np.mean(pytorch_times)
    pytorch_std_time = np.std(pytorch_times)
    
    # Benchmark MAX
    print("Benchmarking MAX...")
    max_times = []
    for _ in range(10):
        max_input = Tensor.from_numpy(test_data_np)
        if len(max_model.input_devices) > 0:
            max_input = max_input.to(max_model.input_devices[0])
        
        start_time = time.time()
        max_result = max_model.execute(max_input)
        max_times.append(time.time() - start_time)
    
    max_avg_time = np.mean(max_times)
    max_std_time = np.std(max_times)
    
    # Compare results for accuracy
    max_result_np = max_result[0].to_numpy()
    pytorch_result_np = pytorch_result.numpy()
    
    # Results
    print(f"\nPerformance Results:")
    print(f"PyTorch: {pytorch_avg_time*1000:.2f} ± {pytorch_std_time*1000:.2f} ms")
    print(f"MAX:     {max_avg_time*1000:.2f} ± {max_std_time*1000:.2f} ms")
    
    if max_avg_time > 0:
        speedup = pytorch_avg_time / max_avg_time
        print(f"Speedup: {speedup:.2f}x")
    
    # Accuracy check
    max_diff = np.abs(pytorch_result_np - max_result_np).max()
    print(f"Max difference: {max_diff:.8f}")
    
    if np.allclose(pytorch_result_np, max_result_np, rtol=1e-4, atol=1e-4):
        print("✅ Results are numerically equivalent")
    else:
        print("⚠️ Results differ beyond tolerance")


def example_4_different_models():
    """Example 4: Converting different types of models."""
    print("\n" + "="*60)
    print("Example 4: Different Model Types")
    print("="*60)
    
    models_to_test = [
        ("Linear only", nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1)), [(1, 10)]),
        ("With ReLU", nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1)), [(1, 10)]),
        ("With LayerNorm", nn.Sequential(nn.Linear(10, 5), nn.LayerNorm(5), nn.Linear(5, 1)), [(1, 10)]),
        ("Embedding", nn.Sequential(nn.Embedding(100, 16), nn.Linear(16, 1)), [(1, 10)]),
    ]
    
    for model_name, model, input_shapes in models_to_test:
        print(f"\nTesting: {model_name}")
        model.eval()
        
        try:
            converter = PyTorchToMAXConverter()
            max_model = converter.convert_model(
                model,
                input_shapes,
                model_name=f"test_{model_name.replace(' ', '_').lower()}"
            )
            
            # Quick inference test
            if "Embedding" in model_name:
                test_input = np.random.randint(0, 100, input_shapes[0], dtype=np.int32)
            else:
                test_input = np.random.randn(*input_shapes[0]).astype(np.float32)
            
            max_input = Tensor.from_numpy(test_input)
            if len(max_model.input_devices) > 0:
                max_input = max_input.to(max_model.input_devices[0])
            
            result = max_model.execute(max_input)
            print(f"  ✅ Success! Output shape: {result[0].shape}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def main():
    """Run all examples."""
    print("PyTorch to MAX Converter - Example Usage")
    print("=" * 60)
    
    try:
        example_1_simple_conversion()
        example_2_batch_inference()
        example_3_performance_comparison()
        example_4_different_models()
        
        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
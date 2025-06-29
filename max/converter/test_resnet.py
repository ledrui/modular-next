#!/usr/bin/env python3
"""
Quick ResNet conversion test.
"""

import torch
import torch.nn as nn
import numpy as np

from converter import PyTorchToMAXConverter
from max.driver import Tensor
from max.dtype import DType

def create_simple_resnet():
    """Create a simple ResNet-like model for testing."""
    
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
    
    return SimpleResNet()

def test_resnet_conversion():
    """Test ResNet conversion."""
    print("ResNet Conversion Test")
    print("=" * 40)
    
    # Create model
    model = create_simple_resnet()
    model.eval()
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create test input
    input_shape = (1, 3, 224, 224)
    test_input = torch.randn(*input_shape)
    
    print(f"Input shape: {input_shape}")
    
    # Test PyTorch inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(test_input)
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"PyTorch output range: [{pytorch_output.min():.4f}, {pytorch_output.max():.4f}]")
    
    # Convert to MAX
    print("\nConverting to MAX...")
    try:
        converter = PyTorchToMAXConverter()
        max_model = converter.convert_model(
            model,
            input_shapes=[input_shape],
            model_name="simple_resnet"
        )
        print("✅ Conversion successful!")
        print(f"MAX model devices: {max_model.devices}")
        
        # Test MAX inference
        print("\nRunning MAX inference...")
        max_input = Tensor.from_numpy(test_input.numpy().astype(np.float32))
        if len(max_model.input_devices) > 0:
            max_input = max_input.to(max_model.input_devices[0])
        
        max_output = max_model.execute(max_input)
        
        if isinstance(max_output, (list, tuple)):
            max_result = max_output[0].to_numpy()
        else:
            max_result = max_output.to_numpy()
        
        print(f"MAX output shape: {max_result.shape}")
        print(f"MAX output range: [{max_result.min():.4f}, {max_result.max():.4f}]")
        
        # Compare results
        pytorch_result = pytorch_output.numpy()
        if pytorch_result.shape == max_result.shape:
            abs_diff = np.abs(pytorch_result - max_result)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            
            print(f"\nComparison:")
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
            
            if np.allclose(pytorch_result, max_result, rtol=1e-4, atol=1e-4):
                print("✅ Results match within tolerance!")
                return True
            else:
                print("⚠️  Results differ beyond tolerance")
                return False
        else:
            print(f"❌ Shape mismatch: PyTorch {pytorch_result.shape} vs MAX {max_result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Conversion or inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_resnet_conversion()
    if success:
        print("\n🎉 ResNet test passed!")
    else:
        print("\n💥 ResNet test failed!")
    
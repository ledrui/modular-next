#!/usr/bin/env python3
"""
Simple test script to create a minimal Llama-like model and test conversion.
Run this with: pixi run python simple_llama_test.py
"""

import torch
import torch.nn as nn


class SimpleLlamaTest(nn.Module):
    """Minimal Llama-like model for testing converter."""
    
    def __init__(self, vocab_size=100, hidden_size=64, num_layers=1):
        super().__init__()
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # Simple RMSNorm-like layer (we'll use LayerNorm as fallback for now)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # MLP with SiLU (similar to Llama's SwiGLU)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.activation = nn.SiLU()
        
        # Output norm and projection
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        # Embedding
        x = self.embed_tokens(input_ids)
        
        # Add residual connection
        residual = x
        
        # Pre-MLP norm
        x = self.input_norm(x)
        
        # SwiGLU-like MLP
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up)
        
        # Residual connection
        x = residual + mlp_out
        
        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


def create_and_save_test_model():
    """Create and save a simple test model."""
    print("Creating simple Llama-like test model...")
    
    # Create small model
    model = SimpleLlamaTest(vocab_size=100, hidden_size=64, num_layers=1)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Test the model
    test_input = torch.randint(0, 100, (1, 8))  # batch=1, seq_len=8
    print(f"Test input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Test output shape: {output.shape}")
    
    # Save the model
    model_path = "simple_llama_test.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model_path


if __name__ == "__main__":
    model_path = create_and_save_test_model()
    print(f"\nTo test conversion, run:")
    print(f"python cli.py convert {model_path} --input-shapes \"1,8\" --output converted/ --verbose")
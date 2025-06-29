#!/usr/bin/env python3
"""
Create a sample Llama-like model for testing the converter.
This creates a minimal but architecturally correct Llama model.
"""

import torch
import torch.nn as nn
import math
from pathlib import Path


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Llama)."""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    """Llama MLP with SwiGLU activation."""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaAttention(nn.Module):
    """Simplified Llama attention (without RoPE for testing)."""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer."""
    
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.self_attn = LlamaAttention(hidden_size, num_heads)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, hidden_states):
        # Pre-attention normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # Pre-MLP normalization
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LlamaModel(nn.Module):
    """Core Llama model."""
    
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, max_position_embeddings=2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hidden_size, intermediate_size, num_attention_heads)
            for _ in range(num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size)

    def forward(self, input_ids):
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """Llama model for causal language modeling."""
    
    def __init__(self, vocab_size=32000, hidden_size=512, intermediate_size=1376, num_hidden_layers=4, num_attention_heads=8):
        super().__init__()
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits


def create_sample_llama_model():
    """Create a small Llama model for testing."""
    print("Creating sample Llama model...")
    
    # Small model for testing
    model = LlamaForCausalLM(
        vocab_size=1000,      # Small vocab for testing
        hidden_size=256,      # Small hidden size
        intermediate_size=688, # 256 * 8/3 ≈ 688 (typical Llama ratio)
        num_hidden_layers=2,   # Just 2 layers for testing
        num_attention_heads=4  # 256 / 4 = 64 head_dim
    )
    
    # Initialize with reasonable values
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def save_model(model, path="sample_llama.pt"):
    """Save the model as a .pt file."""
    print(f"Saving model to {path}...")
    
    # Save the model state dict
    torch.save(model.state_dict(), path)
    
    print(f"✅ Model saved to {path}")
    print(f"File size: {Path(path).stat().st_size / (1024*1024):.1f} MB")
    
    return path


def test_model(model, path):
    """Test that the saved model can be loaded and run."""
    print("Testing model...")
    
    # Test with sample input
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(input_ids)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0, 0, :5].tolist()}")
    
    # Test loading from file
    print("Testing model loading...")
    new_model = LlamaForCausalLM(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=688,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    new_model.load_state_dict(torch.load(path))
    new_model.eval()
    
    with torch.no_grad():
        output2 = new_model(input_ids)
        
    # Check outputs match
    if torch.allclose(output, output2, rtol=1e-5):
        print("✅ Model loading test passed!")
    else:
        print("❌ Model loading test failed!")
    
    return True


def main():
    """Create and save a sample Llama model."""
    print("Creating Sample Llama Model for Converter Testing")
    print("=" * 50)
    
    # Create model
    model = create_sample_llama_model()
    
    # Save model
    model_path = save_model(model, "sample_llama.pt")
    
    # Test model
    test_model(model, model_path)
    
    print("\n" + "=" * 50)
    print("Sample model created successfully!")
    print(f"Use: python cli.py convert {model_path} --input-shapes \"1,10\" --output converted/")
    print("(Input shape: batch=1, sequence_length=10)")


if __name__ == "__main__":
    main()
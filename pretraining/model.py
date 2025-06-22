"""
Small transformer model for pretraining testing.
Based on the 2B model specifications from Eldan et al. (2024) but smaller for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to input tensor."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to tensor."""
    # Split last dimension into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    rotated = torch.cat([-x2, x1], dim=-1)
    
    # Combine with cosine and sine
    return x * cos + rotated * sin

class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.rope = RoPEPositionalEncoding(self.head_dim, max_seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            # Create causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(scores.device)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Transpose back and reshape
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Final projection
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, dim: int, num_heads: int, intermediate_size: int, 
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        
        self.attention = MultiHeadAttention(dim, num_heads, max_seq_len)
        self.feed_forward = SwiGLU(dim, intermediate_size)
        
        self.attention_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.attention_norm(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.ff_norm(x))
        x = x + self.dropout(ff_out)
        
        return x

class SmallTransformer(nn.Module):
    """Small transformer model for pretraining testing."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings (untied from output layer as per research)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
                max_seq_len=config.max_seq_length
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        # Output head (untied from input embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following standard practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100, 
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Simple generation function for testing."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get last token logits
                last_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(last_logits, top_k)
                    last_logits = torch.full_like(last_logits, float('-inf'))
                    last_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we hit EOS token (assume token 2 is EOS)
                if next_token.item() == 2:
                    break
        
        return input_ids
    
    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())

def create_model(config) -> SmallTransformer:
    """Create model with given configuration."""
    model = SmallTransformer(config)
    
    num_params = model.get_num_params()
    print(f"Created model with {num_params:,} parameters")
    
    return model

def main():
    """Test model creation and forward pass."""
    from .config import get_toy_dataset_config
    
    config = get_toy_dataset_config()
    model = create_model(config.model)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    prompt = torch.randint(0, config.model.vocab_size, (1, 10))
    generated = model.generate(prompt, max_length=50)
    print(f"Generated sequence length: {generated.shape[1]}")

if __name__ == "__main__":
    main()
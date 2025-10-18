from typing import Optional
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Plain Transformer block (PyTorch native MHA).
    - Uses key_padding_mask (True = mask/pad) for clarity.
    """
    def __init__(self, hidden_size: int, num_heads: int, ff_mult: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_mult * hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_mult * hidden_size, hidden_size),
            nn.Dropout(0.1),
        )
        self.res_scale = 0.5 # Residual scaling nhỏ
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.ln1(x)
        a, _ = self.attn(qkv, qkv, qkv, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.res_scale * a
        x = x + self.res_scale * self.ff(self.ln2(x))
        return x


class GatedFusion(nn.Module):
    """Token‑wise gated fusion giữa hidden_states và injected.
    gate = sigmoid(Wx * x + Wi * injected + b)
    out  = x + gate * injected
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.wx = nn.Linear(hidden_size, hidden_size, bias=True)
        self.wi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sig = nn.Sigmoid()
        self.mix = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, injected: torch.Tensor) -> torch.Tensor:
        gate = self.sig(self.wx(x) + self.wi(injected))
        # return x + gate * injected
        return self.mix(torch.cat([x, x + gate*injected], dim=-1))


class ReasoningModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_mult: int, depth: int):
        super().__init__()
        self.fuse = GatedFusion(hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_mult) for _ in range(depth)
        ])

    def forward(self, hidden_states: torch.Tensor, injected: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.fuse(hidden_states, injected)
        for blk in self.layers:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .config import HRMCoreConfig
from .blocks import ReasoningModule


class HRMCoreInner(nn.Module):
    """Two-level hierarchical reasoning core (H supervises L) với soft‑halting.

    Soft‑halting: ở mỗi vòng H, tính gate h_t \in (0,1). Tích lũy phân phối dừng và
    xuất ra tổ hợp có trọng số của các trạng thái H_t. Không cần RL, end‑to‑end được.
    """
    def __init__(self, cfg: HRMCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.H_level = ReasoningModule(cfg.hidden_size, cfg.num_heads, cfg.ff_mult, cfg.H_layers)
        self.L_level = ReasoningModule(cfg.hidden_size, cfg.num_heads, cfg.ff_mult, cfg.L_layers)

        # Learnable initial latent states
        self.H_init = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.L_init = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))

        # Halting head (trên pooled H)
        self.halt_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        self.epsilon = 0.01  # ngưỡng còn lại

    @staticmethod
    def _pool(x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean pooling theo chiều sequence, bỏ pad (True=pad)."""
        if key_padding_mask is None:
            return x.mean(dim=1)
        mask = (~key_padding_mask).float()  # [B,S]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        token_embeddings: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        approx_one_step: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về (H_out, L_last) với H_out là tổ hợp soft‑halting của các bước H.
        Shapes:
          z_H, z_L: [B, S, D]
          token_embeddings: [B, S, D]
          key_padding_mask: [B, S] (True=PAD)
        """
        x_H, x_L = z_H, z_L
        B, S, D = x_H.shape
        device = x_H.device

        remaining = torch.ones(B, 1, device=device)  # khối lượng còn lại để phân bổ
        H_accum = torch.zeros_like(x_H)              # tổ hợp weighted của H_t theo token

        for _ in range(self.cfg.H_cycles):
            # L có thể lặp nhiều vòng nhỏ dưới sự điều khiển của H hiện tại
            if approx_one_step:
            # One-step gradient approximation: không lan truyền gradient qua inner loop
                with torch.no_grad():
                    for _ in range(self.cfg.L_cycles):
                        x_L = self.L_level(x_L, injected=x_H + token_embeddings, key_padding_mask=key_padding_mask)
            else:
            # Full backprop: gradient đầy đủ qua H–L
                for _ in range(self.cfg.L_cycles):
                    x_L = self.L_level(x_L, injected=x_H + token_embeddings, key_padding_mask=key_padding_mask)

            # H cập nhật dựa trên L (đã phản ánh input)
            x_H = self.H_level(x_H, injected=x_L, key_padding_mask=key_padding_mask)

            # Tính soft‑halt gate theo pooled H
            pooled = self._pool(x_H, key_padding_mask)  # [B, D]
            halt = self.halt_head(pooled)               # [B, 1] trong (0,1)

            # Phân bổ phần dừng của vòng này, soft nên dùng tỷ lệ còn lại
            allocated = halt * remaining                # [B,1]
            H_accum = H_accum + allocated.unsqueeze(1) * x_H
            remaining = remaining * (1.0 - halt)

        # Nếu còn dư khối lượng (do epsilon), đổ hết vào trạng thái H cuối
        if self.training:
            # Trong training giữ gradient mượt
            H_out = H_accum + remaining.unsqueeze(1) * x_H
        else:
            # Trong eval, ép dồn phần còn lại nếu quá nhỏ
            H_out = H_accum + (remaining.clamp_max(self.epsilon)).unsqueeze(1) * x_H

        return H_out, x_L
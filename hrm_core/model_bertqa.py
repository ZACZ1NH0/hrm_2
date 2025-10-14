from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForQuestionAnswering
from .config import HRMCoreConfig
from .inner import HRMCoreInner


class HRMBertForQA(nn.Module):
    """BERT encoder + (BERT QA head ⊕ HRM QA head) with blend.
    - alpha: trọng số pha trộn (0..1). alpha=1 → chỉ HRM; alpha=0 → chỉ BERT QA head.
    - Có thể fine‑tune toàn bộ hoặc freeze một phần encoder.
    """
    def __init__(self, cfg: HRMCoreConfig, encoder_name: str = 'bert-base-uncased', alpha: float = 0.5, freeze_encoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.alpha = alpha

        # ===== BERT encoder & QA head gốc =====
        self.enc_cfg = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.bert_qa = AutoModelForQuestionAnswering.from_pretrained(encoder_name).qa_outputs  # nn.Linear(H,2)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ===== HRM core + QA head riêng =====
        self.ln_in = nn.LayerNorm(self.enc_cfg.hidden_size)
        self.inner = HRMCoreInner(cfg)
        self.hrm_qa = nn.Linear(self.enc_cfg.hidden_size, 2)

        # Optional: init HRM head từ BERT head để ổn định sớm
        with torch.no_grad():
            self.hrm_qa.weight.copy_(self.bert_qa.weight)
            self.hrm_qa.bias.copy_(self.bert_qa.bias)

    def init_states(self, batch: int, seq_len: int, device: torch.device):
        H0 = self.inner.H_init.expand(batch, seq_len, -1).to(device)
        L0 = self.inner.L_init.expand(batch, seq_len, -1).to(device)
        return H0, L0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = self.ln_in(enc.last_hidden_state)  # [B,S,H]

        B, S, _ = x.shape
        device = x.device
        z_H, z_L = self.init_states(B, S, device)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # HRM reasoning → logits_h
        z_H_out, _ = self.inner(z_H, z_L, token_embeddings=x, key_padding_mask=key_padding_mask)
        logits_h = self.hrm_qa(z_H_out)  # [B,S,2]

        # BERT QA head trực tiếp trên encoder → logits_b
        logits_b = self.bert_qa(x)       # [B,S,2]

        # Blend logits
        logits = self.alpha * logits_h + (1.0 - self.alpha) * logits_b
        start_logits, end_logits = logits[..., 0], logits[..., 1]

        out = {"start_logits": start_logits, "end_logits": end_logits}

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            out["loss"] = (start_loss + end_loss) / 2
        return out
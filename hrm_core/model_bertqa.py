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
        self.alpha = nn.Parameter(torch.tensor(alpha))

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

        self.sf_head = nn.Linear(cfg.hidden_size, 1) # 20/3/2026
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
        sf_mask: Optional[torch.Tensor] = None, # 20/3/2026
    ) -> Dict[str, torch.Tensor]:
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = self.ln_in(enc.last_hidden_state)  # [B,S,H]

        B, S, _ = x.shape
        device = x.device
        z_H, z_L = self.init_states(B, S, device)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # HRM reasoning → logits_h
        z_H_out, _ = self.inner(z_H, z_L, token_embeddings=x, key_padding_mask=key_padding_mask)
        sf_logits = self.sf_head(z_H_out).squeeze(-1)

        sf_prob = torch.sigmoid(sf_logits).unsqueeze(-1) # [B, S, 1]
        z_H_focused = z_H_out * (1 + sf_prob)

        logits_h = self.hrm_qa(z_H_focused)  # [B,S,2]

        # BERT QA head trực tiếp trên encoder → logits_b
        logits_b = self.bert_qa(x)       # [B,S,2]

        # Blend logits
        
        a = torch.sigmoid(self.alpha)
        logits = a * logits_h + (1.0 - a) * logits_b
        start_logits, end_logits = logits[..., 0], logits[..., 1]

        out = {
            "start_logits": start_logits, 
            "end_logits": end_logits, 
            "sf_logits": sf_logits # [B, S]
        }
        #loss

        

        # if start_positions is not None and end_positions is not None:
        #     ignored_index = start_logits.size(1)
        #     start_positions = start_positions.clamp(0, ignored_index)
        #     end_positions = end_positions.clamp(0, ignored_index)
        #     loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        #     start_loss = loss_fct(start_logits, start_positions)
        #     end_loss = loss_fct(end_logits, end_positions)
        #     out["loss"] = (start_loss + end_loss) / 2



        # Tính Loss
        if start_positions is not None and end_positions is not None:
            # 1. QA Loss (đã có)
            ignored_index = start_logits.size(1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            qa_loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2
        
            # 2. SF Loss (Bổ sung mới)
            total_loss = qa_loss
            if sf_mask is not None:
                # Dùng BCEWithLogitsLoss vì đây là bài toán multilabel (mỗi token có thể là SF hoặc không)
                pos_weight = torch.tensor([20.0], device=device)
                sf_loss_fct = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
                # Chỉ tính loss trên các token không phải PAD
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = sf_logits.view(-1)[active_loss]
                    active_labels = sf_mask.view(-1)[active_loss].float()
                    sf_loss = sf_loss_fct(active_logits, active_labels)
                else:
                    sf_loss = sf_loss_fct(sf_logits, sf_mask.float())

                # Mix loss: Bạn có thể điều chỉnh hệ số 1.0 tùy theo độ quan trọng
                total_loss = qa_loss + self.cfg.sf_loss_coef * sf_loss
                out["sf_loss"] = sf_loss
            
            out["loss"] = total_loss
        return out
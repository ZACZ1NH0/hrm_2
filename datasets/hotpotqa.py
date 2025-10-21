import json
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

@dataclass
class QAExample:
    id: str
    question: str
    context: str
    answer_text: str
    answer_start: int

def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)

class HotpotQADataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512, doc_stride: int = 128, is_train: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.is_train = is_train
        self.features = []

        raw = list(load_jsonl(path))
        for ex in raw:
            qa = {
                "id": str(ex.get("id", "")),
                "question": ex["question"],
                "context": ex["context"],
                "answer_text": ex.get("answer_text") or ex.get("answer"),
                "answer_start": int(ex.get("answer_start")),
            }
            sf_char_spans = ex.get("sf_char_spans")  # ví dụ: [[s1,e1], [s2,e2], ...] theo char trong context
            if sf_char_spans is None:
                sf_char_spans = _build_sf_char_spans(ex)
            enc = self.tokenizer(
                qa["question"], qa["context"],
                return_offsets_mapping=True,
                padding="max_length",
                truncation="only_second",
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            offset_mapping = enc["offset_mapping"]
            overflow_to_sample = enc["overflow_to_sample_mapping"]

            for i in range(input_ids.size(0)):
                offsets = offset_mapping[i]
                seq_ids = enc.sequence_ids(i)
                context_mask = torch.tensor([1 if sid == 1 else 0 for sid in seq_ids], dtype=torch.long)

                # --- map answer char -> token span ---
                start_char = qa["answer_start"]
                end_char = start_char + len(qa["answer_text"])
                S = input_ids.size(1)
                start_pos = end_pos = S
                for ti, (sid, (st, ed)) in enumerate(zip(seq_ids, offsets.tolist())):
                    if sid != 1:
                        continue
                    if st <= start_char < ed and start_pos == S:
                        start_pos = ti
                    if st < end_char <= ed and end_pos == S:
                        end_pos = ti
                    if start_pos != S and end_pos != S:
                        break
                if start_pos == S or end_pos == S:
                    s = e = None
                    for ti, (sid, (st, ed)) in enumerate(zip(seq_ids, offsets.tolist())):
                        if sid != 1: continue
                        if st >= start_char and ed <= end_char:
                            if s is None: s = ti
                            e = ti
                    if s is not None and e is not None:
                        start_pos, end_pos = s, e

                # --- optional supporting‑fact mask ---
                sf_mask = None
                if sf_char_spans:
                    sf_mask = torch.zeros(S, dtype=torch.long)
                    for (cs, ce) in sf_char_spans:
                        for ti, (sid, (st, ed)) in enumerate(zip(seq_ids, offsets.tolist())):
                            if sid != 1:
                                continue
                            if not (ed <= cs or st >= ce):
                                sf_mask[ti] = 1

                feat = {
                    "input_ids": input_ids[i],
                    "attention_mask": attention_mask[i],
                    "id": qa["id"],
                    "answer_text": qa["answer_text"],
                    "context_mask": context_mask,
                }
                if sf_mask is not None:
                    feat["sf_mask"] = sf_mask
                if self.is_train:
                    feat["start_positions"] = torch.tensor(start_pos, dtype=torch.long)
                    feat["end_positions"] = torch.tensor(end_pos, dtype=torch.long)
                self.features.append(feat)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_train(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),
        "start_positions": torch.stack([b["start_positions"] for b in batch]),
        "end_positions": torch.stack([b["end_positions"] for b in batch]),
    }
    if "sf_mask" in batch[0]:
        out["sf_mask"] = torch.stack([b["sf_mask"] for b in batch])
    return out


def collate_eval(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "context_mask": torch.stack([b["context_mask"] for b in batch]),
        "id": [b["id"] for b in batch],
        "answer_text": [b["answer_text"] for b in batch],
    }
    if "sf_mask" in batch[0]:
        out["sf_mask"] = torch.stack([b["sf_mask"] for b in batch])
    if "start_positions" in batch[0]:
        out["start_positions"] = torch.stack([b["start_positions"] for b in batch])
        out["end_positions"] = torch.stack([b["end_positions"] for b in batch])
    return out

def _build_sf_char_spans(ex):
    sf_pairs = ex.get("supporting_facts") or []
    context = ex.get("context")
    spans = []

    # context của HotpotQA: list [[title, [sent1, sent2,...]], ...]
    if isinstance(context, list) and len(context) > 0 and isinstance(context[0], list):
        char_cursor = 0
        for title, sents in context:
            for i, sent in enumerate(sents):
                st, ed = char_cursor, char_cursor + len(sent)
                # nếu (title, i) thuộc supporting_facts -> thêm vào spans
                if [title, i] in sf_pairs:
                    spans.append([st, ed])
                char_cursor = ed + 1  # +1 cho dấu ngắt
        return spans
    return None
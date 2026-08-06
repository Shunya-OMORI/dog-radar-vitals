"""介入テキスト d を固定次元ベクトルに埋め込む。

multilingual-e5-large (Wang et al., 2024, "Multilingual E5 Text Embeddings: A Technical
Report") を使う。日本語を含む多言語で強く、かつこのマシンに既にキャッシュ済み
(~/.cache/huggingface/hub/models--intfloat--multilingual-e5-large)のため追加ダウンロード
不要という実務的な理由も込みで選択。E5系は "query: " / "passage: " のプレフィックスを
付けることが公式に推奨されており(検索を意識した対照学習で事前学習されているため)、
ここでは d を検索対象として "passage: " を使う(S2の介入検索シナリオを想定しているため)。
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_ID = "intfloat/multilingual-e5-large"
EMBED_DIM = 1024


@lru_cache(maxsize=1)
def _load_embedder(device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    model = AutoModel.from_pretrained(EMBED_MODEL_ID).to(device)
    model.eval()
    return tokenizer, model


def _average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.inference_mode()
def embed_texts(texts: list[str], device: str = "cuda:0", is_query: bool = False) -> torch.Tensor:
    """d のリストを (N, EMBED_DIM) のL2正規化済みベクトルにする。"""
    prefix = "query: " if is_query else "passage: "
    tokenizer, model = _load_embedder(device)
    batch = tokenizer(
        [prefix + t for t in texts],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    out = model(**batch)
    emb = _average_pool(out.last_hidden_state, batch["attention_mask"])
    return F.normalize(emb, p=2, dim=1)

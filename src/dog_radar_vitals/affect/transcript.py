"""音声から発話内容をテキスト化する(d = VLM(Transcript(音声), 画像) の Transcript 部分)。

Whisper large-v3 (Radford et al., 2022, "Robust Speech Recognition via
Large-Scale Weak Supervision") をそのまま使う。この部分に新規性はなく、
確立済みのASRを"介入の説明"を作るための前段として利用するだけなので、
新しいモデル設計はしていない(R1 の対象は intervention.py 側)。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import Pipeline, pipeline

WHISPER_MODEL_ID = "openai/whisper-large-v3"


@lru_cache(maxsize=1)
def _load_asr_pipeline(device: str = "cuda:0") -> Pipeline:
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        dtype=torch.float16,
        device=device,
    )


def transcribe(
    audio: np.ndarray | str | Path,
    sampling_rate: int | None = None,
    language: str | None = "japanese",
    device: str = "cuda:0",
) -> str:
    """音声(numpy配列 or ファイルパス)をテキストに変換する。

    無音・雑音区間(介入が無い安静区間)では空文字に近い出力になりうる。
    これは仕様であり、d の生成側で「発話なし」を明示的に扱う(II-4の対照区間の要件)。
    """
    asr = _load_asr_pipeline(device)
    kwargs = {"generate_kwargs": {"language": language}} if language else {}
    if isinstance(audio, np.ndarray):
        if sampling_rate is None:
            raise ValueError("audioがnp.ndarrayの場合はsampling_rateが必須")
        result = asr({"array": audio.astype(np.float32), "sampling_rate": sampling_rate}, **kwargs)
    else:
        result = asr(str(audio), **kwargs)
    return result["text"].strip()

"""d = VLM(Transcript(音声), 画像): 「ヒトがイヌに対して行ったはたらきかけ」をテキスト化する。

進捗報告書 Part II-1 の構想の中核部分。VLM に画像(複数フレーム可)と音声の書き起こしを渡し、
「何が起きたか」を一文〜数文のテキストに要約させる。このテキスト d を、別途 RR 間隔由来で
計算する状態変化 Δs=(ΔArousal) と対応づけて学習するのが後続の association_model.py の役割。

## 設計判断とその理由

1. **VLM は google/gemma-3-4b-it を 4bit 量子化(bitsandbytes)で使う。**
   このマシンは RTX 3060 Ti (8GB) 1枚のみで、Whisper large-v3 (fp16, 実測 3.1GB) と
   同時に載せる必要があるため、bf16 のままの Gemma3-4B(推定 8GB 超)は載らない。
   4bit 量子化で実測 3.4GB まで縮み、両方を同一GPUに同時ロードしても 8GB に収まることを
   スモークテストで確認済み(`/tmp/.../smoke_test_vlm.py` 相当、本ファイルの
   `InterventionDescriber` 内の量子化設定はそこで検証した設定をそのまま使っている)。
   このタスクはエッジ実行を考えなくてよい(ユーザ指示)ので、精度優先で最大サイズの
   ローカル利用可能VLMを選んだ。

2. **画像とTranscriptを1回のchat templateで結合し、単一のテキスト生成として d を得る。**
   VLM に「分類」ではなく「自由記述」をさせているのは、Part II-3 の強み2「単なる感情分類
   ではなく、介入テキストと状態変化ベクトルの対応関係そのものを学習する」という設計方針に
   従うため。分類ラベルにすると設計者が事前に介入のカテゴリを決め打ちすることになり、
   Part II-1 が問題視している「目的関数を人間が決め打ちする」失敗を、介入側で繰り返すことになる。

3. **発話が無い(Transcriptが空)区間を「無介入の安静区間」として明示的に扱う。**
   Part II-4 の対照区間の要件(無介入区間を必ず含める)に対応。VLM に「音声なし」であることを
   明示的に伝え、画像だけから机上の空論的な出来事を作文させない(hallucination対策)よう
   プロンプトで明示している。

## 対応する先行研究

- VLM による行動・介入の言語化という発想自体は、robotics 分野の "language-conditioned reward"
  / "VLM as a judge" 系の研究(例: Eureka, Yu et al. 2023; VLM-RM, Rocamonde et al. 2023)に近い。
  ただし本タスクは強化学習の報酬設計ではなく、d と Δs の対応を教師なしで学習する回帰/対照学習の
  入力生成であり、目的が異なる。
- 画像+テキストの統合表現生成という点では CLIP(Radford et al., 2021)の実務的後継にあたる
  VLM(Gemma3, Qwen2-VL 等)のいずれでもよいが、日本語での自由記述能力とローカルでの
  ライセンス(Gemma license)・サイズのバランスから Gemma3-4B-it を採用した。
"""

from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

VLM_MODEL_ID = "google/gemma-3-4b-it"

_SYSTEM_PROMPT = (
    "あなたはイヌとヒトの交流を観察する研究アシスタントです。"
    "与えられた画像(交流の様子)と、その間の音声の書き起こしから、"
    "「ヒトがイヌに対して行ったはたらきかけ」を簡潔な日本語1〜2文で説明してください。"
    "画像から確認できない内容を推測で補わないでください。"
    "書き起こしが空、または『(発話なし)』の場合は、ヒトが特に何もしていない"
    "(無介入の安静区間である)可能性を考慮し、画像から見える行動のみを述べてください。"
)


@lru_cache(maxsize=1)
def _load_vlm(device: str = "cuda:0"):
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, quantization_config=quant_config, device_map=device
    )
    model.eval()
    return processor, model


def describe_intervention(
    images: list[Image.Image] | Image.Image,
    transcript: str,
    device: str = "cuda:0",
    max_new_tokens: int = 96,
) -> str:
    """d = VLM(Transcript(音声), 画像) を計算する。

    images: 対象区間から抽出したフレーム(1枚でも複数枚でもよい)。
    transcript: transcript.transcribe() の出力。空文字は「発話なし」として扱う。
    """
    if isinstance(images, Image.Image):
        images = [images]
    processor, model = _load_vlm(device)

    content: list[dict] = [{"type": "image", "image": img} for img in images]
    transcript_display = transcript.strip() or "(発話なし)"
    content.append({"type": "text", "text": f"音声の書き起こし: {transcript_display}"})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    d = processor.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return d.strip()

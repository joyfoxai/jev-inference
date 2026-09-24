"""Self-contained loading and inference for local or Hugging Face model artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from .model import DecisionModel, encode, question_options, validate_record

DEFAULT_MODEL_ID = "joyfox/Qwen3.5-0.8B-JEV"


def resolve_model(source: str | Path, revision: str | None = None) -> Path:
    local = Path(source)
    if local.is_dir():
        return local.resolve()
    return Path(snapshot_download(
        repo_id=str(source), revision=revision,
        allow_patterns=["decision_config.json", "head.safetensors", "backbone/*", "tokenizer/*"],
    ))


class DecisionEngine:
    def __init__(self, model: DecisionModel, tokenizer, manifest: dict[str, Any], *,
                 model_name: str, device: str, dtype: str, cutoff_len: int):
        self.model = model
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.cutoff_len = cutoff_len
        self._lock = threading.Lock()

    @classmethod
    def load(cls, source: str | Path = DEFAULT_MODEL_ID, *, revision: str | None = None, device: str = "auto",
             dtype: str = "bfloat16", cutoff_len: int = 1024,
             model_name: str = "Qwen3.5-0.8B-JEV") -> "DecisionEngine":
        if dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("dtype must be float32, bfloat16, or float16")
        if cutoff_len < 1:
            raise ValueError("cutoff_len must be positive")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device == "cpu" and dtype == "float16":
            raise ValueError("float16 CPU inference is not supported")
        root = resolve_model(source, revision)
        manifest = json.loads((root / "decision_config.json").read_text())
        required = {"format_version": 2, "adapter": False, "execution": "rows",
                    "option_isolation": False}
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError(f"unsupported decision_config: {key} must be {expected!r}")
        torch_dtype = getattr(torch, dtype)
        backbone = AutoModel.from_pretrained(
            root / "backbone", dtype=torch_dtype,
            attn_implementation=manifest.get("attn_implementation", "sdpa"),
        )
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer requires a pad or EOS token")
        model = DecisionModel(backbone, int(manifest["head_dim"]))
        model.head.load_state_dict(load_file(root / "head.safetensors"))
        model.to(device=device, dtype=torch_dtype).eval()
        return cls(model, tokenizer, manifest, model_name=model_name,
                   device=device, dtype=dtype, cutoff_len=cutoff_len)

    def predict(self, record: dict[str, Any], temperature: float = 1.0) -> dict[str, Any]:
        return self.predict_batch([record], [temperature])[0]

    def predict_batch(self, records: list[dict[str, Any]],
                      temperatures: list[float] | None = None) -> list[dict[str, Any]]:
        if not records:
            raise ValueError("records must be nonempty")
        temperatures = temperatures or [1.0] * len(records)
        if len(temperatures) != len(records):
            raise ValueError("temperatures must match records")
        for record, temperature in zip(records, temperatures):
            validate_record(record)
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                    or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be a finite positive number")
        encoded = [encode(self.tokenizer, record, self.cutoff_len) for record in records]
        with self._lock, torch.inference_mode():
            logits = self.model(encoded, self.tokenizer.pad_token_id)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        outputs, cursor = [], 0
        for record, temperature in zip(records, temperatures):
            answers = {}
            for qid, question in record["questions"].items():
                values = logits[cursor]
                cursor += 1
                keys, _ = question_options(question)
                probs = (values / float(temperature)).softmax(-1).cpu().tolist()
                distribution = dict(zip(keys, probs))
                best = max(range(len(keys)), key=probs.__getitem__)
                answer = {"type": question["type"], "probabilities": distribution,
                          "confidence": probs[best]}
                if question["type"] == "choice":
                    answer["choice"] = keys[best]
                elif question["type"] == "noul":
                    answer["noul"] = distribution["true"]
                else:
                    answer["score"] = sum(index * value for index, value in enumerate(probs))
                    answer["level"] = best
                answers[qid] = answer
            outputs.append(answers)
        return outputs

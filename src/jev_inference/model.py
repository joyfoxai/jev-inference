"""Text encoding and query/key decision model; no training dependencies."""
from __future__ import annotations

import json
import math
from typing import Any

import torch
from torch import nn


def state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    raise ValueError("state must be a string, object, or array")


def question_options(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice requires at least two candidate descriptions")
        keys = list(criteria)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("choice candidate IDs must be nonempty strings")
        texts = [f"{key}: {description}" if description is not None else key
                 for key, description in criteria.items()]
    elif kind == "noul":
        keys = ["false", "true"]
        criteria = criteria or {}
        if not isinstance(criteria, dict) or set(criteria) - set(keys):
            raise ValueError("noul criteria may contain false/true only")
        texts = [f"{key}: {criteria.get(key, key)}" for key in keys]
    elif kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score requires at least two ordered descriptions")
        keys, texts = [str(i) for i in range(len(criteria))], criteria
    else:
        raise ValueError(f"unknown question type: {kind}")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("candidate descriptions must be nonempty text")
    return keys, texts


def validate_record(record: dict[str, Any], *, max_questions: int = 64,
                    max_candidates: int = 128, max_state_chars: int = 200_000) -> None:
    if not isinstance(record, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in record:
        raise ValueError("state is required")
    if len(state_text(record["state"])) > max_state_chars:
        raise ValueError(f"state exceeds {max_state_chars} characters")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    if len(questions) > max_questions:
        raise ValueError(f"at most {max_questions} questions are allowed")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(question, dict):
            raise ValueError("question IDs must be nonempty strings and questions must be objects")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"question {qid}: instructions must be nonempty text")
        keys, _ = question_options(question)
        if len(keys) > max_candidates:
            raise ValueError(f"question {qid}: at most {max_candidates} candidates are allowed")


def encode(tokenizer, record: dict[str, Any], cutoff_len: int) -> dict[str, Any]:
    def tokens(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    ids = tokens("State:\n" + state_text(record["state"]) + "\n")
    state_len = len(ids)
    segments = [0] * state_len
    positions = list(range(state_len))
    reads = []
    for segment, question in enumerate(record["questions"].values(), 1):
        _, options = question_options(question)
        branch = tokens(f"Question ({question['type']}):\n{question['instructions']}\n")
        branch_positions = list(range(state_len, state_len + len(branch)))
        ends = []
        for option in options:
            span = tokens("Option:\n" + option + "\n")
            branch_positions += list(range(state_len + len(branch), state_len + len(branch) + len(span)))
            branch += span
            ends.append(len(ids) + len(branch) - 1)
        suffix = tokens("Decision:\n")
        branch_positions += list(range(state_len + len(branch), state_len + len(branch) + len(suffix)))
        branch += suffix
        reads.append((len(ids) + len(branch) - 1, ends))
        ids += branch
        segments += [segment] * len(branch)
        positions += branch_positions
    longest_row = max(state_len + segments.count(segment) for segment in set(segments) - {0})
    if longest_row > cutoff_len:
        raise ValueError(f"input requires {longest_row} tokens, cutoff_len={cutoff_len}")
    return {"ids": ids, "segments": segments, "positions": positions, "reads": reads}


class DecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head_dim = head_dim
        self.head = nn.ModuleDict({
            "query": nn.Linear(backbone.config.hidden_size, head_dim),
            "key": nn.Linear(backbone.config.hidden_size, head_dim),
        })

    def forward(self, records: list[dict[str, Any]], pad_id: int) -> list[torch.Tensor]:
        rows = []
        for record in records:
            state_len = record["segments"].count(0)
            start = state_len
            for decision, options in record["reads"]:
                offset = start - state_len
                ids = record["ids"][:state_len] + record["ids"][start:decision + 1]
                rows.append({
                    "ids": ids,
                    "positions": record["positions"][:state_len] + record["positions"][start:decision + 1],
                    "reads": [(decision - offset, [position - offset for position in options])],
                })
                start = decision + 1
        device = next(self.parameters()).device
        width = max(len(row["ids"]) for row in rows)
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        positions = torch.zeros_like(ids)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            length = len(row["ids"])
            ids[index, :length] = torch.tensor(row["ids"], device=device)
            positions[index, :length] = torch.tensor(row["positions"], device=device)
            attention[index, :length] = 1
        hidden = self.backbone(input_ids=ids, position_ids=positions,
                               attention_mask=attention, use_cache=False).last_hidden_state
        results = []
        for index, row in enumerate(rows):
            decision, options = row["reads"][0]
            head_dtype = self.head["query"].weight.dtype
            query = self.head["query"](hidden[index, decision].to(head_dtype))
            keys = self.head["key"](hidden[index, options].to(head_dtype))
            results.append((keys @ query / math.sqrt(self.head_dim)).float())
        return results

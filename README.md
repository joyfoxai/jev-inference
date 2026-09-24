# Qwen3.5-0.8B-JEV Inference

Inference utilities for `Qwen3.5-0.8B-JEV`, a text decision model that returns probability distributions over runtime-defined candidates.

- Model: [`joyfox/Qwen3.5-0.8B-JEV`](https://huggingface.co/joyfox/Qwen3.5-0.8B-JEV)
- Source: [`joyfoxai/jev-inference`](https://github.com/joyfoxai/jev-inference)

## Features

- Load a model from a local directory or a Hugging Face repository.
- Run `choice`, `noul`, and `score` questions.
- Run single-record or batched inference.
- Use the Python API or command-line interface.
- Run on CUDA or CPU with BF16, FP16, or FP32 where supported.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Python Usage

### Load a local model

```python
from jev_inference import DecisionEngine

engine = DecisionEngine.load(
    "/path/to/Qwen3.5-0.8B-JEV",
    device="cuda",
    dtype="bfloat16",
    cutoff_len=1024,
)
```

### Load from Hugging Face

```python
from jev_inference import DecisionEngine

engine = DecisionEngine.load(
    "joyfox/Qwen3.5-0.8B-JEV",
    device="cuda",
    dtype="bfloat16",
)
```

The official model is the default, so the repository ID may be omitted:

```python
engine = DecisionEngine.load(device="cuda", dtype="bfloat16")
```

Authenticate with Hugging Face before loading a private repository.

### Run inference

```python
record = {
    "state": "The customer was charged twice and requests an immediate refund.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, invoicing, or refunds",
                "account": "Account access or profile",
                "technical": "Bugs or integrations",
            },
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer request a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgently should this ticket be handled?",
            "criteria": [
                "Normal queue",
                "Priority handling",
                "Immediate human handling",
            ],
        },
    },
}

result = engine.predict(record)
print(result)
```

Example output:

```python
{
    "route": {
        "type": "choice",
        "probabilities": {
            "billing": 0.9692,
            "account": 0.0169,
            "technical": 0.0138,
        },
        "confidence": 0.9692,
        "choice": "billing",
    },
    "refund_requested": {
        "type": "noul",
        "probabilities": {
            "false": 0.0656,
            "true": 0.9344,
        },
        "confidence": 0.9344,
        "noul": 0.9344,
    },
    "urgency": {
        "type": "score",
        "probabilities": {
            "0": 0.1049,
            "1": 0.4928,
            "2": 0.4022,
        },
        "confidence": 0.4928,
        "score": 1.2973,
        "level": 1,
    },
}
```

## Batch Inference

```python
records = [record_1, record_2, record_3]
results = engine.predict_batch(records)
```

Batch inference processes multiple records in one model call and provides better GPU throughput than repeatedly calling `predict()`.

Each record may use a different temperature:

```python
results = engine.predict_batch(
    records,
    temperatures=[1.0, 0.8, 1.2],
)
```

A positive temperature changes the sharpness of the probability distribution without changing the ordering of finite candidate logits.

## Command-Line Usage

Read a request from a JSON file:

```bash
jev-predict \
  --input examples/request.json \
  --device cuda \
  --dtype bfloat16
```

Read a request from standard input:

```bash
cat examples/request.json | jev-predict \
  --device cuda
```

## Question Types

### Choice

Select from runtime-defined candidates:

```python
{
    "type": "choice",
    "instructions": "Select the responsible team.",
    "criteria": {
        "sales": "Pre-sales questions",
        "support": "Post-sales support",
    },
}
```

Candidate IDs are preserved in the output probability mapping.

### Noul

Return a Boolean probability:

```python
{
    "type": "noul",
    "instructions": "Does the customer explicitly request a refund?",
}
```

The output candidates are always `false` and `true`. The `noul` field contains the probability assigned to `true`.

### Score

Return a distribution over ordered levels:

```python
{
    "type": "score",
    "instructions": "Assess the risk level.",
    "criteria": ["Low risk", "Medium risk", "High risk"],
}
```

The result includes the complete level distribution, the most likely `level`, and the probability-weighted expected `score`.

## Device and Precision

Automatically select CUDA when available:

```python
engine = DecisionEngine.load(model_path, device="auto")
```

Use CUDA with BF16:

```python
engine = DecisionEngine.load(
    model_path,
    device="cuda",
    dtype="bfloat16",
)
```

Use CPU with FP32:

```python
engine = DecisionEngine.load(
    model_path,
    device="cpu",
    dtype="float32",
)
```

## Input Limits

- The default token limit is 1,024.
- A record may contain up to 64 questions.
- A question may contain up to 128 candidates.
- The serialized state may contain up to 200,000 characters.
- Oversized inputs raise an error and are never silently truncated.
- Inputs are text-only; images are not supported.
- KV cache is not used.

## Repository Layout

```text
jev-inference/
├── examples/
│   └── request.json
├── src/jev_inference/
│   ├── __init__.py
│   ├── cli.py
│   ├── engine.py
│   └── model.py
├── tests/
│   └── test_model.py
├── pyproject.toml
└── README.md
```

## Tests

```bash
pip install -e '.[test]'
pytest -q
```

## License

Apache-2.0.

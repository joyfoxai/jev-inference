"""Run one decision request from a JSON file or stdin."""
from __future__ import annotations

import argparse
import json
import sys

from .engine import DEFAULT_MODEL_ID, DecisionEngine


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help=f"Local model directory or Hugging Face repo ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--input", help="JSON input file; defaults to stdin")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cutoff-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)
    text = open(args.input).read() if args.input else sys.stdin.read()
    record = json.loads(text)
    engine = DecisionEngine.load(args.model, revision=args.revision, device=args.device,
                                 dtype=args.dtype, cutoff_len=args.cutoff_len)
    print(json.dumps(engine.predict(record, args.temperature), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

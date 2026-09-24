import pytest

from jev_inference.model import question_options, state_text, validate_record


def test_question_types():
    assert question_options({"type": "choice", "criteria": {"a": "A", "b": "B"}})[0] == ["a", "b"]
    assert question_options({"type": "noul"})[0] == ["false", "true"]
    assert question_options({"type": "score", "criteria": ["low", "high"]})[0] == ["0", "1"]


def test_validation():
    record = {"state": {"ticket": 1}, "questions": {
        "route": {"type": "choice", "instructions": "route", "criteria": {"a": "A", "b": "B"}}
    }}
    validate_record(record)
    assert state_text(record["state"]) == '{"ticket": 1}'
    with pytest.raises(ValueError):
        validate_record({"state": "x", "questions": {}})
    with pytest.raises(ValueError):
        question_options({"type": "choice", "criteria": {"a": "A"}})

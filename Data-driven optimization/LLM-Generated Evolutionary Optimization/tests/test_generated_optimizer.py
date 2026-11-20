from pathlib import Path

from llm_fs.validator import dynamic_validate


ROOT = Path(__file__).resolve().parents[1]


def test_llm_generated_optimizer_contract():
    errors = dynamic_validate(ROOT / "generated_algorithms" / "llm_generated_optimizer.py")
    assert errors == []


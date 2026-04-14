"""
Optional LLM-as-judge interface (stub + contract). No API calls until you wire a backend.

Intended categories (align with roadmap / report):
  - refusal: hard | soft | none
  - usefulness_0_10: human-like scale (filled by judge)
  - hallucination_risk: low | medium | high | unknown
  - unsafe_detail_risk: low | medium | high | unknown

Implement ``run_judge_batch`` with your provider (OpenAI, Anthropic, local vLLM, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


RefusalLabel = Literal["hard", "soft", "none"]
RiskLevel = Literal["low", "medium", "high", "unknown"]


@dataclass
class JudgeExample:
    query_id: int
    instruction: str
    response: str
    system_prompt: Optional[str] = None


@dataclass
class JudgeVerdict:
    query_id: int
    refusal: RefusalLabel
    usefulness_0_10: Optional[float]
    hallucination_risk: RiskLevel = "unknown"
    unsafe_detail_risk: RiskLevel = "unknown"
    notes: str = ""
    raw_judge: Dict[str, Any] = field(default_factory=dict)


def run_judge_batch(
    examples: List[JudgeExample],
    *,
    backend: Literal["stub"] = "stub",
    **kwargs: Any,
) -> List[JudgeVerdict]:
    """
    ``backend="stub"`` returns placeholder verdicts so pipelines can be tested without keys.

    Replace with a real backend by branching on ``backend`` and calling your API.
    """
    if backend != "stub":
        raise NotImplementedError(f"Backend {backend!r} is not wired yet. Implement in llm_judge.py.")

    out: List[JudgeVerdict] = []
    for ex in examples:
        out.append(
            JudgeVerdict(
                query_id=ex.query_id,
                refusal="none",
                usefulness_0_10=None,
                hallucination_risk="unknown",
                unsafe_detail_risk="unknown",
                notes="stub backend — replace with LLM judge",
                raw_judge={"backend": "stub"},
            )
        )
    return out


def verdicts_to_json_rows(verdicts: List[JudgeVerdict]) -> List[Dict[str, Any]]:
    rows = []
    for v in verdicts:
        rows.append(
            {
                "id": v.query_id,
                "refusal": v.refusal,
                "usefulness_0_10": v.usefulness_0_10,
                "hallucination_risk": v.hallucination_risk,
                "unsafe_detail_risk": v.unsafe_detail_risk,
                "notes": v.notes,
            }
        )
    return rows


def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Stub LLM judge: writes placeholder labels for pipeline testing.")
    p.add_argument("--input", required=True, help="Responses JSON (id, instruction, response).")
    p.add_argument("--out", required=True, help="Output JSON rows from stub judge.")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    examples = [
        JudgeExample(
            query_id=int(item.get("id", -1)),
            instruction=item.get("instruction", ""),
            response=item.get("response", ""),
            system_prompt=item.get("system_prompt"),
        )
        for item in data
    ]
    verdicts = run_judge_batch(examples, backend="stub")
    rows = verdicts_to_json_rows(verdicts)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} stub verdicts to {outp}")


if __name__ == "__main__":
    main()

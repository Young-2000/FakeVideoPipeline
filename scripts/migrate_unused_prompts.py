#!/usr/bin/env python3
"""Move prompts with zero repo references into src/agent/prompt_unused.py.

This script is intentionally narrow: it only migrates the currently confirmed
unused prompt constants so we do not disturb prompts still imported elsewhere.
It is idempotent enough for local maintenance and fails loudly if the expected
markers are not found.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "src" / "agent" / "prompts.py"
UNUSED_PATH = ROOT / "src" / "agent" / "prompt_unused.py"


@dataclass(frozen=True)
class BlockSpec:
    name: str
    next_name: str


BLOCKS: tuple[BlockSpec, ...] = (
    BlockSpec("PROMPT_EXTRACT_CLUES", "PROMPT_CLUSTER_SHOTS"),
    BlockSpec("PROMPT_CLUSTER_SHOTS", "PROMPT_REFLECT_REFINE"),
    BlockSpec("PROMPT_SUFFICIENCY_JUDGMENT", "PROMPT_DEEPSEARCH_NEXT_STEP"),
    BlockSpec("ACTION_SYSTEM_PROMPT", "# ============================================================\n# Stage B - forgery analysis (NEW)\n# ============================================================"),
    BlockSpec("_FORGERY_POINT_SCHEMA", "# ============================================================\n# Stage C - LLM-as-judge (NEW)\n# ============================================================"),
)


UNUSED_HEADER = '''"""Unused prompts kept for reference.

These prompt definitions are currently not imported anywhere in the repository.
They were migrated out of `src.agent.prompts` to keep the active prompt module
focused on the runtime paths that are still exercised.
"""

from __future__ import annotations

'''


def extract_block(text: str, start_marker: str, end_marker: str) -> tuple[str, str]:
    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError(f"start marker not found: {start_marker}")
    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError(f"end marker not found for: {start_marker}")
    return text[start:end].rstrip() + "\n\n", text[:start] + text[end:]


def normalize_spacing(text: str) -> str:
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text.rstrip() + "\n"


def main() -> None:
    prompts_text = PROMPTS_PATH.read_text(encoding="utf-8")
    extracted_parts: list[str] = []

    for spec in BLOCKS:
        block, prompts_text = extract_block(prompts_text, spec.name, spec.next_name)
        extracted_parts.append(block)

    unused_body = UNUSED_HEADER + "".join(extracted_parts).rstrip() + "\n"

    UNUSED_PATH.write_text(unused_body, encoding="utf-8")
    PROMPTS_PATH.write_text(normalize_spacing(prompts_text), encoding="utf-8")

    print(f"Wrote {UNUSED_PATH}")
    print(f"Updated {PROMPTS_PATH}")


if __name__ == "__main__":
    main()

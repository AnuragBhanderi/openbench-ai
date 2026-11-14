"""Import local benchmark files into OpenBench's JSONL dataset format (v0.5.0, issue #35).

Supported input formats (local files only; nothing is downloaded):

* ``qa-jsonl`` / ``qa-csv``: question/answer pairs. Field names default to ``question`` and
  ``answer`` and can be changed with ``question_field`` / ``answer_field``.
* ``mcq-jsonl``: multiple choice. ``question``, ``choices`` (a list, or an object such as
  ``{"A": ..., "B": ...}``) and ``answer`` (a letter, a 0-based index, or the text of the right
  choice).
* ``mcq-csv``: multiple choice in the common column layout ``question, A, B, C, D, answer``
  (extra lettered columns are used too; a header row is required).

Multiple-choice prompts list the choices as ``A. ...`` lines and ask for the letter only, so
``exact_match`` scores them fairly; the reference answer is the letter.

Every record keeps its original ID and position in ``metadata.source_id`` / ``source_index``,
and the source and licence you give in ``metadata.source`` / ``metadata.license``. Records that
cannot be converted are skipped and listed with the reason. The output is validated with the
normal dataset loader.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openbench.exceptions import ConfigurationError

FORMATS = ("qa-jsonl", "qa-csv", "mcq-jsonl", "mcq-csv")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MCQ_INSTRUCTION = "Answer with the letter of the correct option only."
_ID_SAFE = re.compile(r"[^A-Za-z0-9_.\-:]+")


@dataclass
class ImportResult:
    records: list[dict[str, Any]]
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records)


def _read_rows(path: Path, fmt: str) -> list[tuple[int, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if fmt.endswith("jsonl"):
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as exc:
                rows.append((i, exc))
        return rows
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ConfigurationError(f"{path} has no header row")
    return [(i, row) for i, row in enumerate(reader, start=2)]


def _clean_id(value: Any) -> str:
    return _ID_SAFE.sub("_", str(value)).strip("_")[:120]


def _mcq_choices(row: dict[str, Any], fmt: str) -> list[str]:
    if fmt == "mcq-csv":
        choices = []
        for letter in LETTERS:
            if letter in row and row[letter] not in (None, ""):
                choices.append(str(row[letter]).strip())
            elif letter in row:
                break
            else:
                break
        return choices
    raw = row.get("choices", row.get("options"))
    if isinstance(raw, list):
        return [str(c).strip() for c in raw]
    if isinstance(raw, dict):
        keys = sorted(raw)
        return [str(raw[k]).strip() for k in keys]
    raise ValueError("missing 'choices' (a list or an object)")


def _mcq_answer_letter(answer: Any, choices: list[str]) -> str:
    if isinstance(answer, bool):
        raise ValueError("answer must be a letter, an index or the choice text")
    if isinstance(answer, int):
        if not 0 <= answer < len(choices):
            raise ValueError(f"answer index {answer} is out of range")
        return LETTERS[answer]
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in LETTERS[: len(choices)]:
        return text.upper()
    if text.isdigit() and int(text) < len(choices):
        return LETTERS[int(text)]
    matches = [i for i, c in enumerate(choices) if c.strip().casefold() == text.casefold()]
    if len(matches) == 1:
        return LETTERS[matches[0]]
    raise ValueError(f"answer {text!r} does not match a choice letter, index or text")


def import_benchmark(
    path: str | Path,
    fmt: str,
    *,
    category: str,
    id_prefix: str | None = None,
    source: str | None = None,
    license_name: str | None = None,
    difficulty: str | None = None,
    question_field: str = "question",
    answer_field: str = "answer",
    id_field: str = "id",
) -> ImportResult:
    """Convert a local benchmark file into OpenBench dataset records."""
    if fmt not in FORMATS:
        raise ConfigurationError(f"Unsupported format '{fmt}'. Use one of: {', '.join(FORMATS)}")
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Input file does not exist: {target}")
    if not category or not category.strip():
        raise ConfigurationError("A category is required")
    if difficulty is not None and difficulty not in ("easy", "medium", "hard"):
        raise ConfigurationError("difficulty must be easy, medium or hard")
    prefix = _clean_id(id_prefix or category)
    result = ImportResult(records=[])
    seen: set[str] = set()
    for index, (line, row) in enumerate(_read_rows(target, fmt)):
        try:
            if isinstance(row, Exception):
                raise ValueError(f"invalid JSON ({row})")
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
            question = row.get(question_field)
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"missing or empty '{question_field}'")
            original_id = row.get(id_field)
            if fmt.startswith("mcq"):
                choices = _mcq_choices(row, fmt)
                if len(choices) < 2:
                    raise ValueError("fewer than two choices")
                if any(not c for c in choices):
                    raise ValueError("empty choice text")
                letter = _mcq_answer_letter(row.get(answer_field), choices)
                options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
                prompt = f"{question.strip()}\n\n{options}\n\n{MCQ_INSTRUCTION}"
                expected = letter
                extra = {"choices": choices, "answer_text": choices[LETTERS.index(letter)]}
            else:
                answer = row.get(answer_field)
                if isinstance(answer, (int, float)) and not isinstance(answer, bool):
                    answer = str(answer)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"missing or empty '{answer_field}'")
                prompt, expected, extra = question.strip(), answer.strip(), {}
            suffix = _clean_id(original_id) if original_id not in (None, "") else f"{index + 1:05d}"
            rid = f"{prefix}_{suffix}"
            if rid in seen:
                rid = f"{prefix}_{index + 1:05d}"
            if rid in seen:
                raise ValueError(f"duplicate ID {rid}")
            seen.add(rid)
            metadata: dict[str, Any] = {
                "source_id": None if original_id in (None, "") else str(original_id),
                "source_index": index,
                "source_file": target.name,
                "import_format": fmt,
                **extra,
            }
            if source:
                metadata["source"] = source
            if license_name:
                metadata["license"] = license_name
            if difficulty:
                metadata["difficulty"] = difficulty
            result.records.append({"id": rid, "category": category.strip(), "prompt": prompt, "expected_answer": expected, "metadata": metadata})
        except ValueError as exc:
            result.skipped.append({"line": line, "reason": str(exc)})
    return result

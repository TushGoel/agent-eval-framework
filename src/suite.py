"""
Eval suite — a named, versioned collection of test cases for an agent.

Suites are the unit of enrollment: register a suite once and it runs
automatically on every agent update, catching regressions before they ship.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class EvalCase:
    case_id: str
    input: dict[str, Any]
    expected: Any
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalSuite:
    name: str
    version: str
    cases: list[EvalCase] = field(default_factory=list)
    judge: Optional[Callable[[Any, Any], bool]] = None

    def add_case(self, case: EvalCase) -> None:
        self.cases.append(case)

    def default_judge(self, expected: Any, actual: Any) -> bool:
        """Exact-match judge. Replace with semantic / LLM-as-judge for open-ended tasks."""
        return str(expected).strip().lower() == str(actual).strip().lower()

    def evaluate_case(self, expected: Any, actual: Any) -> bool:
        judge = self.judge or self.default_judge
        return judge(expected, actual)

    def __len__(self) -> int:
        return len(self.cases)


class SuiteRegistry:
    """Central registry for all eval suites. New agents enroll here."""

    def __init__(self) -> None:
        self._suites: dict[str, EvalSuite] = {}

    def enroll(self, suite: EvalSuite) -> None:
        key = f"{suite.name}@{suite.version}"
        self._suites[key] = suite

    def get(self, name: str, version: str = "latest") -> EvalSuite:
        if version == "latest":
            matches = [k for k in self._suites if k.startswith(f"{name}@")]
            if not matches:
                raise KeyError(f"No suite enrolled for '{name}'")
            return self._suites[sorted(matches)[-1]]
        return self._suites[f"{name}@{version}"]

    def list_suites(self) -> list[str]:
        return list(self._suites.keys())

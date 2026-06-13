"""Retrieval and guardrail state machine for the Wolof assistant demo app.

This is an improved version of the original lexical-overlap state machine.
It is intentionally still "no embeddings, no vector database" so it stays
cheap and runs instantly on CPU inside a Hugging Face Space, but it adds the
following on top of the original:

1. Source-aware retrieval: examples are filtered by category AND by source
   family (synth / aya / soynade), so an orthography-normalization example
   from "soynade" is never used as a few-shot example for an "education"
   question and vice versa, unless the user explicitly asks for orthography
   correction.
2. A confidence score for the retrieved context, derived from lexical
   overlap between the question and the retrieved examples.
3. A simple keyword-based safety filter for unsafe categories (e.g. medical
   dosage, self-harm, violence) that the small fine-tuned model should not
   attempt to answer freely.
4. An explicit "should_answer" / "abstain" decision: when the question is
   flagged unsafe, the state machine recommends that the app show a
   limitation message instead of (or in addition to) the model's raw
   generation. When confidence is low, it adds a visible caveat.

Why this design and not embeddings:
- The fine-tuned model is a 0.6B model deployed on a small CPU Space; adding
  a sentence-embedding model would roughly double memory/startup cost for a
  marginal retrieval quality gain on a ~5k row dataset with short Wolof
  sentences, where lexical overlap already works reasonably well.
- A vector database is unnecessary infrastructure for a single-file JSONL
  dataset that fits comfortably in memory.
- If retrieval quality on a larger/multilingual dataset becomes the
  bottleneck, the natural next step would be a small multilingual sentence
  embedding model (e.g. a MiniLM variant) with cosine similarity, behind the
  same `retrieve()` / `prepare()` interface so the app code would not need
  to change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence


ALL_CATEGORY = "All"
DEFAULT_WEB_CATEGORIES = (
    ALL_CATEGORY,
    "education",
    "agriculture",
    "sante",
    "transport",
    "culture",
    "orthography",
    "language",
)

TOKEN_ALIASES = {
    "mathematik": "math",
    "mathematique": "math",
    "mathématique": "math",
    "mathematiques": "math",
    "mathématiques": "math",
    "mathematics": "math",
    "maths": "math",
    "mat": "math",
    "francais": "français",
    "français": "français",
    "anglais": "anglais",
    "wolof": "wolof",
}

# Keywords (lowercase, French/Wolof/English mix) that indicate a question is
# asking for an orthography correction rather than domain knowledge.
ORTHOGRAPHY_KEYWORDS = {
    "orthographe",
    "orthograph",
    "corrige",
    "standard",
    "standardise",
    "standardisé",
    "ortograf",
    "njort",
}

# Keywords that flag a question as potentially unsafe for a small, narrow
# fine-tuned assistant to answer without a strong disclaimer. This is not a
# medical/legal classifier -- it is a coarse trigger for an abstention
# message in the demo app.
UNSAFE_KEYWORDS = (
    "dosage", "dose", "doses", "medicament", "médicament", "medicaments",
    "overdose", "poison", "empoisonn", "suicide", "tuer", "arme", "bomb",
    "bombe", "explosif", "self-harm", "automutilation", "drogue",
    "diagnos", "prescri",
)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [TOKEN_ALIASES.get(token, token) for token in tokens]


def is_all_category(category: str) -> bool:
    return category.strip().lower() in {"all", "*", "toutes", "tous"}


def load_training_rows(path: str | Path) -> List[Dict[str, str]]:
    """Load rows from a JSONL file with instruction/input/output[/source] fields."""
    data_path = Path(path)
    rows = []
    with data_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(validate_training_row(row, line_no))
    return rows


def validate_training_row(row: Dict[str, str], line_no: int) -> Dict[str, str]:
    required = ("instruction", "input", "output")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"Training row {line_no} is missing fields: {missing}")

    clean = {name: str(row.get(name, "")).strip() for name in required}
    clean["source"] = str(row.get("source", "unknown")).strip() or "unknown"
    if not clean["instruction"] or not clean["output"]:
        raise ValueError(f"Training row {line_no} has an empty instruction or output.")
    return clean


@dataclass
class RetrievalResult:
    """Result of a retrieval call, including diagnostics for the UI."""

    examples: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    is_unsafe: bool = False
    is_orthography_request: bool = False
    should_answer: bool = True
    reason: str = "ok"


class CategoryContextStateMachine:
    """Retrieve in-category, source-aware examples and score confidence.

    The state machine answers two questions for the demo app:

    1. Which few-shot examples should be shown to the model (and to the
       user, for transparency) for a given category and question?
    2. Should the app even attempt to show a generated answer, or should it
       show a limitation / abstention message instead?
    """

    def __init__(
        self,
        rows: Iterable[Dict[str, str]],
        categories: Sequence[str] = DEFAULT_WEB_CATEGORIES,
        min_confidence: float = 1.0,
    ) -> None:
        normalized_categories = []
        for category in categories:
            if category not in normalized_categories:
                normalized_categories.append(category)
        if ALL_CATEGORY not in normalized_categories:
            normalized_categories.insert(0, ALL_CATEGORY)

        self.categories = tuple(normalized_categories)
        self.min_confidence = min_confidence
        self.all_rows = list(rows)

        self.rows_by_category: Dict[str, List[Dict[str, str]]] = {
            category: [] for category in self.categories if not is_all_category(category)
        }
        self.sources = set()
        for row in self.all_rows:
            category = row["input"]
            self.sources.add(row.get("source", "unknown"))
            if category in self.rows_by_category:
                self.rows_by_category[category].append(row)

    def available_categories(self) -> List[str]:
        visible = [ALL_CATEGORY]
        visible.extend(
            category
            for category in self.categories
            if not is_all_category(category) and self.rows_by_category.get(category)
        )
        return visible

    def available_sources(self) -> List[str]:
        return sorted(self.sources)

    # ------------------------------------------------------------------
    # Safety / intent flags
    # ------------------------------------------------------------------
    def _is_unsafe(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in UNSAFE_KEYWORDS)

    def _is_orthography_request(self, question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in ORTHOGRAPHY_KEYWORDS)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _candidate_rows(self, category: str, question: str) -> List[Dict[str, str]]:
        if self._is_orthography_request(question):
            # Force source-aware retrieval toward the orthography source,
            # regardless of the selected category, since the user is asking
            # for a spelling correction.
            ortho_rows = [row for row in self.all_rows if row.get("source") == "soynade"]
            if ortho_rows:
                return ortho_rows

        if is_all_category(category):
            return self.all_rows
        if category in self.rows_by_category:
            return self.rows_by_category[category]
        raise ValueError(f"Unknown category: {category}")

    def retrieve(self, category: str, question: str, k: int = 4) -> RetrievalResult:
        is_unsafe = self._is_unsafe(question)
        is_ortho = self._is_orthography_request(question)

        rows = self._candidate_rows(category, question)

        question_tokens = Counter(tokenize(question))
        scored = []
        for row in rows:
            instruction_tokens = Counter(tokenize(row["instruction"]))
            output_tokens = Counter(tokenize(row["output"]))
            instruction_overlap = sum((question_tokens & instruction_tokens).values())
            output_overlap = sum((question_tokens & output_tokens).values())
            score = 2 * instruction_overlap + output_overlap
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)

        selected: List[Dict[str, str]] = []
        seen_outputs = set()
        for score, row in scored:
            if score <= 0:
                continue
            output_key = row["output"].lower()
            if output_key in seen_outputs:
                continue
            selected.append(row)
            seen_outputs.add(output_key)
            if len(selected) >= k:
                break

        top_score = scored[0][0] if scored else 0
        if not selected:
            selected = [row for _, row in scored[:k]]
            confidence = 0.0
        else:
            # Normalize: confidence in [0, 1], saturating once the top match
            # shares several tokens with the question.
            confidence = min(1.0, top_score / 6.0)

        should_answer = not is_unsafe
        reason = "unsafe_topic" if is_unsafe else "ok"

        return RetrievalResult(
            examples=selected,
            confidence=round(confidence, 3),
            is_unsafe=is_unsafe,
            is_orthography_request=is_ortho,
            should_answer=should_answer,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def build_context(self, examples: Sequence[Dict[str, str]]) -> str:
        chunks = []
        for index, row in enumerate(examples, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"Example {index} (source: {row.get('source', 'unknown')})",
                        f"Instruction: {row['instruction']}",
                        f"Expected Wolof answer: {row['output']}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def few_shot_messages(self, examples: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = []
        for row in examples:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Context category: {row['input']}\n"
                        f"Instruction: {row['instruction']}"
                    ),
                }
            )
            messages.append({"role": "assistant", "content": row["output"]})
        return messages

    def augment_instruction(
        self,
        category: str,
        question: str,
        examples: Sequence[Dict[str, str]],
    ) -> str:
        context = self.build_context(examples)
        return (
            "Student question:\n"
            f"{question.strip()}\n\n"
            "In-domain examples from the training data:\n"
            f"{context}\n\n"
            "Task:\n"
            "- Answer the student question in clear Wolof.\n"
            "- Use the examples only as style and domain context.\n"
            "- Do not copy an unrelated example.\n"
            "- Do not output hidden reasoning or <think> tags.\n"
            f"- Keep the answer appropriate for the category: {category}."
        )

    # ------------------------------------------------------------------
    # High-level convenience method for the Space app
    # ------------------------------------------------------------------
    def prepare(self, category: str, question: str, k: int = 4) -> Dict[str, object]:
        """Return everything the app needs to decide what to show the user.

        Returns a dict with:
          - "augmented_instruction": prompt text to feed the model, or None
            if should_answer is False and the app prefers to skip
            generation entirely.
          - "examples": the retrieved examples (for display in the UI).
          - "confidence": float in [0, 1].
          - "should_answer": bool.
          - "limitation_message": a human-readable message to show the
            user when should_answer is False, or as a footer note
            otherwise.
        """
        result = self.retrieve(category, question, k=k)

        if result.is_unsafe:
            limitation = (
                "This assistant is a small fine-tuned demo model for "
                "education/agriculture/health-FAQ/transport/culture topics "
                "in Wolof. It is NOT a medical, legal, or safety authority "
                "and will not answer questions about medication dosages, "
                "self-harm, or weapons. Please consult a qualified "
                "professional or local emergency services."
            )
            return {
                "augmented_instruction": None,
                "examples": [],
                "confidence": result.confidence,
                "should_answer": False,
                "limitation_message": limitation,
            }

        augmented = self.augment_instruction(category, question, result.examples)

        low_conf_threshold = self.min_confidence / 6.0
        if result.confidence < low_conf_threshold and not result.is_orthography_request:
            limitation = (
                "Note: this question does not closely match the "
                "assistant's training data (retrieval confidence "
                f"{result.confidence:.2f}). Treat the answer below as a "
                "draft, not a verified fact."
            )
        else:
            limitation = ""

        return {
            "augmented_instruction": augmented,
            "examples": result.examples,
            "confidence": result.confidence,
            "should_answer": True,
            "limitation_message": limitation,
        }

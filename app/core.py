
"""Core functionality for the Legal Sentence Game."""

import re
from pathlib import Path


# Location of the legal text corpus.
CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "legal_corpus.txt"


def prepare_legal_corpus(text: str) -> list[str]:
    """
    Clean legal text and convert it into lowercase word tokens.

    Args:
        text: Raw legal text.

    Returns:
        A list of cleaned word tokens.
    """
    if not isinstance(text, str):
        raise TypeError("Legal text must be a string.")

    text = text.lower()

    # Keep letters and numbers; remove punctuation and symbols.
    tokens = re.findall(r"[a-z0-9]+", text)

    return tokens


# --- Milestone 1: Prepare a legal text corpus for n-gram model training.
def milestone_1() -> list[str]:
    """Load and prepare the legal text corpus."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Legal corpus not found: {CORPUS_PATH}"
        )

    raw_text = CORPUS_PATH.read_text(encoding="utf-8")
    tokens = prepare_legal_corpus(raw_text)

    if not tokens:
        raise ValueError("The legal corpus is empty.")

    return tokens


# --- Milestone 2: Implement the n-gram next-word predictor.
def milestone_2():
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences.
def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Classify the completed sentence and provide feedback.
def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")

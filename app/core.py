
"""Core functionality for the Legal Sentence Game."""
from collections import defaultdict, Counter
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
def build_ngram_model(tokens: list[str], n: int = 2) -> dict:
    """Build an n-gram model from a list of tokens."""
    if not isinstance(tokens, list):
        raise TypeError("tokens must be a list")

    if n < 2:
        raise ValueError("n must be at least 2")

    model = defaultdict(Counter)

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i + n - 1])
        next_word = tokens[i + n - 1]
        model[context][next_word] += 1

    return dict(model)


def predict_next_word(model: dict, context: list[str]) -> str | None:
    """Predict the most common next word for the given context."""
    if not isinstance(context, list):
        raise TypeError("context must be a list")

    if not context:
        return None

    context_key = tuple(context)
    predictions = model.get(context_key)

    if not predictions:
        return None

    return predictions.most_common(1)[0][0]


def milestone_2():
    """Build a bigram model using the prepared legal corpus."""
    tokens = milestone_1()
    return build_ngram_model(tokens, n=2)


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences.
def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Classify the completed sentence and provide feedback.
def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")

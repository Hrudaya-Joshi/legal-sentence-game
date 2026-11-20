
"""Core functionality for the Legal Sentence Game."""
from collections import defaultdict, Counter
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, GPT2ForSequenceClassification

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

# Milestone 3: GPT-2 legal clause classification

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "legal-clause-classifier"
)

_classifier = None
_tokenizer = None


def load_clause_classifier():
    """Load the locally fine-tuned GPT-2 classifier."""
    global _classifier, _tokenizer

    if _classifier is None or _tokenizer is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_DIR}"
            )

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        _classifier = GPT2ForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )

        _classifier.eval()

    return _tokenizer, _classifier


def classify_legal_clause(text: str) -> dict:
    """Classify a legal sentence and return its label and confidence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text cannot be empty")

    tokenizer, model = load_clause_classifier()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=96,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_id = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0][predicted_id].item()

    label = model.config.id2label[predicted_id]

    return {
        "label": label,
        "confidence": confidence,
    }


def milestone_3():
    """Load the trained legal clause classifier."""
    return load_clause_classifier()


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences.
def milestone_4():
    """Build and return the n-gram model for the CLI game."""
    return milestone_2()


# --- Milestone 5: Classify the completed sentence and provide feedback.
def milestone_5(text: str) -> dict:
    """Classify a completed legal sentence using the GPT-2 classifier."""
    result = classify_legal_clause(text)

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "feedback": (
            "Your legal sentence has been classified successfully."
        ),
    }
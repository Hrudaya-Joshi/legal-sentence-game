
import pytest

from app.core import (
    prepare_legal_corpus,
    milestone_1,
    build_ngram_model,
    predict_next_word,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    classify_legal_clause,
)
from types import SimpleNamespace
import torch
import app.core as core

# Milestone 1: Prepare a legal text corpus.
def test_milestone_1():
    text = "THE Court shall enforce this Agreement!"

    tokens = prepare_legal_corpus(text)

    assert tokens == [
        "the", "court", "shall", "enforce", "this", "agreement"
    ]

    assert "!" not in tokens
    assert all(token == token.lower() for token in tokens)

    corpus = milestone_1()

    assert len(corpus) > 0
    assert "court" in corpus


def test_empty_text():
    assert prepare_legal_corpus("") == []


def test_invalid_text():
    with pytest.raises(TypeError):
        prepare_legal_corpus(123)


# Milestone 2: N-gram prediction.
def test_milestone_2():
    model = milestone_2()

    assert isinstance(model, dict)
    assert len(model) > 0


def test_build_bigram_model():
    tokens = ["the", "court", "shall", "the", "court", "may"]

    model = build_ngram_model(tokens, n=2)

    assert model[("the",)]["court"] == 2
    assert model[("court",)]["shall"] == 1
    assert model[("court",)]["may"] == 1


def test_predict_next_word():
    tokens = ["the", "court", "shall", "the", "court", "may"]
    model = build_ngram_model(tokens, n=2)

    assert predict_next_word(model, ["the"]) == "court"
    assert predict_next_word(model, ["unknown"]) is None


def test_build_trigram_model():
    tokens = ["the", "court", "shall", "the", "court", "shall"]

    model = build_ngram_model(tokens, n=3)

    assert model[("the", "court")]["shall"] == 2


def test_invalid_ngram_order():
    with pytest.raises(ValueError):
        build_ngram_model(["the", "court"], n=1)


# Milestone 3: GPT-2 legal clause classifier.
def test_milestone_3(monkeypatch):
    fake_tokenizer = object()
    fake_model = object()

    monkeypatch.setattr(
        core,
        "load_clause_classifier",
        lambda: (fake_tokenizer, fake_model),
    )

    tokenizer, model = milestone_3()

    assert tokenizer is fake_tokenizer
    assert model is fake_model


def test_classify_legal_clause(monkeypatch):
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {}

    class FakeModel:
        config = SimpleNamespace(
            id2label={0: "Confidentiality", 1: "Parties"}
        )

        def __call__(self, **kwargs):
            return SimpleNamespace(
                logits=torch.tensor([[0.1, 2.0]])
            )

    monkeypatch.setattr(
        core,
        "load_clause_classifier",
        lambda: (FakeTokenizer(), FakeModel()),
    )

    result = classify_legal_clause(
        "The parties agree to comply with all applicable laws."
    )

    assert result["label"] == "Parties"
    assert 0 <= result["confidence"] <= 1


def test_classify_empty_sentence():
    with pytest.raises(ValueError):
        classify_legal_clause("")


def test_classify_invalid_input():
    with pytest.raises(TypeError):
        classify_legal_clause(123)


# Milestone 4: CLI game model.
def test_milestone_4():
    model = milestone_4()

    assert isinstance(model, dict)
    assert len(model) > 0
    assert predict_next_word(model, ["the"]) == "parties"


# Milestone 5: Completed sentence classification and feedback.
def test_milestone_5(monkeypatch):
    monkeypatch.setattr(
        core,
        "classify_legal_clause",
        lambda text: {
            "label": "Parties",
            "confidence": 0.85,
        },
    )

    result = milestone_5(
        "The parties agree to comply with all applicable laws."
    )

    assert result["label"] == "Parties"
    assert result["confidence"] == 0.85
    assert isinstance(result["feedback"], str)
    assert len(result["feedback"]) > 0


import pytest

from app.core import prepare_legal_corpus, milestone_1
from app.core import build_ngram_model, predict_next_word
from app.core import classify_legal_clause
# Milestone 1: Prepare a legal text corpus.
def test_milestone_1():
    # Test text cleaning.
    text = "THE Court shall enforce this Agreement!"

    tokens = prepare_legal_corpus(text)

    assert tokens == [
        "the", "court", "shall", "enforce", "this", "agreement"
    ]

    # Test that punctuation is removed.
    assert "!" not in tokens

    # Test that all words are lowercase.
    assert all(token == token.lower() for token in tokens)

    # Test actual legal corpus loading.
    corpus = milestone_1()

    assert len(corpus) > 0
    assert "court" in corpus


def test_empty_text():
    assert prepare_legal_corpus("") == []


def test_invalid_text():
    with pytest.raises(TypeError):
        prepare_legal_corpus(123)
    

# Milestone 2
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

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

def test_classify_legal_clause():
    result = classify_legal_clause(
        "The parties agree to comply with all applicable laws."
    )

    assert "label" in result
    assert "confidence" in result
    assert isinstance(result["label"], str)
    assert 0 <= result["confidence"] <= 1


def test_classify_empty_sentence():
    with pytest.raises(ValueError):
        classify_legal_clause("")


def test_classify_invalid_input():
    with pytest.raises(TypeError):
        classify_legal_clause(123)
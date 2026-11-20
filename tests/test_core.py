
import pytest

from app.core import prepare_legal_corpus, milestone_1


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

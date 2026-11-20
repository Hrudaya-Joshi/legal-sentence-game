
"""Command-line entry point for the Legal Sentence Game."""

import argparse

from app.core import (
    milestone_4,
    milestone_5,
    predict_next_word,
    prepare_legal_corpus,
)


def play_game() -> None:
    """Play the legal sentence completion and classification game."""

    print("\n===================================")
    print("       LEGAL SENTENCE GAME")
    print("===================================")

    model = milestone_4()

    fragment = input("\nEnter a legal sentence fragment: ").strip()
    tokens = prepare_legal_corpus(fragment)

    if not tokens:
        print("Please enter a valid sentence fragment.")
        return

    prediction = predict_next_word(model, [tokens[-1]])

    if prediction is None:
        print("No prediction is available for this fragment.")
        return

    print(f"\nYour sentence: {fragment}")
    print(f"Model prediction: {prediction}")

    guess = input("Guess the next word: ").strip().lower()

    if not guess:
        print("Please enter a guess.")
        return

    if guess == prediction:
        print("Correct! Great legal drafting!")
    else:
        print("Not quite! Keep practising.")

    completed_sentence = f"{fragment} {guess}"

    print(f"\nCompleted sentence: {completed_sentence}")
    print("\nClassifying your completed sentence...")

    try:
        result = milestone_5(completed_sentence)

        print(f"\nClause category: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Feedback: {result['feedback']}")

    except (FileNotFoundError, RuntimeError) as error:
        print(f"Classification unavailable: {error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-sentence-game",
        description="Practice legal sentence drafting using n-gram prediction and GPT-2 classification."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="Check that everything is wired up.")
    sub.add_parser("play", help="Start the legal sentence completion game.")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Legal Sentence Game is ready.")

    elif args.command == "play":
        play_game()


if __name__ == "__main__":
    main()

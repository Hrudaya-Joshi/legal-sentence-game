# Legal Sentence Game

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/dpcpzmpr3b/badge.svg)](https://msrit.ask.thili.ai/b/dpcpzmpr3b)

A command-line game for law students to practice drafting legal sentences. It uses an n-gram model to predict the next word and a fine-tuned GPT-2 sequence classifier to identify legal clause categories and provide feedback.

## Features

- Prepare and clean a legal text corpus.
- Build a bigram and n-gram next-word prediction model.
- Complete legal sentence fragments through an interactive CLI.
- Classify completed sentences using a fine-tuned GPT-2 model.
- Display the predicted clause category and confidence score.

## Project milestones

- [x] **1. Prepare a legal text corpus for n-gram model training.**
- [x] **2. Implement the n-gram next-word predictor.**
- [x] **3. Train and integrate a GPT-2 legal clause classifier.**
- [x] **4. Create the command-line game interface.**
- [x] **5. Classify completed sentences and provide feedback.**

## Requirements

- Python 3.12
- Git
- Internet connection for downloading the training dataset and pretrained GPT-2 checkpoint.

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the GPT-2 classifier

The project uses the CUAD legal clause classification dataset and a pretrained GPT-2 checkpoint.

Run:

```bash
python -m app.train_classifier
```

The training script fine-tunes the classifier and saves the model and tokenizer locally in:

```text
models/legal-clause-classifier/
```

The training script uses a sample of 1,200 clauses and one training epoch by default.

**Important:** The trained model files are not included in GitHub. Run the training script before using the classification feature on a new computer.

## Run the game

Check that the CLI is working:

```bash
python -m app.cli hello
```

Start the game:

```bash
python -m app.cli play
```

The game asks you to enter a legal sentence fragment and guess the next word. It compares your guess with the n-gram prediction, completes the sentence using the model prediction, and classifies the sentence you entered.

## Run tests

```bash
python -m pytest -v
```

The automated tests use mocked classifier components where appropriate, so they do not require locally saved GPT-2 model weights.

## Project structure

```text
legal-sentence-game/
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── core.py
│   ├── llm.py
│   ├── storage.py
│   └── train_classifier.py
├── data/
│   └── legal_corpus.txt
├── tests/
│   ├── test_core.py
│   └── test_smoke.py
├── models/
│   └── legal-clause-classifier/
├── .github/
│   └── workflows/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Model evaluation

In the initial training run, the classifier achieved **63.33% validation accuracy** on a held-out test split of 120 clauses.

This is an initial result from a small training experiment, not a benchmark of production-level performance.

## Settings and secrets

The project includes an optional `.env` configuration for the chat integration in `app/llm.py`.

Never commit `.env` or expose API keys in source code. Use `.env.example` as a configuration reference.

## Future improvements

- Add a scoring system for sentence completion.
- Support different legal domains and difficulty levels.
- Display multiple next-word predictions as hints.
- Improve classification accuracy with more training data and evaluation.

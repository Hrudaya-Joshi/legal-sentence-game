# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dpcpzmpr3b/badge.svg)](https://ask.thili.ai/b/dpcpzmpr3b)

This project develops a command-line game for law students to practice drafting legal sentences. Players complete partial sentences using n-gram predictions, and a fine-tuned GPT-2 model provides instant feedback on the generated clause types. It's a fun way to hone legal drafting skills.

## What you are building

**Goal.** You will build a command-line interface (CLI) game where law students complete legal sentences using n-gram predictions and receive instant feedback on their generated clause types from a fine-tuned GPT-2 model.

**Done when.** The CLI game is fully functional, allowing players to complete legal sentences using n-gram predictions. After completion, the game classifies the full sentence with a fine-tuned GPT-2 model and provides instant feedback on the generated clause types.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a legal text corpus for n-gram model training.**  
      Hint: Use the text processing techniques from the 'Build a Next-Word Predictor from Scratch' drill to clean and prepare your legal corpus.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Construct the n-gram model and its prediction logic as demonstrated in the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Load the fine-tuned GPT-2 model for legal clause classification.**  
      Hint: Load the pre-trained or fine-tuned GPT-2 model, as shown in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **4. Create a CLI game interface presenting partial legal sentences for completion.**  
      Hint: Design a simple command-line interface to present sentences and capture user input.
- [ ] **5. After user completion, classify the full sentence and provide feedback.**  
      Hint: Apply the GPT-2 model's classification capabilities from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill to evaluate the completed sentences.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement a scoring system based on the GPT-2 classification accuracy of the completed sentences.
- Allow players to choose different legal sub-domains or difficulty levels for sentence generation.
- Display the n-gram model's top N predictions as hints to the player during sentence completion.

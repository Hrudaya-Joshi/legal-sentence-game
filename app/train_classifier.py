
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2ForSequenceClassification,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "legal-clause-classifier"

# Start with a small sample for CPU training.
SAMPLE_SIZE = 1200
MAX_LENGTH = 96
BATCH_SIZE = 2
EPOCHS = 1


def main():
    print("Loading CUAD legal clause dataset...")

    dataset = load_dataset(
        "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
        split="train",
    )

    # Build the label mappings from the dataset.
    label_pairs = {}
    for row in dataset:
        label_pairs[row["class_id"]] = row["label"]

    id2label = {
        class_id: label
        for class_id, label in sorted(label_pairs.items())
    }
    label2id = {label: class_id for class_id, label in id2label.items()}

    # Select a reproducible sample for the initial CPU training run.
    sample_size = min(SAMPLE_SIZE, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(sample_size))

    print(f"Using {sample_size} clauses.")
    print(f"Number of clause categories: {len(label2id)}")

    print("Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["clause"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = batch["class_id"]
        return encoded

    print("Tokenizing clauses...")
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )

    split = tokenized.train_test_split(test_size=0.1, seed=42)

    train_data = split["train"].with_format("torch")
    test_data = split["test"].with_format("torch")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
    )

    print("Loading GPT-2 classification model...")
    config = GPT2Config.from_pretrained(
        "gpt2",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        config=config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Starting training on CPU...")

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            total_loss += loss.item()

            if step % 25 == 0:
                print(f"Epoch {epoch + 1}, step {step}, loss {loss.item():.4f}")

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {average_loss:.4f}")

    print("Evaluating model...")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    accuracy = correct / total if total else 0
    print(f"Validation accuracy: {accuracy:.2%}")

    print("Saving trained model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()

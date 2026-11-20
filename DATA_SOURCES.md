# Dataset Sources and Data Preparation

## 1. Training Dataset

The project uses the Contract Understanding Atticus Dataset (CUAD), specifically the clause classification dataset hosted on Hugging Face.

- **Dataset:** CUAD v1 Contract Understanding Clause Classification
- **Hugging Face repository:** https://huggingface.co/datasets/dvgodoy/CUAD_v1_Contract_Understanding_clause_classification
- **Original project:** https://github.com/TheAtticusProject/cuad
- **License listed by the dataset host:** CC BY 4.0
- **Available records:** 13,155 labeled clauses
- **Categories:** 41 legal clause categories

The dataset contains labeled clauses extracted from commercial legal contracts.

## 2. Data Loading

The training script loads the training split using the Hugging Face `datasets` library:

```python
dataset = load_dataset(
    "dvgodoy/CUAD_v1_Contract_Understanding_clause_classification",
    split="train",
)
```

The dataset provides clause text and corresponding category identifiers and labels.

## 3. Data Preparation

The training pipeline performs these operations:

1. Creates mappings between category IDs and category names.
2. Shuffles the dataset using random seed `42`.
3. Selects up to 1,200 clauses for the initial experiment.
4. Tokenizes clause text using the GPT-2 tokenizer.
5. Truncates and pads sequences to a maximum length of 96 tokens.
6. Splits the selected data into 90% training and 10% validation sets using seed `42`.

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Base model | GPT-2 |
| Training sample | 1,200 clauses |
| Number of categories | 41 |
| Maximum sequence length | 96 |
| Batch size | 2 |
| Epochs | 1 |
| Training device | CPU |
| Optimizer | AdamW |
| Learning rate | 0.00005 |

## 5. Local Legal Corpus

The project also contains `data/legal_corpus.txt`.

This is a small illustrative seed corpus containing legal-style sentences used by the n-gram sentence prediction component. It is separate from the CUAD dataset used for GPT-2 classification training.

## 6. Reproducing Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python -m app.train_classifier
```

An internet connection is required to download the dataset and pretrained GPT-2 checkpoint on the first run.

The trained tokenizer and model are saved locally in:

`models/legal-clause-classifier/`

Model weights are not committed to the GitHub repository.

## 7. Limitations

The initial experiment uses only a subset of the available dataset and one training epoch. Its validation accuracy is preliminary and should not be treated as production-level legal classification performance.

The model provides experimental clause categorization, not legal advice or a substitute for professional legal review.

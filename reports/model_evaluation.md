# GPT-2 Legal Clause Classifier – Model Evaluation

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | GPT-2 |
| Dataset | CUAD legal clause classification dataset |
| Training samples | 1,200 |
| Validation samples | 120 |
| Training epochs | 1 |
| Batch size | 2 |
| Maximum sequence length | 96 |
| Device | CPU |

## Evaluation Result

The initial training run produced the following result:

**Validation accuracy: 63.33%**

The accuracy was measured on a held-out validation split containing 120 samples.

## Interpretation

The result demonstrates that the fine-tuned GPT-2 classifier learned to distinguish legal clause categories in the experimental dataset.

This is an initial experiment using a relatively small training sample and one epoch. It should not be interpreted as production-level classification performance.

## Limitations

- Training used only 1,200 clauses from the available dataset.
- Only one training epoch was performed.
- The validation set contained 120 samples.
- Per-class precision, recall, and F1-score have not yet been evaluated.
- Further training and evaluation are required to establish generalization performance.

## Reproducing the Experiment

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python -m app.train_classifier
```

The trained model is saved locally in:

```text
models/legal-clause-classifier/
```

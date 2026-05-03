# Data

```python
df = pd.read_csv("hallucination_dataset.csv")
df.head()
```

Each row contains a transcription, the extracted `address` and `title`, and ground truth labels (`address WRONG`, `title WRONG`) telling us whether each extraction is a hallucination.

Since the `title` and `address` are pre-computed we also get their corresponding probabilities. For each output we get both absolute and normalized probablities, labeled with suffixes `_abs` and `_norm` respectivally.

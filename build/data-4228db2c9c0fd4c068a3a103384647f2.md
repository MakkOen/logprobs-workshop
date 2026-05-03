# Data

```python
df = pd.read_csv("hallucination_dataset.csv")
df.head()
```

Each row contains a transcription, the extracted `address` and `title`, and ground truth labels (`address WRONG`, `title WRONG`) telling us whether each extraction is a hallucination.

The dataset already has logprob-based confidence columns pre-computed (`address_prob_norm`, `address_prob_abs`, `title_prob_norm`) so we can jump straight to exploring the signals.

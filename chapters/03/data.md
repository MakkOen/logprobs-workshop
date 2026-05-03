# Data

We're working with police/firefighter/EMS radio transcripts from San Francisco dispatch. Each row is a short transcript labelled as either an **incident** (1) or **noise** (0).

An incident must have two things: what is happening and where it's happening. Everything else (unit acknowledgements, admin chatter, resolved calls, ambiguous traffic) is noise.

```python
df = pd.read_csv("filtering_dataset.csv")
df.head(5)
```


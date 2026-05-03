# Model Confidence

When the model extracts a multi-token output like an address, we can compute a confidence score over the whole sequence using the per-token logprobs. The dataset has two variants pre-computed: absolute sequence probability (`address_prob_abs`) and length-normalized (`address_prob_norm`).

Do hallucinated outputs have lower confidence? Let's check:

```python
address_correct_df = df[df["address WRONG"] == 0]
address_wrong_df = df[df["address WRONG"] == 1]

plt.figure(figsize=(10, 6))
plt.hist(address_correct_df['address_prob_norm'], bins=30, alpha=0.5, label='Correct Address', density=True)
plt.hist(address_wrong_df['address_prob_norm'], bins=30, alpha=0.5, label='Wrong Address', density=True)
plt.xlabel('address_prob_norm')
plt.ylabel('Density')
plt.title('Distribution of address_prob_norm')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

Same for the absolute probability:

```python
plt.figure(figsize=(10, 6))
plt.hist(address_correct_df['address_prob_abs'], bins=30, alpha=0.5, label='Correct Address', density=True)
plt.hist(address_wrong_df['address_prob_abs'], bins=30, alpha=0.5, label='Wrong Address', density=True)
plt.xlabel('address_prob_abs')
plt.ylabel('Density')
plt.title('Distribution of address_prob_abs')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

And for title:

```python
title_correct_df = df[df["title WRONG"] == 0]
title_wrong_df = df[df["title WRONG"] == 1]

plt.figure(figsize=(10, 6))
plt.hist(title_correct_df['title_prob_norm'], bins=30, alpha=0.5, label='Correct title', density=True)
plt.hist(title_wrong_df['title_prob_norm'], bins=30, alpha=0.5, label='Wrong title', density=True)
plt.xlabel('title_prob_norm')
plt.ylabel('Density')
plt.title('Distribution of title_prob_norm')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

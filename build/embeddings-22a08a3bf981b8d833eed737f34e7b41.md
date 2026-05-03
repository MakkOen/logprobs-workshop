# Semantic Similarity

A hallucinated address or title should be semantically distant from the original transcription. We embed both and compute cosine similarity as a second signal.

```python
def get_embedding(text, model="text-embedding-3-small"):
    if pd.isna(text) or text == "":
        return None
    text = str(text).replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

df['transcription_embedding'] = df['transcription'].progress_apply(get_embedding)
df['address_embedding'] = df['address'].progress_apply(get_embedding)
df['title_embedding'] = df['title'].progress_apply(get_embedding)
```

```python
def calculate_cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return None
    return cosine_similarity([vec1], [vec2])[0][0]

df['similarity_transcription_address'] = df.apply(
    lambda row: calculate_cosine_similarity(row['transcription_embedding'], row['address_embedding']), axis=1
)
df['similarity_transcription_title'] = df.apply(
    lambda row: calculate_cosine_similarity(row['transcription_embedding'], row['title_embedding']), axis=1
)
```

Plot the distributions to see how well similarity separates correct from wrong outputs:

```python
address_correct_df = df[df["address WRONG"] == 0]
address_wrong_df = df[df["address WRONG"] == 1]
title_correct_df = df[df["title WRONG"] == 0]
title_wrong_df = df[df["title WRONG"] == 1]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(address_correct_df['similarity_transcription_address'].dropna(), bins=30, alpha=0.5, label='Correct Address', density=True)
plt.hist(address_wrong_df['similarity_transcription_address'].dropna(), bins=30, alpha=0.5, label='Wrong Address', density=True)
plt.xlabel('Cosine Similarity')
plt.title('Transcription vs Address Similarity')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(title_correct_df['similarity_transcription_title'].dropna(), bins=30, alpha=0.5, label='Correct Title', density=True)
plt.hist(title_wrong_df['similarity_transcription_title'].dropna(), bins=30, alpha=0.5, label='Wrong Title', density=True)
plt.xlabel('Cosine Similarity')
plt.title('Transcription vs Title Similarity')
plt.legend()

plt.tight_layout()
plt.show()
```

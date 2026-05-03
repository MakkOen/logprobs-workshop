# Classification using LLM

We prompt the model to output exactly one token (`incident` or `noise`) and read the logprob of that token as our confidence score.

## Single prediction

```python
INSTRUCTIONS = """Given a police/firefighter/EMS radio transcript, decide whether the transcript is about an incident happening or just noise.
An incident must contain:
* what is happening
* address where it's happening.
Output only 'incident' or 'noise'.
"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": df.iloc[0].transcription},
    ],
    top_logprobs=2,
    include=["message.output_text.logprobs"],
)
```

The predicted token and its probability:

```python
response.output[0].content[0].logprobs[0].top_logprobs[0].token
math.exp(response.output[0].content[0].logprobs[0].top_logprobs[0].logprob)
```

## Run over the full dataset

We sum up the probability mass on the `incident` token across the top logprobs. This handles cases where the model outputs `Incident` (capitalised) or other variants:

```python
output_token = []
output_prob = []

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": row.transcription},
        ],
        top_logprobs=4,
        include=["message.output_text.logprobs"],
    )
    output_token.append(response.output_text)
    prob = 0.0
    for top_logprob in response.output[0].content[0].logprobs[0].top_logprobs:
        if top_logprob.token.lower() == "incident":
            prob += math.exp(top_logprob.logprob)
    output_prob.append(prob)

df["output_token"] = output_token
df["output_prob"] = output_prob
```

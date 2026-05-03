# Multi-token Response

For a response with multiple tokens, logprobs are returned for each position. The first token's top alternatives:

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello from ML Prague",
    top_logprobs=2,
    include=["message.output_text.logprobs"],
)

sum = 0.0
for logprob in response.output[0].content[0].logprobs[0].top_logprobs:
    print(f"token: {logprob.token}, prob: {math.exp(logprob.logprob)}")
    sum += math.exp(logprob.logprob)
print(f"total: {sum}")
```

To get the probability of the entire sequence, multiply the chosen token's probability at each position (or equivalently, sum the log-probs and exponentiate):

```python
sum = 0.0
for logprob in response.output[0].content[0].logprobs:
    print(f"token: {logprob.top_logprobs[0].token}, prob: {math.exp(logprob.logprob)}")
    sum += logprob.logprob

print(math.exp(sum))
```

Sequence probability drops quickly with length. Normalize by dividing the log-sum by the number of tokens:

```python
# Normalized
math.exp(sum / len(response.output[0].content[0].logprobs))
```

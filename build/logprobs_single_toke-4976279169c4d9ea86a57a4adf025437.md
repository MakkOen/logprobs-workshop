# Logprobs

Add `top_logprobs` and `include=["message.output_text.logprobs"]` to get the probability distribution over the top N tokens at each position.

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="Give me a random number. Answer only the number",
    top_logprobs=10,
    include=["message.output_text.logprobs"],
)

print(response.output_text)
```

The raw response contains log-probabilities. Convert with `math.exp()` to get actual probabilities:

```python
sum = 0.0
for logprob in response.output[0].content[0].logprobs[0].top_logprobs:
    print(f"token: {logprob.token}, prob: {math.exp(logprob.logprob)}")
    sum += math.exp(logprob.logprob)
print(f"total: {sum}")
```

The probabilities across all top alternatives sum to ≤ 1. The remainder is mass spread across tokens not in the top 10.

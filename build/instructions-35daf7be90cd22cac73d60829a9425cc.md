# Instructions

Before we get into logprobs, a quick recap of how to steer a model with a system instruction.

```python
INSTRUCTIONS = "Answer only in Czech"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": "Hello from ML Prague"},
    ],
)

print(response.output_text)
```

The `system` message sets the model's behaviour. Everything in `user` is the actual input.

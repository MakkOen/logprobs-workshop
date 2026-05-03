# Note: Reasoning Models

## Providers locking down capabilities

Modern frontier models like `gpt-5` (and OpenAI's `o`-series reasoning models) do not support logprobs. The API simply returns an error, a deliberate provider decision to limit access to model internals:

```python
response = client.responses.create(
    model="gpt-5-mini",
    input="Give me a random number. Answer only the number",
    top_logprobs=10,
    include=["message.output_text.logprobs"],
)
# → API error: logprobs not supported for this model
```

This is a growing trend. As providers push users toward their most capable (and most expensive) models, low-level access like logprobs gets quietly dropped. Keep this in mind when choosing models for production confidence scoring. `gpt-4o-mini` still supports it, for now.

## Effect of reasoning on logprobs

Even on models that do support logprobs, adding chain-of-thought reasoning to the prompt affects the confidence on the final token. This isn't a bug or a workaround, just something to be aware of.

Without reasoning in the prompt, the model goes straight to the answer. The final token logprob reflects raw pattern-matching confidence:

```python
INSTRUCTIONS = """Given a text, decide whether it's about a cat or a dog.
Output only dog or cat as your last token"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "system", "content": INSTRUCTIONS},
           {"role": "user", "content": "Hello from ML Prague"}],
    top_logprobs=2,
    include=["message.output_text.logprobs"],
)
math.exp(response.output[0].content[0].logprobs[-1].top_logprobs[0].logprob)
```

With reasoning in the prompt, the model works through the problem first. The final token logprob is now conditioned on all the reasoning tokens that came before it, typically higher confidence, but also more influenced by the reasoning chain:

```python
INSTRUCTIONS = """Given a text, decide whether it's about a cat or a dog.
First think about your reasoning and then output dog or cat as your last token"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "system", "content": INSTRUCTIONS},
           {"role": "user", "content": "Hello from ML Prague"}],
    top_logprobs=2,
    include=["message.output_text.logprobs"],
)

print(response.output_text)
math.exp(response.output[0].content[0].logprobs[-1].top_logprobs[0].logprob)
```

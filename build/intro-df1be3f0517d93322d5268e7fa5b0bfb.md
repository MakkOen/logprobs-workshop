# Responses API

Throughout this workshop we use the **OpenAI Responses API** (`client.responses.create`). It's the newer, unified API that replaces the older Chat Completions API, same models, cleaner interface.

## Sanity check

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello from ML Prague",
)

print(response.output_text)
```

## Response object

The full response object contains a lot more than just the text. The key fields we'll use in this workshop:

```python
response.output_text          # the generated text, shorthand
response.output               # list of output items
response.output[0].content    # list of content blocks in the first output item
response.output[0].content[0].text   # the text of the first content block
response.output[0].content[0].logprobs  # per-token logprob data (when requested)
```

You can inspect the full structure with:

```python
import json
print(json.dumps(response.model_dump(), indent=2))
```

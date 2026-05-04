# Responses API

Throughout this workshop we use the **OpenAI Responses API** (`client.responses.create`). It's the newer, unified API that replaces the older Chat Completions API. 

:::{note}
Everything done here is also possible with the Chat Completions API. Both APIs are offered by most providers and self-hosting frameworks, and the techniques in this workshop are not limited to either.
:::

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
```

You can inspect the full structure with:

```python
import json
print(json.dumps(response.model_dump(), indent=2))
```

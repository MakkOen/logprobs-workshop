# LLM Judge

A third signal: ask a second model to review each extraction and decide whether it's supported by the transcription. We get a single token (`yes`/`no`) back and use its logprob as the confidence.

```python
INSTRUCTIONS_TITLE = """
Given a radio transcript and description, decide whether the given description is incorrect or not supported by the transcription fully.
Output only yes for incorrect description, no for correct.
"""
INSTRUCTIONS_ADDRESS = """
Given a radio transcript and address, decide whether the given address is incorrect or not supported by the transcription fully.
Output only yes for incorrect description, no for correct.
"""

output_prob_title = []
output_prob_address = []

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    res_title = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": INSTRUCTIONS_TITLE},
            {"role": "user", "content": f"Transcript: {row.transcription}\nDescription: {row.title}"}
        ],
        top_logprobs=4,
        include=["message.output_text.logprobs"]
    )
    prob_title = sum(
        math.exp(lp.logprob)
        for lp in res_title.output[0].content[0].logprobs[0].top_logprobs
        if lp.token.lower().strip() == "yes"
    )
    output_prob_title.append(prob_title)

    res_address = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": INSTRUCTIONS_ADDRESS},
            {"role": "user", "content": f"Transcript: {row.transcription}\nAddress: {row.processed_address}"}
        ],
        top_logprobs=4,
        include=["message.output_text.logprobs"]
    )
    prob_address = sum(
        math.exp(lp.logprob)
        for lp in res_address.output[0].content[0].logprobs[0].top_logprobs
        if lp.token.lower().strip() == "yes"
    )
    output_prob_address.append(prob_address)

df['llm_judge_title_prob'] = output_prob_title
df['llm_judge_address_prob'] = output_prob_address
```

Visualize how well the judge separates hallucinations:

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df[df['address WRONG'] == 0]['llm_judge_address_prob'], bins=10, alpha=0.5, label='Correct Address', density=True)
plt.hist(df[df['address WRONG'] == 1]['llm_judge_address_prob'], bins=10, alpha=0.5, label='Wrong Address', density=True)
plt.xlabel('Judge "Yes" Probability')
plt.title('LLM Judge: Address')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df[df['title WRONG'] == 0]['llm_judge_title_prob'], bins=10, alpha=0.5, label='Correct Title', density=True)
plt.hist(df[df['title WRONG'] == 1]['llm_judge_title_prob'], bins=10, alpha=0.5, label='Wrong Title', density=True)
plt.xlabel('Judge "Yes" Probability')
plt.title('LLM Judge: Title')
plt.legend()

plt.tight_layout()
plt.show()
```

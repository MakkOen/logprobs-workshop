# Real Use Case: Hallucination Detection

The original task: given a police/firefighter/EMS radio transcription, extract the address and a short title describing what's happening.

Our task: given the transcription and the extracted output, decide whether the address or title is wrong or hallucinated.

We'll look at three independent signals and then combine them:

1. **Model confidence** (logprobs on the extracted tokens)
2. **Semantic similarity** (embedding distance between transcription and output)
3. **LLM judge** (a second model reviewing the output)

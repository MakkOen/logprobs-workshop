# Outro

## What we covered

We are used to treating LLM as black-boxes. And praying to them when they don't give us what they want. In this workshop we have shown, on real usecase and data, that we don't need to abandon our agency when using LLMs and we can go back to our ML roots and create proper models with them:

- **Binary classification** — instead of parsing yes/no text, we read the token probability directly as a continuous confidence score and tuned the threshold to hit our precision/recall target
- **Hallucination detection** — we combined three independent signals (logprob confidence, embedding similarity, LLM judge) and used Optuna to find the optimal thresholds across all six features

The core idea throughout: logprobs let you treat an LLM output as a probability distribution, not just a string. That's the bridge back to classical ML.

---

## A note on simplification

The approaches in this workshop were intentionally kept simple to keep the focus on logprobs. In practice, you can push results further with:

- **Prompt engineering** — clearer instructions reduce ambiguity and concentrate probability mass on the correct tokens
- **In-context learning** — a few well-chosen examples shift the model's calibration towards your specific domain
- **Fine-tuning** — adapts the model's weights directly to your task, improving both accuracy and confidence quality

:::{admonition} These techniques complement, not replace
:class: tip
None of these are alternatives to logprob-based confidence scoring: they make it work better. A well-prompted or fine-tuned model produces sharper probability distributions, which means cleaner signals and better downstream classification.
:::

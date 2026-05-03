# Binary Classification

A natural first use case for logprobs: instead of just asking the model for a yes/no answer, we ask it to output a single token (`incident` or `noise`) and read off the probability of that token directly.

This gives us a **continuous confidence score** for every prediction, which we can then threshold however we like, trading precision against recall.

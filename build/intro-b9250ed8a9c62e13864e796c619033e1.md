# Binary Classification

Binary classificator: getting yes/no answer to a question is an essential ML primitve and more than usual use case for LLMs.

A natural use for logprobs: instead of just asking the model for a yes/no and taking its answer as an output. We look at the logprobs of the yes/no tokens.

This gives us a **confidence score** for every prediction, which we can then threshold however we like, trading precision against recall.

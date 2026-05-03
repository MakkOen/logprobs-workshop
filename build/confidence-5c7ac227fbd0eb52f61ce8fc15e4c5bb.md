# Model Confidence

For a selected token $X_i$ with corresponding probability $p_i$, we can read $p_i$ as the model's **confidence** in that token given everything that came before it:

$$p_i = P(X_i \mid X_0, X_1, \ldots, X_{i-1})$$

## Single-token output

When the output is a single token (like `yes` / `no`, or a label), this is straightforward: the token's probability **is** the model's confidence in its answer.

:::{admonition} Example
:class: tip
Ask the model to classify a sentence as `spam` or `ham`. It outputs `spam` with $p = 0.94$. That's the model saying: *"I'm 94% sure this is spam."*
:::

## Multi-token output

For longer outputs it gets more complicated. The probability of the full sequence is the product of each token's conditional probability:

$$P(X_0, \ldots, X_n) = \prod_{i=0}^{n} p_i$$

This shrinks fast with length, which makes raw sequence probability hard to compare across outputs of different lengths. We'll deal with this properly in the chapters ahead.

:::{admonition} Coming up
:class: seealso
Chapter 02 covers exactly how to extract these logprobs from the API, handle multi-token sequences, and normalise for length.
:::

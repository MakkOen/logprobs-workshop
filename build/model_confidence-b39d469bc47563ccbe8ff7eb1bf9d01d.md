# Model Confidence

::::{admonition} Proper Confidence Estimation
:class: note dropdown

To properly estimate confidence we would need to repeatedly query the model on the same input, calculate the semantic mean of the sampled outputs, and then measure each output's distance to that mean.

We can't do this from logprobs alone: they only show us the branches along the single path the model took. We have no information about the probabilities of paths not taken:

:::{mermaid}
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 50}}}%%
flowchart TD
    INPUT["`Input`"]
    a1[ ] & a2[ ] & a3[ ]
    b1[ ] & b2[ ] & b3[ ] & b4[ ] & b5[ ] & b6[ ]
    c1[ ] & c2[ ] & c3[ ] & c4[ ] & c5[ ] & c6[ ] & c7[ ]

    INPUT --> a1 & a2 & a3

    a1 --> b1 & b2
    a2 --> b3 & b4
    a3 --> b5 & b6

    b1 --> c1 & c2
    b3 --> c3 & c4
    b5 --> c5 & c6
    b6 --> c7

    style INPUT fill:none,stroke:#555,stroke-width:2px,rx:6,ry:6,color:#000000

    style a1 fill:#1a1a2e,color:#ffffff,stroke:#52b788,stroke-width:3px
    style a2 fill:#1a1a2e,color:#ffffff,stroke:#1a6fa8,stroke-width:3px
    style a3 fill:#1a1a2e,color:#ffffff,stroke:#e07b39,stroke-width:3px

    style b3 fill:#1a1a2e,color:#ffffff,stroke:#1a6fa8,stroke-width:3px
    style b4 fill:#1a1a2e,color:#ffffff,stroke:#1a6fa8,stroke-width:3px

    style c3 fill:#1a1a2e,color:#ffffff,stroke:#1a6fa8,stroke-width:3px
    style c4 fill:#1a1a2e,color:#ffffff,stroke:#1a6fa8,stroke-width:3px

    style b1 stroke:#52b788,stroke-width:3px
    style c1 stroke:#52b788,stroke-width:3px
    style b6 stroke:#e07b39,stroke-width:3px
    style c7 stroke:#e07b39,stroke-width:3px

    linkStyle default stroke-width:1.5px,opacity:0.3
    linkStyle 0 stroke:#52b788,stroke-width:2.5px,opacity:1
    linkStyle 3 stroke:#52b788,stroke-width:2.5px,opacity:1
    linkStyle 9 stroke:#52b788,stroke-width:2.5px,opacity:1
    linkStyle 1 stroke:#1a6fa8,stroke-width:2.5px,opacity:1
    linkStyle 5 stroke:#1a6fa8,stroke-width:2.5px,opacity:1
    linkStyle 11 stroke:#1a6fa8,stroke-width:2.5px,opacity:1
    linkStyle 2 stroke:#e07b39,stroke-width:2.5px,opacity:1
    linkStyle 8 stroke:#e07b39,stroke-width:2.5px,opacity:1
    linkStyle 15 stroke:#e07b39,stroke-width:2.5px,opacity:1
:::

<p>
<span style="color:#1a6fa8">■</span> path taken &nbsp;
<span style="color:#1a1a2e">■</span> known logprobs &nbsp;
<span style="color:#52b788">■</span> possible path (unknown) &nbsp;
<span style="color:#e07b39">■</span> possible path (unknown)
</p>

:::{note}
Logprobs are a good approximation when:
- The output isn't too long
- There are fewer semantically equivalent correct answers (e.g. `"dog"` vs `"canine"`: the more synonyms, the more the true confidence is spread across paths we never see)
:::
::::

When the model extracts a multi-token output like an address, we can compute a confidence score over the whole sequence using the per-token logprobs. The dataset has two variants pre-computed: absolute sequence probability (`address_prob_abs`) and length-normalized (`address_prob_norm`).

Do hallucinated outputs have lower confidence? Let's check:

```python
address_correct_df = df[df["address WRONG"] == 0]
address_wrong_df = df[df["address WRONG"] == 1]

plt.figure(figsize=(10, 6))
plt.hist(address_correct_df['address_prob_norm'], bins=30, alpha=0.5, label='Correct Address', density=True)
plt.hist(address_wrong_df['address_prob_norm'], bins=30, alpha=0.5, label='Wrong Address', density=True)
plt.xlabel('address_prob_norm')
plt.ylabel('Density')
plt.title('Distribution of address_prob_norm')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

Same for the absolute probability:

```python
plt.figure(figsize=(10, 6))
plt.hist(address_correct_df['address_prob_abs'], bins=30, alpha=0.5, label='Correct Address', density=True)
plt.hist(address_wrong_df['address_prob_abs'], bins=30, alpha=0.5, label='Wrong Address', density=True)
plt.xlabel('address_prob_abs')
plt.ylabel('Density')
plt.title('Distribution of address_prob_abs')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

And for title:

```python
title_correct_df = df[df["title WRONG"] == 0]
title_wrong_df = df[df["title WRONG"] == 1]

plt.figure(figsize=(10, 6))
plt.hist(title_correct_df['title_prob_norm'], bins=30, alpha=0.5, label='Correct title', density=True)
plt.hist(title_wrong_df['title_prob_norm'], bins=30, alpha=0.5, label='Wrong title', density=True)
plt.xlabel('title_prob_norm')
plt.ylabel('Density')
plt.title('Distribution of title_prob_norm')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

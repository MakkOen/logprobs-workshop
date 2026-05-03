# The Black Box

We usually treat an LLM as a black box: text goes in, (somehow related) text comes out. And we try to influence what comes out by carefully manipulating (okay, begging and praying over) what goes in.

:::{mermaid}
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 60}}}%%
flowchart LR
    I["`**Input text**`"] --> LLM["`<br/><br/>LLM<br/><br/><br/>`"]
    LLM --> O["`**Output text**`"]

    style LLM fill:#1a1a2e,color:#ffffff,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style I fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style O fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    linkStyle default stroke-width:2.5px
:::

:::{div} text-center
*"Make no mistakes."*
:::

---

## Reality: more goes in

The prompt isn't the only dial you can turn. Models expose sampling parameters that directly shape the output distribution.

:::{mermaid}
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 60}}}%%
flowchart LR
    I["`**Input text**`"] --> LLM["`<br/><br/>LLM<br/><br/><br/>`"]
    T["`*temperature*`"] --> LLM
    K["`*top_k*`"] --> LLM
    P["`*top_p*`"] --> LLM
    LLM --> O["`**Output text**`"]

    style LLM fill:#1a1a2e,color:#ffffff,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style I fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style T fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style K fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style P fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style O fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    linkStyle default stroke-width:2.5px
:::

:::{admonition} What do these do?
:class: dropdown
- **temperature** — controls randomness. Lower = more deterministic, higher = more creative/chaotic.
- **top_k** — limits sampling to the *k* most likely next tokens.
- **top_p** (nucleus sampling) — limits sampling to the smallest set of tokens whose cumulative probability exceeds *p*.
:::

## And more comes out

The model doesn't just return text. It also returns **logprobs**: the log-probabilities it assigned to each token it generated.

:::{mermaid}
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 60}}}%%
flowchart LR
    I["`**Input text**`"] --> LLM["`<br/><br/>LLM<br/><br/><br/>`"]
    T["`*temperature*`"] --> LLM
    K["`*top_k*`"] --> LLM
    P["`*top_p*`"] --> LLM
    LLM --> O["`**Output text**`"]
    LLM --> L["`**logprobs**`"]

    style LLM fill:#1a1a2e,color:#ffffff,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style L fill:#1a6fa8,color:#ffffff,stroke:#444,stroke-width:2.5px,rx:6,ry:6
    style I fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style T fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style K fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style P fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style O fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    linkStyle default stroke-width:2.5px
:::

:::{admonition} Key point of the workshop
:class: important
Logprobs are a window into the model's confidence. Instead of just reading what the model said, we can read *how sure it was* and use that as a signal in our own ML pipelines.
:::

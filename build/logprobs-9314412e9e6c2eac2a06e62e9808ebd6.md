# Logprobs

At every generation step the LLM builds a full probability distribution over its vocabulary and then **samples** from it.

## Token candidates at each step

For every position in the output, the model assigns a probability to every possible next token:

<table style="border:none; margin: 1em 0; border-spacing:0; line-height:1.8; font-size:1.1em">
<tr>
  <td style="border:none; padding:0"></td>
  <td style="border:none; opacity:0.4; padding:0 0.5em; vertical-align:middle">cat</td>
  <td style="border:none; opacity:0.4; padding:0 0.5em; vertical-align:middle">0.05</td>
</tr>
<tr>
  <td style="border:none; font-weight:bold; font-size:1.2em; padding-right:1em; text-align:right; white-space:nowrap">The quick brown fox jumps over the lazy</td>
  <td style="border:none; font-weight:bold; font-size:1.2em; padding:0 0.5em; vertical-align:middle">dog</td>
  <td style="border:none; font-weight:bold; font-size:1.2em; padding:0 0.5em; vertical-align:middle">0.80 ✓</td>
</tr>
<tr>
  <td style="border:none; padding:0"></td>
  <td style="border:none; opacity:0.4; padding:0 0.5em; vertical-align:middle">fox</td>
  <td style="border:none; opacity:0.4; padding:0 0.5em; vertical-align:middle">0.10</td>
</tr>
<tr>
  <td style="border:none; padding:0"></td>
  <td style="border:none; opacity:0.4; padding:0 0.5em; vertical-align:middle">mouse</td>
  <td style="border:none; opacity:0.4; padding:0 0.5em; vertical-align:middle">0.05</td>
</tr>
</table>

:::{admonition} Why log-probabilities?
:class: note dropdown
Probabilities of long sequences get tiny fast (multiplying many small numbers → underflow). Log-probabilities turn multiplication into addition and keep things numerically stable. `logprob = log(prob)`, so `prob = exp(logprob)`. A logprob of `0.0` means 100% confidence; `-∞` means impossible.
:::

## Generation as a branching tree

Each token selection opens a new branch. The model navigates this tree one step at a time:

:::{mermaid}
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 60}}}%%
flowchart TD
    ROOT["Input:<br/><b>The quick brown</b>"]

    ROOT -->|0.05| cat
    ROOT -->|0.10| dog
    ROOT -->|0.80| fox
    ROOT -->|0.05| mouse

    fox -->|0.05| runs
    fox -->|0.75| jumps
    fox -->|0.20| hides

    style ROOT fill:#2d3748,color:#ffffff,stroke:#555,stroke-width:2.5px,rx:6,ry:6
    style fox fill:#1a1a2e,color:#ffffff,stroke:#52b788,stroke-width:2.5px,rx:6,ry:6
    style jumps fill:#1a1a2e,color:#ffffff,stroke:#52b788,stroke-width:2.5px,rx:6,ry:6
    style cat fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6,opacity:0.4
    style dog fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6,opacity:0.4
    style mouse fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6,opacity:0.4
    style runs fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6,opacity:0.4
    style hides fill:#f8f9fa,stroke:#555,stroke-width:2.5px,rx:6,ry:6,opacity:0.4

    linkStyle default stroke-width:2.5px
    linkStyle 2 stroke:#52b788,stroke-width:2.5px
    linkStyle 5 stroke:#52b788,stroke-width:2.5px
:::

The bold green path is the one the model actually took. The faded branches are the roads not travelled, but the model still computed probabilities for all of them.

## What gets exposed

LLMs usually expose not just the selected token but also the **top candidates and their log-probabilities** at every step. This is what the API returns when you ask for `logprobs`.

:::{admonition} Heads up: reasoning models
:class: warning
Not all models and providers expose the logprobs, more on this: [Reasoning Models](../02/reasoning_models.md).
:::

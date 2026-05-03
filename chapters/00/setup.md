# Setup

Open `workshop/playground.ipynb` and run the following cells to verify everything is working before the workshop starts.

## 1. Install dependencies

:::::{tab-set}
::::{tab-item} Colab
:selected:
Run this cell first to install the required packages:

```python
!pip install openai optuna shap gdown scikit-learn tqdm
```
::::
::::{tab-item} Local
If you followed the previous page, all packages are already installed.
::::
:::::

## 2. Imports

```python
import gdown
import json
import math
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
import shap
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
```

## 3. API key & client

:::::{tab-set}
::::{tab-item} Colab
:selected:
Load the key from Colab Secrets (key icon in the left sidebar, name it `OPENAI_API_KEY`):

```python
import os
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
client = OpenAI()
```
::::
::::{tab-item} Local
Set your API key before initializing the client:

```python
import os
os.environ['OPENAI_API_KEY'] = 'YOUR_KEY_HERE'
client = OpenAI()
```
::::
:::::

## 4. Download datasets

:::::{tab-set}
::::{tab-item} Automatic (gdown)
:selected:

```python
import gdown

filtering_url = 'https://drive.google.com/file/d/16He3YIPwqexOMRYiEoW_df9t5mEQID-Q/view?usp=drive_link'
gdown.download(filtering_url, 'filtering_dataset.csv', fuzzy=True, quiet=False)

hallucination_url = 'https://drive.google.com/file/d/172SSOM4vH84Dz_SSY5OPRrtr9i1NT3o7/view?usp=drive_link'
gdown.download(hallucination_url, 'hallucination_dataset.csv', fuzzy=True, quiet=False)
```
::::
::::{tab-item} Manual
Download the CSV files manually and place them in your working directory:

- [filtering_dataset.csv](https://drive.google.com/file/d/16He3YIPwqexOMRYiEoW_df9t5mEQID-Q/view?usp=drive_link)
- [hallucination_dataset.csv](https://drive.google.com/file/d/172SSOM4vH84Dz_SSY5OPRrtr9i1NT3o7/view?usp=drive_link)
::::
:::::

## 5. Sanity check

Run this cell. If it prints a response you're all set:

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello from ML Prague",
)

print(response.output_text)
```

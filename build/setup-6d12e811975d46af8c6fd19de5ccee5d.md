# Setup

Open `workshop/playground.ipynb` and run the following cells to verify everything is working before the workshop starts.

## 1. Install dependencies & load datasets

The first cell installs required packages and downloads the two datasets we'll use during the workshop:

```python
!pip install optuna shap gdown

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

client = OpenAI()

# Download datasets
filtering_url = 'https://drive.google.com/file/d/TODO/view?usp=drive_link'
gdown.download(filtering_url, 'filtering_dataset.csv', quiet=False)

hallucination_url = 'https://drive.google.com/file/d/TODO/view?usp=drive_link'
gdown.download(hallucination_url, 'hallucination_dataset.csv', quiet=False)
```

**If running locally (not Colab):** set your API key before initializing the client:

```python
import os
os.environ['OPENAI_API_KEY'] = 'YOUR_KEY_HERE'
client = OpenAI()
```

**If running on Colab:** load the key from Colab Secrets (key icon in the left sidebar, name it `OPENAI_API_KEY`):

```python
import os
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
client = OpenAI()
```

## 2. Sanity check

Run this cell. If it prints a response you're all set:

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello from ML Prague",
)

print(response.output_text)
```

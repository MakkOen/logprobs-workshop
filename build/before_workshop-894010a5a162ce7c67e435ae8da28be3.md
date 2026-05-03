# Before workshop

1. Clone the logprobs workshop project: [https://github.com/MakkOen/logprobs-workshop](https://github.com/MakkOen/logprobs-workshop) and navigate to the `workshop` folder which contains the following:
    * jupyter notebook `playground.ipynb`
    * `pyproject.toml` and `requirements.txt` to set up the environment with everything we need

2. Set up the Python environment using your tool of choice:
   - **uv**:
     ```bash
     cd workshop
     uv sync
     ```
   - **conda**:
     ```bash
     cd workshop
     conda create --name logprobs-workshop python=3.14 --file requirements.txt
     conda activate logprobs-workshop
     ```
   - **venv**:
     ```bash
     cd workshop
     python3.14 -m venv logprobs-workshop
     source logprobs-workshop/bin/activate   # On Windows: logprobs-workshop\Scripts\activate
     pip install -r requirements.txt
     ```
   - **virtualenv**:
     ```bash
     cd workshop
     virtualenv --python=python3.14 logprobs-workshop
     source logprobs-workshop/bin/activate   # On Windows: logprobs-workshop\Scripts\activate
     pip install -r requirements.txt
     ```

3. Run the testing jupyter notebook:
    ```bash
    # uv:
    uv run jupyter notebook playground.ipynb
    # others:
    jupyter notebook playground.ipynb
    ```

4. Got to [https://pastebin.com/TODO](https://pastebin.com/TODO), password is on the whiteboard, and copy one of the API keys to the notebook:
    ```python
    OPENAI_API_KEY = ...
    ```

## Recommended Alternative:
Open the [google colab notebook](https://colab.research.google.com/drive/1StpfS7yP27syhP7NQBo-aKVJQJcCu1ar?usp=drive_link) and make your own copy (`File -> Save a copy in Drive`)

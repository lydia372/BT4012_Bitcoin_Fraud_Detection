# BT4012 Bitcoin Fraud Detection

Binary classification of illicit Bitcoin transactions using graph-based and tabular features.

## Dataset

Place the following CSV files in the project root directory (same folder as the notebook):

- `train.csv` — labelled transactions (features + `label` column)
- `test.csv` — unlabelled transactions for submission
- `txs_edgelist.csv` — transaction graph edges (`txId1`, `txId2`)
- `sample_submission.csv` — submission template

## Setup

Requires **Python 3.10+**.

```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=bt4012 --display-name "BT4012"
```

## Running the notebook

```bash
# Option A: Jupyter Notebook
jupyter notebook BT4012_bitcoin_experiments.ipynb

# Option B: JupyterLab
jupyter lab BT4012_bitcoin_experiments.ipynb

# Option C: VS Code
# Open the .ipynb file in VS Code, then select the "BT4012" kernel
# from the kernel picker in the top-right corner.
```

When the notebook starts, select the **BT4012** kernel (the one registered in step 4 above).

## Project structure

```
.
├── BT4012_bitcoin_experiments.ipynb   # Main experiment notebook
├── requirements.txt                   # Python dependencies
├── train.csv                          # Training data
├── test.csv                           # Test data
├── txs_edgelist.csv                   # Transaction graph edges
├── sample_submission.csv              # Submission template
└── submission_*.csv                   # Generated submission files
```

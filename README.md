# SpamClassifier

A simple Streamlit app for scam email detection using a Kaggle-style spam dataset.

## Import dataset

Place the Kaggle `spam.csv` (columns `v1` and `v2`) in the project root, or run the helper:

```powershell
python import_dataset.py --source "C:\path\to\spam.csv"
```

If you omit `--source`, the script will try to find `spam.csv` in your Downloads folder.

## Run the app

```powershell
cd c:\SpamClassifier
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```


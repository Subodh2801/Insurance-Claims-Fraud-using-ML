# Insurance Claims Fraud ML

**Project name:** Insurance Claims Fraud ML

## Author & links (this repository)

| | |
|---|---|
| **Name** | Sai Subodh |
| **Email** | saisubodh2812004@gmail.com |
| **GitHub profile** | https://github.com/Subodh2801 |
| **This repo** | https://github.com/Subodh2801/Insurance-Claims-Fraud-using-ML |

Flask app for **fraud detection** and **loss severity** prediction on insurance claims, with training and evaluation views.  
College / team work — acknowledge your group in reports as required by your instructor.

## Requirements

- **Python 3.11** (see `.python-version`)
- Dependencies: `requirements.txt`

## Setup (first time)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

**Windows:** double-click `Start_App.bat`, or:

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Trained models

Model files (`*.pkl` in `trained_models/`) are not stored in Git (see `.gitignore`). After cloning, train or copy models locally:

```bash
python train_and_save_models.py
```

## Add this project to GitHub (first time)

1. Create an **empty** repo on GitHub under your account (no README if you already committed locally).
2. In this folder, connect and push:

```bash
cd "d:\Second Quarter\ML project"
git remote add origin https://github.com/Subodh2801/Insurance-Claims-Fraud-using-ML.git
git branch -M main
git push -u origin main
```

If `origin` already exists: `git remote set-url origin https://github.com/Subodh2801/Insurance-Claims-Fraud-using-ML.git` then `git push -u origin main`.

**Teammates:** use your own GitHub username and repo name in the URL instead.

## Project layout

| Path | Purpose |
|------|--------|
| `app.py` | Flask application |
| `ml_pipeline.py` | Data prep and ML pipeline |
| `train_and_save_models.py` | Train and save `.pkl` models |
| `templates/` | HTML templates |
| `trained_models/` | Saved models (local only; `.pkl` ignored by Git) |

## License

College / team project — set a license if you need one for submission.

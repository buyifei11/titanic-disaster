# Titanic Disaster – Survival Prediction (Python & R with Docker)

This repository predicts which passengers survived the Titanic disaster using
**logistic regression**, implemented twice:

- **Python** (pandas + scikit-learn) in `src/python/`
- **R** (readr + dplyr + base `glm`) in `src/r/`

Each version runs inside its own Docker container. Every data-cleaning,
feature-engineering, modeling and evaluation step is printed to the terminal,
so you only need to read the terminal output.

> The dataset is **not** included in this repository. Step 2 explains how to download it.

---

## Quick start (if you already have Docker and the data)

```bash
git clone https://github.com/buyifei11/titanic-disaster.git
cd titanic-disaster
# put train.csv, test.csv, gender_submission.csv into src/data/
docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

The detailed steps follow below.

---

## Repository structure

```
titanic-disaster/
├── README.md
├── .gitignore              # ignores src/data/* (the dataset) and hidden files
├── CODEOWNERS
└── src/
    ├── data/               # put the Kaggle CSV files here (not tracked by git)
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

---

## Prerequisites

1. **Git** – https://git-scm.com/downloads
2. **Docker Desktop** – https://www.docker.com/products/docker-desktop/
   Open Docker Desktop and wait until it shows **Engine running** before continuing.
   Check that Docker works:
   ```bash
   docker --version
   ```
3. A free **Kaggle account** – https://www.kaggle.com

You do **not** need Python or R installed on your computer; Docker provides them.

---

## Step 1 – Clone the repository

```bash
git clone https://github.com/buyifei11/titanic-disaster.git
cd titanic-disaster
```

**All commands below must be run from this repository root folder** (`titanic-disaster/`).

---

## Step 2 – Download the data

1. Go to https://www.kaggle.com/competitions/titanic/data and sign in.
2. If asked, click **Join Competition** and accept the rules (required before downloading).
3. Click **Download All**. You get `titanic.zip`.
4. Unzip it and move these three files into the `src/data/` folder of this repository:
   - `train.csv` (required)
   - `test.csv` (required)
   - `gender_submission.csv` (optional, used for the test-set comparison)

Alternative using the Kaggle CLI (requires a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

Check that the files are in place:

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

---

## Step 3 – Run the Python version

Build the image (first build takes about 1–3 minutes):

```bash
docker build -t titanic-python src/python
```

Run the container. The `-v` option mounts your local `src/data` folder into the
container, so the script can read the data and write its predictions back:

**macOS / Linux / Git Bash**
```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

**Windows PowerShell**
```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

**Windows Command Prompt (cmd)**
```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

Predictions are written to `src/data/predictions_python.csv`.

---

## Step 4 – Run the R version

Build the image. **The first build can take 5–15 minutes** (longer on Apple Silicon Macs),
because R packages are compiled. Lots of compiler output is normal.

```bash
docker build -t titanic-r src/r
```

Run the container:

**macOS / Linux / Git Bash**
```bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

**Windows PowerShell**
```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

**Windows Command Prompt (cmd)**
```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-r
```

Predictions are written to `src/data/predictions_r.csv`.

---

## What the scripts do (and print)

Both scripts follow the same eight steps and print the result of each:

| Step | Description |
|------|-------------|
| 1 | Load `train.csv`; print its shape, first rows and overall survival rate |
| 2 | Explore: survival rate by sex and by passenger class |
| 3 | Split `train.csv` into 80% training / 20% validation (seed 42) |
| 4 | Clean data: fill missing `Age` and `Fare` with the training median and `Embarked` with the most common port; drop `Cabin` (mostly missing) and `Ticket`; create `Title` (from `Name`), `FamilySize` and `IsAlone`; encode categorical variables |
| 5 | Fit a logistic regression and print its coefficients |
| 6 | Print training and validation accuracy |
| 7 | Clean `test.csv` with the same rules and predict survival |
| 8 | Report test-set accuracy (see note below) |

**Differences between the two models**

- **Python:** features `Pclass, Sex, Age, Fare, FamilySize, IsAlone, Title, Embarked`,
  with one-hot encoding. scikit-learn's L2 regularization keeps coefficients stable.
- **R:** `Pclass` is treated as a categorical factor, and `Sex` is left out because
  `Title` (Mr/Mrs/Miss/Master) already encodes it. Including both made the
  unregularized `glm()` unstable (very large coefficients and standard errors).

**Example Python results:** training accuracy 0.826, validation accuracy 0.838.
The R split differs from the Python split, so its numbers are slightly different.

**Note on test accuracy:** Kaggle's `test.csv` does not contain the `Survived`
column, so the true test accuracy can only be obtained by submitting the
predictions file on the Kaggle competition page. The scripts therefore report:

1. **Validation accuracy** on the 20% of `train.csv` held out from training, and
2. **Test-set accuracy measured against `gender_submission.csv`**, Kaggle's sample
   labels (which predict that all women survive and all men die). This number shows
   how closely the model agrees with that baseline, not its true accuracy.

---

## Package versions

- **Python 3.11** (`python:3.11-slim`): numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1 (see `src/python/requirements.txt`)
- **R 4.4.1** (`rocker/r-ver:4.4.1`): readr 2.1.5, dplyr 1.1.4 (see `src/r/install_packages.R`)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: /app/data/train.csv not found` | The CSV files are not in `src/data/`, or the `docker run` command was not run from the repository root. Run `ls src/data` to check. |
| `Cannot connect to the Docker daemon` | Docker Desktop is not running. Start it, wait for **Engine running**, and retry. |
| `zsh: command not found: docker` | Docker Desktop is not installed, or the terminal was opened before installing it. Install Docker Desktop and open a new terminal. |
| Changes to a script do not show up | Re-run the `docker build` command before `docker run`; the script is copied into the image at build time. |
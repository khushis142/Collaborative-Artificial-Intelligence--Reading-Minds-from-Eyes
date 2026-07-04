# Reading Minds from Eyes: Decoding Cognitive Load with Machine Learning

A machine learning pipeline that detects **cognitive load** from eye-tracking signals — pupil dilation, fixations, blinks, and saccades — using classical ML classifiers, rigorously evaluated across multiple cross-validation strategies.

This project replicates and extends the benchmark set by **Jin et al. (2025)**, exceeding their reported 62.3% accuracy on the hardest classification task while providing a much more thorough (and honest) evaluation of what that accuracy actually means.

## Overview

When people are under mental strain — working memory tasks, visual search, sustained attention — their eyes behave differently: pupils dilate or constrict, fixations lengthen or shorten, blink rate drops, and saccades speed up. This project turns those raw ocular signals into a labeled feature set and trains classifiers to predict the wearer's cognitive state.

```
Raw eye signal → Feature extraction → Exploratory analysis → Modelling → Evaluation
```

**Dataset:** 347,196 sliding-window samples (512-sample windows, 18 columns) from 21 participants, each labeled with one of 7 cognitive states:

| Label | State |
|---|---|
| 0 | Rest (no task) |
| 1–3 | Working Memory (WM), increasing difficulty |
| 4–6 | Visual Attention (VA), increasing difficulty |

**Features per window:** IPA / LHIPA (pupillary activity indices), fixation count and duration, blink rate, saccade speed and peak speed, and pupil diameter.

**Two classification tasks:**
- **Rest vs Load** — is the person under any cognitive load at all? (easy, ~99% accuracy across all models)
- **WM vs VA** — what *type* of load are they under? (hard, the paper's actual benchmark task)

## Key Result

Split strategy alone swings WM-vs-VA accuracy by **10–15 percentage points**. Reporting a single number without saying how the data was split is close to meaningless for this kind of task.

| Split Strategy | SVM | Random Forest | KNN | XGBoost | Logistic Regression |
|---|---|---|---|---|---|
| Global 80/20 (optimistic) | 85.5% | 87.0% | 79.7% | 84.5% | 86.7% |
| Jin et al. per-participant | 71.2% | 79.4% | 72.9% | 80.4% | 72.3% |
| Participant-level (held-out subjects) | 75.0% | 71.7% | 69.7% | 70.0% | 75.2% |
| **LOPO (all 21 participants, averaged)** | **75.1%** | 74.7% | 69.2% | 73.7% | 75.7% |
| Jin et al. (2025) reported benchmark | 62.3% | — | — | — | — |

All strategies beat the published benchmark; the Leave-One-Participant-Out (LOPO) result is the most defensible number, since it tests generalization to every participant exactly once rather than relying on a single random train/test draw.

## Repository Structure

```
.
├── notebooks/
│   ├── 01_exploration.ipynb        # EDA: class balance, NaN checks, feature distributions
│   ├── 02_baselines.ipynb          # Baselines with a global time-based 80/20 split
│   ├── 03_participant_split.ipynb  # Baselines with a held-out-subjects split
│   └── 03_jin_split.ipynb          # Baselines replicating Jin et al.'s per-participant split
├── scripts/
│   └── feature_importance.py       # SHAP / coefficient / permutation importance across all models
├── results/                        # Metrics CSVs, confusion matrices, importance plots
├── data.zip                        # Eye-tracking feature dataset
├── requirements.txt
├── PROJECT_EXPLANATION.md          # Full write-up: methodology, rationale, related work
└── Project_explanation_and_status.docx
```

## Methodology

**1. Data verification** — confirm shape, columns, and integrity before any modelling.

**2. Exploratory analysis** (`01_exploration.ipynb`) — class balance across all 7 states and both binary splits, missing-value checks, and box plots of each feature by state to check discriminability before feature selection.

**3. Baseline modelling** — four classical models (SVM/LinearSVC, Random Forest, KNN, XGBoost), later joined by Logistic Regression as a fifth baseline, trained on both binary tasks under **three different split strategies**:

- **Global 80/20** — sorted by participant then window index, no shuffling. Optimistic: the same participants can appear in both train and test.
- **Jin et al. per-participant** — for each participant and label, first 80% of windows (by index) go to train, last 20% to test. The closest replication of the original paper's protocol; required because the experiment used a block task design (WM → VA → Rest sequentially), so a naive temporal cut per participant would leave entire task blocks out of training.
- **Participant-level (cross-subject)** — 4 of 21 participants held out entirely as test subjects (`numpy.random.default_rng(42)`). Tests generalization to unseen individuals — the realistic deployment scenario.

**4. Leave-One-Participant-Out (LOPO) cross-validation** — every participant serves as the test set exactly once, with results averaged across all 21 folds. This is the most stable, most defensible cross-subject estimate in the project, removing the dependency on a single random held-out split.

**5. Feature importance / interpretability** (`feature_importance.py`) — three methods, matched to model type:
- **SHAP values** for tree-based models (Random Forest, XGBoost)
- **Standardized model coefficients** for linear models (Logistic Regression, SVM)
- **Permutation importance** for KNN, which has no native notion of feature weight

All features are z-scored via `StandardScaler` fit only on training data to avoid leakage; for the participant-level and LOPO splits, the scaler is fit exclusively on training participants.

## Related Work

This project builds directly on:
- **Jin et al. (2025)** — the paper this dataset and benchmark (62.3% SVM accuracy on WM vs VA) come from.
- **Kosch et al. (2023)** — a 579-paper survey validating eye-tracking as a reliable cognitive workload signal.
- **Ekin et al. (2025)** — confirms WM and VA loads produce distinct, but individual-dependent, ocular signatures.
- **Molloy et al. (2026)** — a systematic review confirming SVM and Random Forest as consistently strong choices for this class of problem.

Full citations and paper-by-paper takeaways are in `PROJECT_EXPLANATION.md`.

## Setup

```bash
git clone https://github.com/khushis142/Collaborative-Artificial-Intelligence--Reading-Minds-from-Eyes.git
cd Collaborative-Artificial-Intelligence--Reading-Minds-from-Eyes
pip install -r requirements.txt
unzip data.zip
```

Then run the notebooks in order: `01_exploration.ipynb` → `02_baselines.ipynb` → `03_participant_split.ipynb` / `03_jin_split.ipynb`.

## Authors

Collaborative project by [Khushi Sharma](https://github.com/khushis142) and Avantika Ajit for Collaborative Artificial Intelligence Lab SS26, Universität Stuttgart

See `PROJECT_EXPLANATION.md` for the full methodological write-up, including step-by-step rationale for every design decision.
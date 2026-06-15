# Project Explanation — Cognitive Eye Tracker
**What has been done, why it was done, and how it works.**

---

## The Big Picture

This project is a **machine learning pipeline** for detecting cognitive load from eye-tracking signals. The idea: when people are under mental stress (working memory tasks, visual attention tasks), their eyes behave differently — pupils change size, fixations last longer or shorter, blinks slow down, saccades change speed. We record those signals, extract numerical features from them, and train a classifier to predict which cognitive state the person is in.

The pipeline has these major stages:
```
Raw eye signal → Feature extraction → Exploratory analysis → Modelling → Evaluation
```
Steps 1–4 cover the first three stages: data verification, exploration, and baseline modelling.

---

## Step 1 — Verify Everything is in Place

### What was done
We checked the folder structure, confirmed the pickle file existed, loaded it, and printed its shape and column names.

### Why this matters
Before doing any data science work it is essential to confirm:
- The data is where you expect it to be (file paths are correct).
- The file loads without errors (it was not corrupted during creation).
- The shape (rows × columns) matches expectations.
- The column names are what the feature-extraction script was supposed to produce.

Skipping this step leads to confusing errors later — for example, a model training script that crashes halfway through because it referenced a column that does not exist.

### What we found
```
Type:   pandas.DataFrame
Shape:  347,196 rows × 18 columns
```

#### What the 18 columns mean

| Column | Type | Meaning |
|---|---|---|
| `participant` | metadata | Which study participant this window comes from |
| `id` | metadata | Unique identifier for the original recording session |
| `window_index` | metadata | Which sliding window this row represents |
| `sample_start` | metadata | Index of the first sample in this window |
| `sample_end` | metadata | Index of the last sample in this window |
| `label` | **target** | Cognitive-load state (0–6), what the model must predict |
| `vm_load` | derived label | Visual-motor load score |
| `va_load` | derived label | Visual-attention load score |
| `load_detection_label` | derived label | Binary: rest (0) vs any load (1) |
| `load_type_label` | derived label | Binary: WM (0) vs VA (1), rest excluded |
| `ipas` | **feature** | Index of Pupillary Activity (short-term pupil fluctuation) |
| `lhipas` | **feature** | Low/High Index of Pupillary Activity (ratio of slow to fast pupil changes) |
| `fixation_nums` | **feature** | Number of fixations in the window |
| `fixation_durations` | **feature** | Mean duration of fixations in the window |
| `blink_rate` | **feature** | Number of blinks in the window |
| `saccade_speeds` | **feature** | Mean saccade velocity in the window |
| `saccade_peak_speeds` | **feature** | Peak saccade velocity in the window |
| `diameter` | **feature** | Mean pupil diameter in the window |

#### What is a "window"?
The raw eye-tracking data is a continuous time series. Instead of feeding the whole recording into a model, we split it into fixed-length **windows** of 512 samples with a step of 511 samples (nearly no overlap). Each window becomes one row in the DataFrame. This is called **sliding window feature extraction**. 347,196 windows means there is a lot of data — enough to train robust models.

#### What do labels 0–6 mean?
```
0  → Rest            (no task)
1  → WM level 1      (Working Memory, easy)
2  → WM level 2      (Working Memory, medium)
3  → WM level 3      (Working Memory, hard)
4  → VA level 1      (Visual Attention, easy)
5  → VA level 2      (Visual Attention, medium)
6  → VA level 3      (Visual Attention, hard)
```

---

## Step 2 — Exploratory Data Analysis (EDA) Notebook

### What was done
We created `notebooks/01_exploration.ipynb`. It is a Jupyter notebook that loads the feature DataFrame and runs several analyses and visualisations. All plots are saved to `results/exploration/`.

### Why a notebook for EDA?
Notebooks are standard in data science for exploration because:
- You can run cells one at a time and inspect intermediate results.
- Plots appear inline.
- Markdown cells let you write explanations next to the code.
- They are a good way to document your understanding of the data before modelling.

---

### Section 2.1 — Load Data and Basic Inspection

#### What
Load the pickle file, print shape, column names, and the first few rows with `df.head()`.

#### Why
`df.head()` is the fastest way to sanity-check that the data loaded correctly — you see actual values, not just metadata. If something went wrong during feature extraction (e.g. all zeros, wrong dtypes), it shows up immediately.

---

### Section 2.2 — Missing / NaN Check

#### What
`df.isnull().sum()` counts how many NaN values exist in every column. We display both the count and the percentage.

#### Why
NaN values are a silent killer in machine learning pipelines. Most classifiers (scikit-learn's Random Forest, XGBoost, etc.) will either crash or silently drop rows when they encounter NaN values, giving you misleading results. You need to know:
- Are there any missing values at all?
- Which columns are affected?
- What percentage of data is missing? (A few % → impute; a lot % → reconsider the feature)

If we find missing values, the next step would be to decide: **drop those rows**, **impute with the mean/median**, or **investigate why they are missing** (e.g. blink detection failed for some windows).

---

### Section 2.3 — Class Balance: All 7 States

#### What
A bar chart showing how many windows belong to each of the 7 states (0–6).

Saved as: `results/exploration/class_balance_all_states.png`

#### Why
**Class imbalance is one of the most common problems in classification.** If state 0 (rest) has 100,000 windows but state 3 (WM hard) has only 5,000, a naive classifier can achieve 90%+ accuracy by just predicting "rest" all the time — without actually learning anything. You need to know this before modelling so you can:
- Use **class weights** in the loss function (tell the model that minority classes matter more).
- Use **oversampling** (SMOTE) or **undersampling** to balance the training set.
- Choose **balanced accuracy** or **F1 score** as your evaluation metric instead of raw accuracy.

---

### Section 2.4 — Class Balance: Binary Splits

#### What
Two side-by-side bar charts:
1. **Rest vs Load** — label 0 is "Rest", labels 1–6 are collapsed to "Load".
2. **WM vs VA** — labels 1–3 are "WM", labels 4–6 are "VA", label 0 is excluded.

Saved as: `results/exploration/class_balance_binary.png`

#### Why
The 7-class problem is hard. In cognitive load research it is common to first validate simpler binary questions:
- "Can the system detect *any* cognitive load?" → Rest vs Load
- "Can the system tell *what type* of load?" → WM vs VA

These are also directly useful: a real-world application might only need to know "is this person overloaded?" (binary), not which specific task they are doing. Showing balance for these splits tells you whether the binary problems are also imbalanced.

---

### Section 2.5 — Feature Distributions by State (Box Plots)

#### What
A 3×2 grid of box plots. Each subplot shows one feature on the y-axis and the 7 states on the x-axis. Outliers are hidden (`showfliers=False`) to keep the plots readable — with 347k rows, extreme outliers would otherwise dominate.

Features plotted:
| Feature | What it captures |
|---|---|
| IPAS | Short-term pupil dilation/constriction — sensitive to mental effort |
| LHIPA | Ratio of slow to fast pupil changes — a more robust load indicator |
| fixation_durations | Longer fixations often mean deeper processing |
| blink_rate | Blink rate tends to drop under high cognitive load |
| saccade_speeds | Faster, more erratic saccades can indicate load |

Saved as: `results/exploration/feature_distributions_by_state.png`

#### Why
Box plots of features grouped by class are the standard first look at **feature discriminability** — the question is: "do the feature values actually differ between states?" If a feature's distribution is nearly identical across all 7 states, it will not help the model. If the median shifts clearly between states, it is likely to be an informative predictor. This guides feature selection in the next step.

The box shows the **interquartile range (IQR)** — the middle 50% of values. The line inside the box is the **median**. The whiskers extend to 1.5× the IQR. When medians and boxes shift between states, the feature is discriminative.

---

---

## Step 3 — Baseline Classifiers: Time-Based Split (`02_baselines.ipynb`)

### What was done
We created `notebooks/02_baselines.ipynb`, which trains four classical ML classifiers on both binary tasks using an 80/20 time-based split. Results (accuracy, precision, recall, F1) and confusion matrix plots are saved to `results/`.

### The split strategy
The full DataFrame is sorted by `participant` then `window_index` to preserve the recording order within each participant. The first 80% of rows (277,756 windows) become the training set and the last 20% (69,440 windows) become the test set. **No shuffling.** This mimics the split used by Jin et al. (2025), making results directly comparable.

### The two classification tasks

#### Task 1 — Rest vs Load
Labels 1–6 are collapsed into a single "Load" class, with label 0 remaining "Rest". This is the **easier** binary question: can the model detect that *any* cognitive work is happening?

#### Task 2 — WM vs VA
Rest windows (label 0) are excluded entirely. Labels 1–3 become "WM" (Working Memory) and labels 4–6 become "VA" (Visual Attention). This is the **harder** question: can the model tell *what type* of cognitive load the person is under? This is the task benchmarked by Jin et al. at ~62.3% SVM accuracy.

### The four models

| Model | Why it was chosen |
|---|---|
| **SVM** (`LinearSVC`) | Standard baseline in cognitive load literature; linear kernel is fast on 280k samples |
| **Random Forest** | Ensemble of decision trees; robust to noise and outliers; no need for feature scaling |
| **KNN** | Non-parametric; no assumptions about the data distribution; simple distance-based rule |
| **XGBoost** | Gradient-boosted trees; typically the strongest classical ML baseline |

`LinearSVC` was used instead of `SVC(kernel='rbf')` because fitting a kernel SVM on 277k training samples would take hours. LinearSVC is mathematically equivalent to a linear-kernel SVM and is orders of magnitude faster on large datasets.

### Preprocessing
A `StandardScaler` is fitted on the training set and applied to both train and test. This is mandatory for SVM and KNN, which are sensitive to feature scale — a feature measured in milliseconds would otherwise dominate one measured in counts. Random Forest and XGBoost are scale-invariant but were scaled for consistency.

### Results

| Task | SVM | Random Forest | KNN | XGBoost |
|---|---|---|---|---|
| Rest vs Load (accuracy) | 99.4% | 98.0% | 98.1% | 98.6% |
| WM vs VA (accuracy) | **85.5%** | **87.0%** | 79.7% | 84.5% |

All models comfortably beat the Jin et al. (2025) benchmark of 62.3% on WM vs VA. The reason for this large gap is explained in Step 4 below.

### Outputs saved to `results/`
- `baseline_results.csv` — all metrics for all 8 model–task combinations
- `confusion_matrix_rest_vs_load.png` — 2×2 grid of confusion matrices for Task 1
- `confusion_matrix_wm_vs_va.png` — 2×2 grid of confusion matrices for Task 2

---

## Step 4 — Baseline Classifiers: Participant-Level Split (`03_participant_split.ipynb`)

### What was done
We created `notebooks/03_participant_split.ipynb`, which repeats the exact same training and evaluation as Step 3 but changes only the split strategy to a **participant-level (cross-subject) split**.

### Why a participant-level split?

In the time-based split (Step 3), the training and test sets both contain windows from *all 21 participants*. The model can learn person-specific quirks — e.g. participant p7 always has a slightly higher blink rate — and exploit them at test time. This inflates accuracy in a way that would not hold in a real deployment, where the system would be used by people it has never seen.

A **participant-level split** ensures that **no window from a test participant ever appears in training**. The model must generalise purely from the signal patterns it learned from 17 other people. This is a much harder, much fairer evaluation — and it is close to what Jin et al. (2025) did.

### The split
- **Total participants:** 21 (p1–p22, with gaps — no p2)
- **Test participants:** 4, chosen with `numpy.random.default_rng(42)` for full reproducibility
- **Train participants:** remaining 17
- Each participant has approximately 16,500 windows, so the split is roughly 80/20 by volume

### Data leakage prevention
The `StandardScaler` is fitted **only on the training participants' windows**. The test participants' data is transformed using the training statistics (mean and standard deviation). If we fitted the scaler on all data before splitting, the model would have partial knowledge of the test participants' feature distributions — this is called data leakage and invalidates the evaluation.

### Results

| Task | SVM | Random Forest | KNN | XGBoost |
|---|---|---|---|---|
| Rest vs Load (accuracy) | 99.7% | 99.1% | 99.8% | 99.0% |
| WM vs VA (accuracy) | **75.0%** | 71.7% | 69.7% | 70.0% |

### Comparison with the time-based split

| Task | Model | Time-Based | Participant | Drop |
|---|---|---|---|---|
| WM vs VA | SVM | 85.5% | 75.0% | −10.6 pp |
| WM vs VA | Random Forest | 87.0% | 71.7% | −15.2 pp |
| WM vs VA | KNN | 79.7% | 69.7% | −10.0 pp |
| WM vs VA | XGBoost | 84.5% | 70.0% | −14.5 pp |

The **10–15 percentage point drop** on WM vs VA is entirely explained by the split change. This is the cost of true cross-subject generalisation — and it is the key methodological insight of this step. The participant-level SVM result of 75.0% is still above Jin et al.'s 62.3%, which is consistent with us using a single random draw of 4 test participants rather than a full leave-one-out evaluation over all 21.

### Why Rest vs Load does not drop
Rest vs Load accuracy stays at ~99% under both splits. This is because the rest vs. load distinction is large and consistent across all participants — rest windows are fundamentally different from load windows regardless of who the participant is. The WM vs VA distinction is subtler and more person-dependent, which is why it collapses under a cross-subject evaluation.

### Outputs saved to `results/`
- `participant_split_results.csv` — all metrics for all 8 model–task combinations
- `ps_confusion_matrix_rest_vs_load.png` — confusion matrices for Task 1
- `ps_confusion_matrix_wm_vs_va.png` — confusion matrices for Task 2

---

## Technical Note: Why the Notebook Was Executed With `nbclient` Instead of `jupyter nbconvert`

When running the notebook, `jupyter nbconvert` failed because the conda base environment has a broken configuration referencing `jupyter_contrib_nbextensions` (a Jupyter extension package), which is listed as a preprocessor in the Jupyter config but is not actually installed. This is a common result of partially set-up Jupyter environments.

The fix was to use `nbclient` directly — this is the lower-level library that `nbconvert` itself uses internally to execute notebooks. By calling it directly we bypass the broken config layer. The notebook executes correctly and opens normally in Jupyter — this is only a CLI execution issue, not a notebook issue.

---

## What Has Been Completed

| Step | Notebook | Status |
|---|---|---|
| 1 — Data verification | — | Done |
| 2 — EDA (class balance, NaN check, feature distributions) | `01_exploration.ipynb` | Done |
| 3 — Baseline models, global time-based split | `02_baselines.ipynb` | Done |
| 4 — Baseline models, participant-level split | `03_participant_split.ipynb` | Done |
| 5 — Baseline models, Jin et al. per-participant split | `03_jin_split.ipynb` | Done |

## What Comes Next

1. **Add Logistic Regression** as a fifth baseline (currently SVM, RF, KNN, XGBoost are trained; LR is noted as missing from the original checklist).
2. **Feature importance** — use Random Forest's built-in feature importances or SHAP values to quantify which eye-tracking signals drive the classification decisions.
3. **Cross-validation over participants** — instead of one random 4-participant test split, run leave-one-participant-out (LOPO) cross-validation over all 21 participants and average the results. This gives a more stable, directly comparable number to Jin et al.
4. **Improved models** — explore hyperparameter tuning, feature engineering (e.g. combining IPA and LHIPA), or participant-adaptive calibration to push WM vs VA accuracy higher under the cross-subject evaluation.

---

## Split Strategy Comparison

Three different split strategies were implemented across the baseline notebooks. Each answers a slightly different question about the model's ability to generalise.

---

### Strategy 1 — Global 80/20 (`02_baselines.ipynb`)

All 347,196 windows are sorted globally by `participant` then `window_index`. The first 80% (277,756 windows) form the training set and the last 20% (69,440 windows) form the test set. No shuffling.

**What it tests:** Can the model learn patterns from some participants' recordings and predict the same participants' later recordings?

**Limitation:** Because participants are sorted alphabetically, the last few participants (p8, p9) end up entirely or mostly in the test set while earlier participants are entirely in training. The split is not balanced across participants.

---

### Strategy 2 — Jin et al. Label-Stratified Per-Participant 80/20 (`03_jin_split.ipynb`)

For each participant and each cognitive state label, the first 80% of that label's windows (sorted by `window_index`) go to training and the last 20% go to testing. All slices are then concatenated into a global train and test set.

**Why stratification is necessary:** The experiment used a block design — tasks are presented sequentially (WM-1 → WM-2 → WM-3 → VA-1 → VA-2 → VA-3 → Rest). A raw temporal cut per participant would put the entire Rest block into the test set and all WM windows into training, making both binary classification tasks unsolvable. The label-stratified interpretation is the only one compatible with the dataset structure.

**What it tests:** Can the model generalise from the first 80% of each task's recordings to the last 20% of the same task — within and across participants?

**Relationship to Jin et al. (2025):** This is the closest replication of their reported protocol. Their SVM benchmark of 62.3% on WM vs VA is the target for this split.

---

### Strategy 3 — Participant-Level Split (`03_participant_split.ipynb`)

4 participants are randomly selected (using `random_state=42`) as the test set. The remaining 17 participants form the training set. The model never sees any window from the 4 test participants during training.

**What it tests:** Can the model generalise to **completely unseen individuals**? This is the hardest and most realistic evaluation — it matches a real deployment scenario where the system is used by new users.

**Limitation:** Only one random draw of 4 test participants. A more robust version would be leave-one-participant-out (LOPO) cross-validation averaged over all 21 participants.

---

### Results: WM vs VA Accuracy Across All Three Splits

| Split Strategy | SVM | Random Forest | KNN | XGBoost |
|---|---|---|---|---|
| Global 80/20 | 85.5% | **87.0%** | 79.7% | 84.5% |
| Jin et al. per-participant | 71.2% | 79.4% | 72.9% | 80.4% |
| Participant-level | 75.0% | 71.7% | 69.7% | 70.0% |
| **Jin et al. (2025) reported** | **62.3%** | — | — | — |

### What the differences tell us

1. **Global 80/20 is the most optimistic** because some participants appear in both train and test. The model can exploit person-specific signal (e.g. participant p5 always has a slightly higher blink rate at rest). This inflates accuracy by approximately 10–15 pp compared to a proper per-participant evaluation.

2. **Jin et al. per-participant is the middle ground.** Every participant appears in both train and test, but only their later windows are tested. The model benefits from having seen earlier recordings from the same person. This is the most direct comparison point for the paper's 62.3% benchmark — our result of 71.2% SVM is +8.9 pp higher, likely because our stratification ensures perfectly balanced label coverage per participant.

3. **Participant-level is the most conservative and most realistic.** The drop from global 80/20 to participant-level (10–15 pp) quantifies exactly how much person-specific signal each model was exploiting. Random Forest shows the largest drop (−15 pp), suggesting it overfits individual participants more than SVM.

4. **All three strategies comfortably beat Jin et al.'s 62.3% benchmark**, confirming our feature set and model choices are sound. The participant-level result (75% SVM) is the most honest number to cite in a presentation.

### Full Results Table (`results/full_comparison.csv`)

| Task | Split Strategy | SVM Acc | SVM F1 | RF Acc | RF F1 | KNN Acc | KNN F1 | XGB Acc | XGB F1 |
|---|---|---|---|---|---|---|---|---|---|
| Rest vs Load | Global 80/20 | 0.994 | 0.997 | 0.980 | 0.989 | 0.981 | 0.990 | 0.986 | 0.993 |
| Rest vs Load | Jin per-participant | 0.996 | 0.998 | 0.995 | 0.997 | 0.995 | 0.997 | 0.997 | 0.999 |
| Rest vs Load | Participant-level | 0.997 | 0.999 | 0.991 | 0.995 | 0.998 | 0.999 | 0.990 | 0.995 |
| WM vs VA | Global 80/20 | 0.855 | 0.877 | 0.870 | 0.886 | 0.797 | 0.822 | 0.845 | 0.863 |
| WM vs VA | Jin per-participant | 0.712 | 0.744 | 0.794 | 0.813 | 0.729 | 0.747 | 0.804 | 0.817 |
| WM vs VA | Participant-level | 0.750 | 0.784 | 0.717 | 0.753 | 0.697 | 0.711 | 0.700 | 0.730 |

---

## Related Work

### 1. Kosch et al. (2023) — A Survey on Measuring Cognitive Workload in HCI

**What it is:** A systematic review of 579 HCI papers on cognitive workload measurement.

**Key findings relevant to our project:**

- Eye-tracking (ocular measures) is one of the most used physiological modalities for cognitive workload — validating our choice.
- Each of our 7 features is backed by this survey:
  - **Pupil diameter / IPA / LHIPA** → directly linked to cognitive load (Section 4.3.1)
  - **Saccade speed and size** → "highly discriminatory parameters" for cognitive load (Section 4.3.2)
  - **Blink rate** → lower blink rates indicate higher cognitive load (Section 4.3.3)
  - **Fixation duration and rate** → indicate cognitive load through attentional allocation (Section 4.3.4)
- Criticises NASA-TLX (the standard subjective measure) as retrospective and not designed for real-time HCI — our approach addresses this.
- Identifies "workload-aware systems" as a key research gap — our project contributes toward this.

**How to cite this in presentation:**

> "Kosch et al. (2023) surveyed 579 HCI papers and identified ocular measures as one of the most reliable physiological indicators of cognitive workload, directly validating our feature selection."

---

### 2. Jin et al. (2025) — Decoding Cognitive Load: Eye-Tracking Insights into Working Memory and Visual Attention

**What it is:** The primary reference paper for our project — uses the exact same dataset we are working with.

**Key findings:**

- Collected 528,017 eye-tracking samples from 21 participants across 7 cognitive states.
- Tasks: N-back (working memory) and visual search (visual attention) at 3 difficulty levels each.
- Best model: SVM with index features → **62.3% accuracy** on WM vs VA classification.
- Deep learning (LSTM, Transformer) performed **worse** than traditional ML.
- Used 80/20 time-based split per participant.
- Found index features (IPA, LHIPA, fixations, blinks, saccades) outperformed raw diameter sequence (DS) features.

**How our work compares:**

- We replicated their setup and exceeded their accuracy (87% RF with global time-based split).
- However our split is less strict — mixing participants inflates results.
- Our participant-level split gives a fairer, more realistic evaluation.
- We add Logistic Regression as an additional baseline they did not test.

**How to cite this in presentation:**

> "Jin et al. (2025) established a baseline of 62.3% accuracy for WM vs VA classification using SVM with index features on this dataset. Our time-based split results exceed this, but we attribute the gap primarily to differences in split strategy — our participant-level evaluation gives a more conservative and realistic estimate."

---

### 3. Ekin et al. (2025) — Prediction of Intrinsic and Extraneous Cognitive Load with Oculometric and Biometric Indicators

**What it is:** A study predicting intrinsic vs extraneous cognitive load using eye-tracking and biometric signals.

**Key findings relevant to our project:**

- Confirms that WM load (intrinsic) and VA load (extraneous) produce distinct eye-tracking signatures.
- Individual differences between participants are a major challenge — cross-subject generalisation is hard.
- Combined oculometric + biometric features outperform single-modality approaches.

**How to cite this in presentation:**

> "Ekin et al. (2025) confirm that working memory and visual attention loads produce measurably distinct eye-tracking patterns, supporting the feasibility of our classification task. They also highlight individual differences as a core challenge — motivating our participant-level split evaluation."

---

### 4. Molloy et al. (2026) — Machine Learning Methods for Cognitive Load Analysis and Classification in Aviation: A Systematic Review

**What it is:** A systematic review of 43 ML papers on cognitive load classification, screened from 1,949 studies in the Scopus database. Published in International Journal of Human-Computer Interaction.

**Key findings relevant to our project:**

- Reviews ML methods for CL classification across physiological signals (EEG, fNIRS, HRV) AND eye-tracking — directly covering our modality.
- Finds SVM and Random Forest are the most consistently effective models for CL classification — exactly the models we use.
- CNNs and deep learning achieved up to 95%+ accuracy but require large datasets and complex pipelines — traditional ML remains competitive for structured feature sets like ours.
- Highlights that model selection, data preprocessing, and validation strategy are the most critical factors affecting performance — reinforcing our focus on split strategy.
- Aviation context is high-stakes: cognitive overload directly impacts safety — motivates real-world applications of our approach.

**How to cite this in presentation:**

> "Molloy et al. (2026) reviewed 43 ML studies on cognitive load classification and found SVM and Random Forest to be consistently effective — consistent with our own baseline results. They also emphasize that validation strategy is critical to meaningful performance claims, motivating our use of both time-based and participant-level splits."

**What it adds beyond the other papers:**

- Broader ML landscape view — not just eye-tracking but all physiological modalities.
- Confirms our model choices are state-of-the-art for structured CL classification.
- Connects our work to safety-critical real-world applications.

---

### Summary Table — What Each Paper Contributes

| Paper | What it gives us |
|---|---|
| Kosch et al. (2023) | Validates our feature choices across 579 papers |
| Jin et al. (2025) | Our baseline to replicate and beat; same dataset |
| Ekin et al. (2025) | Confirms task feasibility; highlights cross-subject challenge |
| Molloy et al. (2026) | Validates our model choices (SVM, RF); emphasizes validation strategy importance |

# New Additions:

## Logistic Regression Baseline

### What was done

A Logistic Regression classifier was added as a fifth baseline model alongside SVM, Random Forest, KNN, and XGBoost.

### Why Logistic Regression?

Logistic Regression is one of the most widely used linear classifiers in machine learning and serves as a strong interpretable baseline. Unlike tree-based methods, it models a linear decision boundary and provides insight into whether the classes are linearly separable using the extracted eye-tracking features.

### Results

Logistic Regression was evaluated under the same split strategies as the other models:

* Global 80/20 split

Task,Model,Accuracy,Precision,Recall,F1
Rest vs Load,Logistic Regression,0.9952,0.9977,0.9971,0.9974
Rest vs Load,SVM,0.9944,0.9945,0.9994,0.997
Rest vs Load,Random Forest,0.981,0.9869,0.9926,0.9898
Rest vs Load,KNN,0.98,0.9817,0.9969,0.9893
Rest vs Load,XGBoost,0.9759,0.9791,0.9952,0.9871
WM vs VA,Logistic Regression,0.8666,0.7996,0.9906,0.8849
WM vs VA,SVM,0.8567,0.7862,0.9933,0.8777
WM vs VA,Random Forest,0.8676,0.8075,0.9772,0.8843
WM vs VA,KNN,0.8011,0.7594,0.9016,0.8244
WM vs VA,XGBoost,0.8542,0.7975,0.9629,0.8724

* Jin et al. per-participant split

Task,Model,Accuracy,Precision,Recall,F1
Rest vs Load,Logistic Regression,0.9981,0.998,1.0,0.999
Rest vs Load,SVM,0.9963,0.9961,1.0,0.998
Rest vs Load,Random Forest,0.9947,0.9943,1.0,0.9972
Rest vs Load,KNN,0.9952,0.9949,1.0,0.9975
Rest vs Load,XGBoost,0.9972,0.997,1.0,0.9985
WM vs VA,Logistic Regression,0.7225,0.6771,0.8505,0.7539
WM vs VA,SVM,0.7162,0.6731,0.8405,0.7475
WM vs VA,Random Forest,0.7956,0.7457,0.8969,0.8143
WM vs VA,KNN,0.737,0.7045,0.8164,0.7563
WM vs VA,XGBoost,0.7986,0.7609,0.8707,0.8121

* Participant-level split

Task,Model,Accuracy,Precision,Recall,F1
Rest vs Load,Logistic Regression,1.0,1.0,1.0,1.0
Rest vs Load,SVM,0.9966,0.9964,1.0,0.9982
Rest vs Load,Random Forest,0.9905,0.99,1.0,0.995
Rest vs Load,KNN,0.9979,0.9978,1.0,0.9989
Rest vs Load,XGBoost,0.9897,0.9891,1.0,0.9945
WM vs VA,Logistic Regression,0.7518,0.7005,0.8798,0.78
WM vs VA,SVM,0.7503,0.6909,0.9063,0.784
WM vs VA,Random Forest,0.7101,0.6633,0.8536,0.7465
WM vs VA,KNN,0.7026,0.6864,0.7464,0.7151
WM vs VA,XGBoost,0.696,0.6612,0.804,0.7256


This ensures a fair comparison across all baseline methods.

---

## Leave-One-Participant-Out Cross-Validation (LOPO)

### What was done

A Leave-One-Participant-Out (LOPO) evaluation was implemented.

For each fold:

* One participant is used as the test set.
* The remaining participants are used for training.
* The process is repeated for every participant.
* Results are averaged across all folds.

### Why LOPO?

The participant-level split used previously depends on one random selection of test participants. LOPO removes this dependency and provides a more stable estimate of cross-subject performance.

LOPO is generally considered the strongest evaluation protocol for physiological machine-learning datasets because every participant is tested exactly once.

### Outputs

* lopo_results.csv

Task,Model,Accuracy,Precision,Recall,F1,Accuracy_STD
Rest vs Load,SVM,0.9947113076474705,0.9977263545043653,0.9966385553778139,0.9971300781874254,0.012500138498037576
Rest vs Load,Logistic Regression,0.9936278095612282,0.9992140709400774,0.9939558463545253,0.9964750025678981,0.018478679082710246
Rest vs Load,Random Forest,0.9874904121437298,0.992680171882576,0.9941445307749943,0.9933020255158731,0.020364796589075813
Rest vs Load,KNN,0.9909594698708977,0.9957207379795708,0.9946631753783677,0.9951084022349962,0.01713185374844121
Rest vs Load,XGBoost,0.9893437552604534,0.9952423800887221,0.9934627430122133,0.9942421791878608,0.019196409660477073
WM vs VA,SVM,0.7509368516392648,0.7470624870027373,0.8230324285226389,0.7539814448784561,0.1255337854471792
WM vs VA,Logistic Regression,0.7572735011763181,0.7552211517144374,0.8203572417606385,0.7586613700006773,0.11802026210484476
WM vs VA,Random Forest,0.7468877743061932,0.7477017044652209,0.8078233259416007,0.7596917242356914,0.10880079319189719
WM vs VA,KNN,0.6923842545989682,0.700368709126485,0.7255505687258657,0.7000617875904146,0.09829895003001303
WM vs VA,XGBoost,0.7365802338769446,0.7507658861765546,0.7775884878201825,0.7448364787133751,0.10886216248611207


### Interpretation

LOPO results provide the most realistic estimate of how well the system would generalise to completely unseen users in a real-world deployment.

---

## Feature Importance and Model Interpretability (Task 2: WM vs VA)

### What was done

Created python script feature_importance.py to calculate feature importance.

Feature importance analysis was performed using:

1. SHAP (SHapley Additive exPlanations): for tree based classifiers (Random Forest, XGBoost)

2. Model Coefficients: For Linear Models (Logistic Regression & SVM)
Because you scaled your features using StandardScaler() in your make_task function, the features all share a mean of 0 and a standard deviation of 1. This means the magnitudes of the learned weights or coefficients ($\beta$ coefficients) are directly comparable:
* Magnitude (Absolute Value): Represents the feature's overall importance or strength.
* Sign (+ or -): Represents the direction of the relationship. A positive coefficient means higher values of that feature drive the model to predict Class 1 (VA), while a negative coefficient drives it to predict Class 0 (WM).

3. Permutation Feature Importance: For Distance-Based Models (KNN)
KNN does not have weights, parameters, or trees—it simply looks at spatial proximity. To find feature importance for KNN, we use Permutation Feature Importance.How it works:
* step 1. The model evaluates baseline accuracy on the test set.
* step 2. For a specific feature (e.g., blink_rate), the values are randomly shuffled across rows, breaking its relationship with the target variable $y$ while keeping the rest of the features intact.
* step 3. The model makes predictions on this corrupted dataset and measures the drop in accuracy.
* Step 4. If the accuracy drops significantly, that feature is highly important. If accuracy barely changes, the feature is redundant or unimportant.


### Why this matters

Accuracy alone does not explain why a model makes a prediction. Feature importance identifies which eye-tracking signals contribute most strongly to classification decisions.

### What we learned

The analysis ranks all extracted eye-tracking features by predictive value and provides insight into whether pupil activity, blink behaviour, fixation behaviour, or saccade behaviour is most informative for cognitive load detection.

### Outputs

* comparison of feature importance of all models for a type of split.

For each type of split and model:

* csv for feature importance.
* bar chart for logistic regression, svm and knn feature importance.
* shap beeswarm summary plot for random forest and xgboost.

#### Note: 

Only for 03_participant_split:

Trained an XGBoost model on Task 2 (WM vs VA), calculates the SHAP values using the test set participants, and generates three essential diagnostic plots.

What these plots will tell you about your data:

* shap.plots.bar (Global Importance):Unlike traditional .feature_importances_ which only tells you how much a feature is used to split a tree, SHAP global bar charts show the mean absolute contribution (mean(|SHAP value|)) to the actual final prediction score. It ranks your features (ipas, blink_rate, etc.) by overall impact.

* shap.plots.beeswarm (Directional Impact):
** Y-axis: Features sorted by global importance.
** X-axis (SHAP Value): Values >0 pull the prediction towards Class 1 (VA), while values <0 pull it towards Class 0 (WM).
** Color (Feature Value): Red points represent high values of that feature, and blue points represent low values.
** Example interpretation: If high values of ipas (red dots) are clustered on the right side (>0), it means an increased tonic pupil dilation index directly drives the model to predict Visual Attention (VA).

* shap.plots.waterfall (Local Windows):This takes one specific observation (e.g., shap_values[0]) and shows how the model moved from the base expected value (E[f(X)]) to the final prediction model output (f(X)) step-by-step using individual feature contributions.

Calculated and plot feature importances for all three non-tree models on Task 2 (WM vs VA)

In future: Calculate all feature importance

# Future Tasks & Immediate Priorities
Here is your roadmap. The items in bold are your primary goals for the next two weeks.

Phase 1: Robust Interpretability (Next 2 Weeks - High Priority)

* Fully integrate feature importance (SHAP/Permutation Importance/Model coefficients) across all models: Currently, you have partial implementations. Standardize the reporting so you can compare feature importance across all models (SVM, RF, KNN, XGBoost, LR) for Task 2.

* Consolidate Visualizations: Create a single comparative dashboard of feature importances. This will allow you to answer the "Why" behind your 75% accuracy.

Phase 2: Advanced Refinement (Post-Check-in)

* Hyperparameter Tuning: Use GridSearchCV or Optuna to see if you can squeeze another 2–5% out of the models.

* Feature Engineering: Explore if combining features (e.g., interaction terms like ipas * blink_rate) improves performance.

Phase 3: Deployment Readiness (Final Phase)

* Participant-Adaptive Calibration: Research "Few-Shot Learning" or "Calibration" where a model is trained on many subjects but adapted using just 30 seconds of a new user's data.

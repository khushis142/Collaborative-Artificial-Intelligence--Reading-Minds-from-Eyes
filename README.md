# Reading Minds from Eyes: Decoding Cognitive Load from Eye-Tracking Signals with Machine Learning

Collaborative Artificial Intelligence Group, University of Stuttgart
Authors: Avantika Ajit, Khushi Shah
Supervisor: Dr. Xiaofu Jin

## What this project does

When a person's cognitive load goes up, their eyes behave differently: pupils change size, fixations last longer or shorter, blink rate drops, and saccades speed up or slow down. This project builds a machine learning pipeline that takes raw eye-tracking recordings, turns them into numerical features, and trains classifiers to predict which cognitive state a person is in from those features alone.

The pipeline follows this path:

```
Raw eye signal -> Feature extraction -> Exploratory analysis -> Modelling -> Evaluation -> Interpretability
```

Two questions drive the project:

1. Can a model tell whether someone is at rest or under cognitive load (**Rest vs Load**)?
2. Can a model tell what *type* of load someone is under, working memory or visual attention (**WM vs VA**)?

The second question is the harder and more interesting one, and it is the main benchmark used throughout this repository. It is also the task reported by Jin et al. (2025), whose 62.3% SVM accuracy is the number we compare all our results against.

## Dataset

- 528,017 raw eye-tracking samples from 21 participants (ids p1 to p22, no p2).
- 7 cognitive states: rest, working memory (3 difficulty levels), visual attention (3 difficulty levels).
- Recorded during an N-back task (working memory) and a visual search task (visual attention).
- Raw samples were cut into sliding windows of 512 samples with a step of 511 (near-zero overlap), producing **347,196 windows**, each described by 8 engineered features plus participant and label metadata.

### Features extracted per window

| Feature | What it captures |
|---|---|
| `ipas` | Index of Pupillary Activity, short-term pupil fluctuation |
| `lhipas` | Ratio of slow to fast pupil changes, a more robust load indicator |
| `fixation_nums` | Number of fixations in the window |
| `fixation_durations` | Mean fixation duration |
| `blink_rate` | Number of blinks in the window |
| `saccade_speeds` | Mean saccade velocity |
| `saccade_peak_speeds` | Peak saccade velocity |
| `diameter` | Mean pupil diameter |

### Labels

```
0  Rest
1-3  Working Memory (easy, medium, hard)
4-6  Visual Attention (easy, medium, hard)
```

These are collapsed into two binary tasks used for classification:

- **Task 1: Rest vs Load** — label 0 vs labels 1-6.
- **Task 2: WM vs VA** — rest excluded, labels 1-3 vs labels 4-6.

## Models

Five classical machine learning models are trained and compared on both tasks:

- Logistic Regression
- Linear SVM (`LinearSVC`, chosen over an RBF kernel purely for speed on ~280k training rows)
- Random Forest
- K-Nearest Neighbours
- XGBoost

All features are standardised with `StandardScaler`, fitted on the training set only and applied to the test set, to avoid leaking test statistics into training.

## Evaluation strategies

Accuracy on this kind of data depends heavily on *how* you split train and test, so four different strategies were implemented and compared:

| Split strategy | What it tests | Best WM vs VA accuracy |
|---|---|---|
| Global 80/20 (time-ordered) | Same participants in train and test | 87.0% (Random Forest) |
| Jin et al. per-participant split | Same participants, later windows held out | 80.4% (XGBoost) |
| Participant-level split (4 held-out participants) | Fully unseen individuals | 75.2% (Logistic Regression) |
| Leave-One-Participant-Out (LOPO, 21 folds) | Fully unseen individuals, averaged over everyone | 75.7% +/- 11.8% (Logistic Regression) |
| Jin et al. (2025) reported benchmark | — | 62.3% (SVM) |

**Every model, under every split strategy, beats the published 62.3% benchmark.** The more cross-subject-realistic the split, the lower and more honest the accuracy becomes: the 10-15 percentage point drop between the global split and the participant-level split quantifies how much person-specific signal the models were exploiting when participants leaked across train and test. LOPO is the strategy we consider the most trustworthy, since it tests generalisation to every participant exactly once instead of relying on a single random draw of held-out people.

Rest vs Load stays around 99% under every split, because rest and load are physiologically distinct regardless of who the person is. WM vs VA is the harder, more person-dependent distinction, which is why it is the one that drops.

## Interpretability

Feature importance for Task 2 (WM vs VA) was computed with a method appropriate to each model type:

- **SHAP** for the tree-based models (Random Forest, XGBoost).
- **Standardised model coefficients** for the linear models (Logistic Regression, SVM), which are directly comparable because all features share the same scale.
- **Permutation importance** for KNN, which has no internal weights to inspect.

Across all five models and every split strategy, the ranking is consistent:

1. `fixation_durations` — by far the strongest predictor.
2. `saccade_speeds` and `saccade_peak_speeds` — consistently the next most important.
3. `blink_rate` and `fixation_nums` — moderate contribution.
4. `ipas` and `lhipas` — consistently the least discriminative features for this task.

## Hyperparameter tuning

Optuna was used to tune all five models, with the search nested inside `GroupKFold` (grouped by participant) so that tuning never mixes windows from the same person across the inner train/validation split. Tuning was **not** run for LOPO: doing it properly would require re-running the search inside every one of the 21 folds, which is computationally prohibitive, and doing it cheaply (tuning once, reusing the result) would quietly reintroduce the same cross-subject leakage that LOPO exists to avoid.

Tuning results on the participant-level split:

| Model | Default Acc | Tuned Acc | Gain |
|---|---|---|---|
| Logistic Regression | 75.18% | 75.28% | +0.10 pp |
| SVM | 75.03% | 75.04% | +0.01 pp |
| Random Forest | 71.01% | 71.54% | +0.53 pp |
| KNN | 70.26% | 72.24% | +1.98 pp |
| XGBoost | 69.60% | 71.82% | +2.22 pp |

The gains are small everywhere, and Logistic Regression, already the best default model, barely moves. This suggests the current feature set has a real performance ceiling around 75% for cross-subject WM vs VA classification, and that pushing past it will need richer features or participant-adaptive calibration rather than more tuning.



## Key takeaways

- 75% (LOPO) is likely close to the practical ceiling for this feature set.
- Logistic Regression outperforming the tree-based models suggests the WM vs VA decision boundary is close to linear in this feature space.
- Our results (75.7% LOPO) substantially exceed the 62.3% reported by Jin et al. (2025); we do not have access to their exact split implementation, so we treat this as most likely due to differences in evaluation protocol rather than a stronger model.
- LOPO variance is high (+/- 11.8 pp): per-participant accuracy ranges from 52.3% (near chance, for one participant) to 91.2% (for another), showing that the WM/VA signal itself is far more separable for some individuals than others.
- Only classical machine learning was evaluated here. Per Jin et al., deep learning (LSTM, Transformer) underperforms classical ML on this dataset, which is why it was not a priority for this project.

## References

- Kosch et al. (2023), *A survey on measuring cognitive workload in human-computer interaction*, ACM Computing Surveys.
- Jin et al. (2025), *Decoding Cognitive Load: Eye-Tracking Insights into Working Memory and Visual Attention*, ETRA '25. Primary reference and source of the 62.3% benchmark.
- Ekin et al. (2025), *Prediction of Intrinsic and Extraneous Cognitive Load with Oculometric and Biometric Indicators*, Scientific Reports.
- Molloy et al. (2026), *Machine Learning Methods for Cognitive Load Analysis and Classification in Aviation: A Systematic Review*, International Journal of Human-Computer Interaction.

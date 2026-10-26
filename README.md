# Task 3 – Netflix Audience Rating Classification

Multi-class classification: predict the **audience rating category** of a Netflix title (Kids, Older Kids, Teens, Adults) from its content attributes.

## Method
1. **Analyze rating categories**: distribution of the raw rating labels, then grouping into 4 audience categories (rare and unrated labels handled)
2. **Prepare the training dataset**: features (type, duration, release year, date added, main country, director known, number of genres, genres), One-Hot encoding, standardization, multi-hot genres, all inside a scikit-learn Pipeline, stratified 80% / 20% split
3. **Train classification models**: Baseline (most frequent class), Logistic Regression, Decision Tree, Random Forest
4. **Optimize model performance**: hyperparameter tuning with `GridSearchCV` (Decision Tree) and `RandomizedSearchCV` (Random Forest), 5-fold cross-validation, scored on macro F1
5. **Evaluate prediction accuracy**: accuracy, macro F1, weighted F1, per-class report, confusion matrices, feature importance

## Rating groups
| Category | Raw ratings |
|---|---|
| Kids | TV-Y, TV-G, G |
| Older Kids | TV-Y7, TV-Y7-FV, TV-PG, PG |
| Teens | TV-14, PG-13 |
| Adults | TV-MA, R, NC-17 |

Unrated or missing labels (NR, UR, NaN) are removed.

## Key points
- The classes are imbalanced (Adults and Teens dominate), so a "most frequent class" baseline already scores well. Models are compared mainly on **macro F1**, not accuracy alone.
- `rating` is the target and is never used as a feature.
- Tuning is done on the training set only, so the test scores are an honest estimate.

## Results (test set, 20%)
8,705 titles after cleaning (82 unrated titles removed). Class shares: Adults 46%, Teens 30%, Older Kids 17%, Kids 6.5%.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline (most frequent) | 0.460 | 0.158 | 0.290 |
| Logistic Regression | 0.586 | 0.534 | 0.588 |
| Decision Tree | 0.539 | 0.516 | 0.541 |
| Random Forest | 0.624 | 0.558 | 0.610 |
| Decision Tree (tuned) | 0.550 | 0.522 | 0.555 |
| **Random Forest (tuned)** | **0.626** | **0.578** | **0.621** |

Best parameters: Random Forest `n_estimators=500, max_depth=20, min_samples_leaf=2, max_features="sqrt"`; Decision Tree `criterion="gini", max_depth=10, min_samples_leaf=1`.

Per-class F1 of the best model: Adults 0.737, Kids 0.563, Teens 0.548, Older Kids 0.462.

## Analysis
- The tuned Random Forest reaches **62.6% accuracy**, +16.6 points over the baseline (46%). Tuning improved its macro F1 by +0.020.
- Random Forest beats a single Decision Tree, as expected: averaging many trees reduces overfitting.
- Adults is predicted best (F1 0.737), which is likely helped by it being the largest class. Older Kids is the hardest category (F1 0.462); the confusion matrices in the notebook show where its titles end up.
- The ceiling is modest because the dataset only has coarse attributes (genre, country, year, type). What really determines a rating (violence, language, themes) is not in the data.

## Run the project
Put `Dataset.csv` in the same folder as the notebook, then:
```bash
pip install -r requirements_task3.txt
jupyter notebook task3_rating_classification.ipynb
```

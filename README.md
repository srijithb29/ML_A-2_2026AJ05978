# Adult Income Classification: Five ML Models and a Streamlit App
Submission by: Srijith B
BITS ID: 2026aj05978
## Problem Statement
This project predicts whether an adult's annual income is greater than $50K using demographic and employment attributes from the UCI Adult dataset. Five machine-learning models are trained and compared using six evaluation metrics.

## Dataset
- Source: [UCI Machine Learning Repository, Adult dataset (ID 2)](https://archive.ics.uci.edu/dataset/2/adult).
- The dataset contains 48,842 records: 32,561 in `adult.data` for training and 16,281 in `adult.test` for testing.
- The 14 predictors cover age, work, education, family, demographic, and country attributes. The target is `income` (`<=50K` or `>50K`).
- `test_data.csv` contains a reproducible 5,000-row sample from the official test split, selected with `random_state=42`.
- The source uses `?` for unknown values. Numeric fields are median-imputed and standardized; categorical fields are filled with their most frequent value and one-hot encoded.

## Project Links
GitHub repository: https://github.com/srijithb29/ML_A-2_2026AJ05978

Streamlit entry point: `streamlit_app.py`. Redeploy the app to publish the Adult version.

## Models and Results

### Comparison Table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Support Vector Machine |     0.8470 | 0.9001 |      0.7993 |   0.7502 | 0.7695 | 0.5473 |
| Logistic Regression    |     0.8442 | 0.9013 |      0.7926 |   0.7507 | 0.7677 | 0.5417 |
| KNN                    |     0.8350 | 0.8515 |      0.7732 |   0.7549 | 0.7632 | 0.5278 |
| Decision Tree          |     0.8136 | 0.7395 |      0.7412 |   0.7395 | 0.7403 | 0.4807 |
| Naive Bayes            |     0.5490 | 0.7616 |      0.6458 |   0.6827 | 0.5437 | 0.3264 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Scores 0.8442 accuracy and the highest AUC (0.9013). It identifies the lower-income class well, but recall for `>50K` is 0.57. |
| Decision Tree | Reaches 0.8136 accuracy and 0.7403 macro F1. It performs below the linear models but is straightforward to interpret. |
| KNN | Achieves 0.8350 accuracy and 0.7632 macro F1. Recall is more balanced across the two income classes than for Logistic Regression. |
| Naive Bayes | Has the lowest accuracy (0.5490) and macro F1 (0.5437). Its 0.94 recall for `>50K` comes with lower recall for `<=50K` (0.43). |
| Support Vector Machine | Has the highest accuracy (0.8470) and macro F1 (0.7695). Like Logistic Regression, it detects the `<=50K` class more reliably than `>50K`. |

**Overall winner by accuracy: Support Vector Machine.** It reaches 0.8470 accuracy, while Logistic Regression has the highest AUC at 0.9013. The test sample is class-imbalanced, so the lower `>50K` recall is worth noting alongside accuracy.

## Repository Structure
```
├── streamlit_app.py
├── train_models.py
├── requirements.txt
├── README.md
├── adult/                      # optional local UCI files; downloaded if missing
├── test_data.csv
├── adult_classification_results.csv
├── confusion_matrices.png
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── label_info.json
```

## Running the Project
```
pip install -r requirements.txt
python train_models.py      # downloads Adult data if needed, trains models, and saves artifacts
python -m streamlit run streamlit_app.py
```

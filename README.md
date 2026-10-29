# Customer Churn Prediction & Retention Analytics

## Project Overview
This project demonstrates a machine learning workflow for predicting which subscription customers may leave a service. It compares Logistic Regression and Random Forest models and evaluates their classification performance.

## Business Question
Can customer account and service attributes help identify customers who may be at risk of leaving, so a business can investigate potential retention opportunities?

## Dataset
The file `data/customer_churn.csv` contains 1,200 synthetic customer records created for portfolio demonstration. It is not actual customer data, and the results do not establish real-world predictive performance.

## Tools & Technologies
- Python
- pandas
- scikit-learn
- Machine learning classification
- Model evaluation and visualization

## Project Workflow
1. Load and validate the dataset.
2. Prepare features and the target variable.
3. Build a preprocessing pipeline.
4. Split the data into stratified training and test sets.
5. Train Logistic Regression and Random Forest models.
6. Compare accuracy, precision, recall, F1-score, and ROC-AUC.
7. Export test predictions, evaluation metrics, and visualizations.

## Model Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 72.08% | 72.08% |
| Precision | 67.78% | 67.39% |
| Recall | 61.62% | 62.63% |
| F1-score | 64.55% | 64.92% |
| ROC-AUC | 0.810 | 0.779 |

These metrics were obtained from the project's held-out test set.

## Key Findings
- Both models achieved the same test accuracy of approximately 72.08%.
- Logistic Regression achieved higher precision and ROC-AUC in this run.
- Random Forest achieved slightly higher recall and F1-score.
- The better model depends on the business objective and the relative cost of missed at-risk customers versus unnecessary retention outreach.

Because the dataset is synthetic, these results demonstrate the modeling workflow only. They should not be interpreted as evidence of performance on real customers.

## Generated Outputs

### Model Evaluation
- `outputs/model_metrics.csv`
- `outputs/test_predictions.csv`
- `outputs/confusion_matrix.png`

### Feature Analysis
- `outputs/feature_importance.png`

The script may generate additional evaluation reports depending on the files included in the project.

## How to Run
From the project directory, run:

```bash
python -m pip install -r requirements.txt
python src/train_model.py
```

Python must be installed on your computer. These commands install the dependencies and run model training and evaluation.

## Responsible Use and Limitations
A real deployment would require validation on representative future data, monitoring for data drift, privacy review, and careful selection of decision thresholds. Any retention intervention should be evaluated to determine whether it actually helps customers.

Risk scores should not be used to automatically penalize customers or deny them services.

## Future Improvements
- Evaluate cross-validation and calibration.
- Tune classification thresholds based on business costs.
- Investigate model errors and potential data leakage.
- Validate the workflow on an appropriate real-world dataset.
- Develop a monitoring and reporting dashboard.
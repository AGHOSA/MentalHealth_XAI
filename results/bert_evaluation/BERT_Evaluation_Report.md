# BERT Model Evaluation Report

**Project:** Mental Health Classification Through Social Media  
**Evaluation date:** 2026-10-05  
**Model:** bert-base-uncased fine-tuned on Reddit dataset  
**Evaluation device:** cuda

## 1. Objective

The objective of this evaluation is to measure the performance of a
fine-tuned BERT text-classification model on Reddit text labelled as
depression or non-depression.

This is a text-classification experiment, not a clinical diagnostic test.

## 2. Experimental Setup

- Model: `bert-base-uncased`, fine-tuned for binary classification
- Dataset size: 1,840 records
- Training split: 1,472 records (80%)
- Validation split: 184 records (10%)
- Test split: 184 records (10%)
- Random seed: 42
- Evaluation device: cuda

The evaluation script reproduces the stratified train-validation-test
split using the same seed and split procedure as the training script.
This assumes the original CSV content and row order have not changed.

## 3. Evaluation Metrics

| Metric | Result |
|---|---:|
| Accuracy | 94.02% |
| Balanced accuracy | 93.13% |
| Macro precision | 92.69% |
| Macro recall | 93.13% |
| Macro F1-score | 92.90% |
| ROC-AUC | 98.11% |

## 4. Confusion Matrix

Rows represent actual labels; columns represent predicted labels.

| Actual class | Predicted non-depression | Predicted depression |
|---|---:|---:|
| Non-depression | 50 | 5 |
| Depression | 6 | 123 |

The model correctly classified 173 of 184
test samples and incorrectly classified 11 samples.

## 5. Class-wise Classification Report

See `classification_report.csv` for the complete precision, recall,
F1-score, and support values for each class.

## 6. Visualizations

- `confusion_matrix.png`: visual comparison of actual and predicted labels.
- `roc_curve.png`: ROC curve showing the relationship between the
  true-positive rate and false-positive rate across thresholds.

## 7. Interpretation

The test accuracy was 94.02%, and the macro F1-score
was 92.90%. The ROC-AUC was 98.11%.

Macro-averaged metrics give equal weight to both classes, while balanced
accuracy averages recall across the classes. These metrics complement
accuracy when class counts are unequal.

The confusion matrix shows 5 non-depression samples predicted as
depression and 6 depression samples predicted as non-depression.

## 8. Limitations

1. The test set contains only 184 samples.
2. Results depend on the dataset's labelling quality and representativeness.
3. Reddit text may contain slang, sarcasm, contextual ambiguity, and
   other language patterns that affect predictions.
4. The evaluation does not establish performance on new platforms or
   unseen populations.
5. The model must not be used as a standalone mental-health diagnosis
   tool. Predictions should not replace assessment by qualified
   mental-health professionals.
6. The reported performance should be confirmed against an independently
   preserved test split if exact split reproducibility cannot be verified.

## 9. Output Files

- `metrics.json`
- `classification_report.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `test_predictions.csv`

## 10. Conclusion

The fine-tuned BERT model achieved 94.02% accuracy,
92.90% macro F1-score, and 98.11%
ROC-AUC on the evaluated test split. These results are encouraging for
this dataset, but independent validation is needed before making claims
about performance in real-world mental-health screening.

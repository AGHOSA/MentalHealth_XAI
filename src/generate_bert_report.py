
import json
from pathlib import Path
from datetime import date

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_DIR / "results" / "bert_evaluation"
OUTPUT_FILE = EVAL_DIR / "BERT_Evaluation_Report.md"

metrics_path = EVAL_DIR / "metrics.json"
report_path = EVAL_DIR / "classification_report.csv"

with open(metrics_path, "r", encoding="utf-8") as file:
    metrics = json.load(file)

class_report = pd.read_csv(report_path, index_col=0)

def pct(value):
    return f"{value * 100:.2f}%"

cm = metrics["confusion_matrix"]
tn, fp = cm[0]
fn, tp = cm[1]

report = f"""# BERT Model Evaluation Report

**Project:** Mental Health Classification Through Social Media  
**Evaluation date:** {date.today().isoformat()}  
**Model:** {metrics["model"]}  
**Evaluation device:** {metrics["device"]}

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
- Test split: {metrics["test_samples"]} records (10%)
- Random seed: {metrics["seed"]}
- Evaluation device: {metrics["device"]}

The evaluation script reproduces the stratified train-validation-test
split using the same seed and split procedure as the training script.
This assumes the original CSV content and row order have not changed.

## 3. Evaluation Metrics

| Metric | Result |
|---|---:|
| Accuracy | {pct(metrics["accuracy"])} |
| Balanced accuracy | {pct(metrics["balanced_accuracy"])} |
| Macro precision | {pct(metrics["macro_precision"])} |
| Macro recall | {pct(metrics["macro_recall"])} |
| Macro F1-score | {pct(metrics["macro_f1"])} |
| ROC-AUC | {pct(metrics["roc_auc"])} |

## 4. Confusion Matrix

Rows represent actual labels; columns represent predicted labels.

| Actual class | Predicted non-depression | Predicted depression |
|---|---:|---:|
| Non-depression | {tn} | {fp} |
| Depression | {fn} | {tp} |

The model correctly classified {tn + tp} of {metrics["test_samples"]}
test samples and incorrectly classified {fp + fn} samples.

## 5. Class-wise Classification Report

See `classification_report.csv` for the complete precision, recall,
F1-score, and support values for each class.

## 6. Visualizations

- `confusion_matrix.png`: visual comparison of actual and predicted labels.
- `roc_curve.png`: ROC curve showing the relationship between the
  true-positive rate and false-positive rate across thresholds.

## 7. Interpretation

The test accuracy was {pct(metrics["accuracy"])}, and the macro F1-score
was {pct(metrics["macro_f1"])}. The ROC-AUC was {pct(metrics["roc_auc"])}.

Macro-averaged metrics give equal weight to both classes, while balanced
accuracy averages recall across the classes. These metrics complement
accuracy when class counts are unequal.

The confusion matrix shows {fp} non-depression samples predicted as
depression and {fn} depression samples predicted as non-depression.

## 8. Limitations

1. The test set contains only {metrics["test_samples"]} samples.
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

The fine-tuned BERT model achieved {pct(metrics["accuracy"])} accuracy,
{pct(metrics["macro_f1"])} macro F1-score, and {pct(metrics["roc_auc"])}
ROC-AUC on the evaluated test split. These results are encouraging for
this dataset, but independent validation is needed before making claims
about performance in real-world mental-health screening.
"""

OUTPUT_FILE.write_text(report, encoding="utf-8")

print("Report generated successfully:")
print(OUTPUT_FILE)
print("\nClass-wise report preview:")
print(class_report.to_string())

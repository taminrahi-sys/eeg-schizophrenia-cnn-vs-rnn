# Results

## Deep Learning Models

The RNN achieved strong classification performance on file-level EEG classification:

| Metric | Value |
|---|---|
| Recall | 0.8696 |
| F1-Score | 0.8333 |
| ROC-AUC | 0.8338 |
| PR-AUC | 0.8605 |

The CNN and RNN models showed improved performance on file-level
classification compared to window-level classification, indicating that
aggregated EEG decisions improve robustness.

The deep learning models achieved strong ROC-AUC and PR-AUC values,
demonstrating effective detection of schizophrenia-related EEG patterns.

No statistically significant difference between CNN and RNN performance
was observed across the cross-validation folds.

---

## Machine Learning Models

| Model | Mean Accuracy | Std |
|---|---|---|
| XGBoost | 0.906 | 0.067 |
| SVM | 0.870 | 0.076 |

XGBoost achieved the most stable performance among the machine learning
models and demonstrated lower variance across folds compared to SVM.

---

## Visualization

### CNN vs RNN

![DL Comparison](results/figures/DL_boxplot.png)

### XGBoost vs SVM

![ML Comparison](results/figures/ML_boxplot.png)

### CNN Training

![CNN Training](results/figures/CNN_train_val_accuracy.png)

### RNN Training

![RNN Training](results/figures/RNN_train_val_accuracy.png)

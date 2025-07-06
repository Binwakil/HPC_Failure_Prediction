# HPC Failure Prediction - Research Pipelines

This project implements five robust machine learning pipelines for predicting node- or job-level failures in High-Performance Computing (HPC) clusters, based on the Google Cluster 2019 dataset. The work focuses on hybrid architectures, anomaly detection, explainability, and ensemble modeling.

---

## 📁 Dataset

* Source: Google Cluster 2019 workload traces
* File: `Data/borg_traces_data.csv`
* Records: \~405,894 rows with 34 features
* Target: `failed` (binary classification)

---

## ✅ Completed Pipelines

### 1. Temporal-Graph Hybrid (LSTM + GCN)

* **Input**: CPU histogram sequences + graph structure from `alloc_collection_id` and `machine_id`
* **Architecture**: LSTM + Graph Convolution Network
* **Goal**: Failure prediction <10 minutes ahead
* **Output**: Node-level prediction
* **Result**: \~87% Accuracy, AUC \~0.88

### 2. Early-Failure LSTM + Attention

* **Input**: 6-timestep CPU histograms
* **Model**: LSTM with attention layer
* **Goal**: Learn temporal patterns with interpretability
* **Attention**: Plotted per-sample
* **Issue**: Class imbalance (low failure recall)

### 3. XGBoost + SMOTE + SHAP

* **Features**: CPU stats, memory usage, alloc ID, scheduling class
* **Technique**: Oversampling using SMOTE + XGBoost
* **Explainability**: SHAP values for global and local interpretability
* **Result**: 95% Accuracy, AUC \~0.99

### 4. Autoencoder + One-Class SVM

* **Train**: On healthy sequences only
* **Detect**: Outlier reconstruction errors
* **Post-processing**: One-Class SVM for anomaly detection
* **Visualization**: Error distribution + confusion matrix

### 5. Stacked Ensemble

* **Base Models**: All 4 above (XGB, LSTM, Hybrid, SVM)
* **Meta Model**: Logistic Regression on predictions
* **Explainability**: SHAP applied to meta-model
* **Purpose**: Consolidated decision maker with high accuracy

---

## 📊 Visualization

All plots saved to `visualization/`:

* Confusion matrices
* ROC curves
* SHAP feature plots
* Attention weights
* Reconstruction error histograms

---

## 💾 Predictions

All individual pipeline predictions are saved in `predictions/`:

* Format: `*_preds.npy`, `*_labels.npy`
* Used for ensemble training

---

## 🧪 Reproducibility

Each pipeline includes:

* Data cleaning
* Feature extraction
* Model training (train/val/test split)
* Metric logging
* Computational efficiency tracking
* Visualization + explainability

---

## 🔍 Future Work

* Finalize SHAP bar plot for ensemble model
* Improve failure recall in LSTM pipeline
* Export models for inference API
* Write academic paper or thesis chapter

---

## 🧠 Author & Contributions

Developed by: `binwa` with support from GPT-4.

---

For questions or contributions, please contact: [aicareerhub.ai@gmail.com](mailto:aicareerhub.ai@gmail.com)

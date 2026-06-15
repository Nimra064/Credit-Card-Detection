# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using classification algorithms on an imbalanced real-world dataset. Built with Python in a Jupyter Notebook environment.

---

## 📌 Overview

Credit card fraud is a major financial threat affecting millions of consumers and businesses worldwide. This project applies supervised machine learning techniques to automatically classify transactions as **legitimate** or **fraudulent**, helping financial institutions flag suspicious activity in real time.

The core challenge addressed in this project is **class imbalance** — fraudulent transactions represent a very small fraction of all transactions — which requires careful handling during preprocessing and model evaluation.

---

## 🗂️ Repository Structure

```
Credit-Card-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook: EDA, preprocessing, modeling, evaluation
├── README.md                           # Project documentation
└── .gitignore                          # Files excluded from version control
```

---

## 📦 Dataset

The dataset is hosted on Google Drive and contains labeled credit card transaction records.

🔗 **Dataset:** [Google Drive Link](https://drive.google.com/drive/folders/1ZDvOVUdoxrRwp8fW-l0ReQ1pPGJxv5DP?usp=sharing)

The dataset typically includes:
- **Transaction features** — amount, time, and anonymized PCA components (V1–V28)
- **Class label** — `0` for legitimate, `1` for fraudulent
- Highly imbalanced: fraudulent transactions make up a small minority of records

> Download the dataset and place it in the project root directory before running the notebook.

---

## 🔍 Project Pipeline

The notebook follows a complete end-to-end ML workflow:

1. **Data Loading** — Import dataset and inspect structure, shape, and data types
2. **Exploratory Data Analysis (EDA)** — Visualize class distribution, transaction patterns, and feature correlations
3. **Data Preprocessing**
   - Handle missing values
   - Feature scaling (StandardScaler)
   - Address class imbalance (e.g., undersampling, oversampling with SMOTE, or class weights)
4. **Model Training** — Train one or more classification models such as:
   - Logistic Regression
   - Random Forest
   - Decision Tree
   - XGBoost
5. **Model Evaluation** — Evaluate using fraud-appropriate metrics:
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC Score
6. **Results & Insights** — Summarize findings and model performance

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Nimra064/Credit-Card-Detection.git
cd Credit-Card-Detection
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost jupyter
```

### 3. Download the Dataset

Download the dataset from the [Google Drive link](https://drive.google.com/drive/folders/1ZDvOVUdoxrRwp8fW-l0ReQ1pPGJxv5DP?usp=sharing) and place the CSV file in the project root directory.

### 4. Run the Notebook

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

Or open it in **Google Colab** for a cloud-based environment (no local setup required).

---

## 📊 Key Metrics

For fraud detection, accuracy alone is misleading due to class imbalance. The following metrics are prioritized:

| Metric      | Why It Matters                                          |
|-------------|--------------------------------------------------------|
| **Precision** | Of transactions flagged as fraud, how many truly are? |
| **Recall**    | Of all actual fraud cases, how many did we catch?     |
| **F1-Score**  | Harmonic mean of precision and recall                 |
| **ROC-AUC**   | Overall discriminative ability of the model           |

> For full results and confusion matrices, refer to the output cells in `Credit_Card_Fraud_Detection.ipynb`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Pandas & NumPy** | Data manipulation |
| **Scikit-learn** | ML models and evaluation |
| **Imbalanced-learn** | Handling class imbalance (SMOTE) |
| **Matplotlib & Seaborn** | Data visualization |
| **Jupyter Notebook** | Interactive development |

---

## ⚠️ Handling Class Imbalance

Fraudulent transactions are rare events. This project addresses the imbalance problem using one or more of the following strategies:

- **Undersampling** — Reduce majority class samples
- **Oversampling (SMOTE)** — Synthetically generate minority class samples
- **Class Weights** — Penalize misclassification of the minority class more heavily during training

---

## 📄 License

This project is open-source and available for educational and research purposes.

---

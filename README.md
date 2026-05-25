# 🏦 Loan Approval Prediction — Machine Learning Project

## 📌 Project Overview

This project builds a machine learning pipeline to predict whether a loan application will be **approved or rejected**, based on applicant financial and demographic data. It covers the full data science workflow — from raw data exploration to model comparison and evaluation.

---

## 🎯 Problem Statement

Banks and financial institutions face the challenge of evaluating thousands of loan applications efficiently and fairly. This project automates the decision-making process using supervised classification models, helping identify creditworthy applicants based on historical patterns.

---

## 📂 Dataset

**File:** `loan_approval_data.csv`

**Key Features used:**
| Feature | Description |
|---|---|
| `Applicant_Income` | Monthly income of the applicant |
| `Credit_Score` | Applicant's credit score |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Applicant's savings amount |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Duration of the loan |
| `Education_Level` | Highest educational qualification |
| `Employment_Status` | Current employment status |
| `Employer_Category` | Type/category of employer |
| `Marital_Status` | Marital status of applicant |
| `Loan_Purpose` | Reason for taking the loan |
| `Property_Area` | Urban / Semi-Urban / Rural |
| `Gender` | Gender of the applicant |
| `Age` | Age of the applicant |
| `Loan_Approved` | **Target variable** — Yes / No |

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Environment:** Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Data Loading & Exploration
- Loaded dataset using `pandas`
- Explored data shape, data types, and basic statistics via `.info()` and `.describe()`

### 2. Data Preprocessing
- **Missing value imputation:**
  - Numerical columns → filled with **mean** (using `SimpleImputer`)
  - Categorical columns → filled with **most frequent value**
- **Label Encoding:** Applied to `Education_Level` and `Loan_Approved`
- **One-Hot Encoding:** Applied to multi-category columns (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`)

### 3. Exploratory Data Analysis (EDA)
- **Pie chart** — Distribution of loan approval outcomes
- **Bar charts** — Counts by Education Level, Employment Status, Employer Category, Loan Term
- **Box plots** — Income, Credit Score, DTI Ratio, and Savings vs Loan Approval status
- **Correlation Heatmap** — Feature-level correlation matrix using `seaborn`

### 4. Feature Engineering
- Created polynomial features: `DTI_Ratio_sq`, `Credit_Score_sq`
- Applied log transformation: `Applicant_Income_log = log1p(Applicant_Income)`
- Dropped original `Credit_Score` and `DTI_Ratio` after transformation

### 5. Model Training & Evaluation
Split the data — **80% training / 20% testing** — and applied `StandardScaler` before model fitting.

**Models Compared:**

| Model | Metric |
|---|---|
| Logistic Regression | Precision, Recall, F1-Score |
| K-Nearest Neighbors (K=5) | Precision, Recall, F1-Score |
| Gaussian Naive Bayes | Precision, Recall, F1-Score |

Both base and feature-engineered versions of the dataset were tested for each model.

---

## 📊 Evaluation Metrics

- **Precision** — Of predicted approvals, how many were truly approved?
- **Recall** — Of all actual approvals, how many were correctly identified?
- **F1-Score** — Harmonic mean of Precision and Recall

---

## 🚀 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place `loan_approval_data.csv` in the same directory as the notebook
4. Open and run `loan.ipynb` in Jupyter Notebook or JupyterLab

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── loan.ipynb               # Main Jupyter Notebook
├── loan_approval_data.csv   # Dataset (required)
└── README.md                # Project documentation
```

---

## 🙋 Author

> Built with curiosity and code. Feel free to fork, star ⭐, and connect!

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

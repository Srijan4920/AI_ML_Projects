# 📊 Loan Classification - Credit Risk Modelling

This machine learning project focuses on predicting the likelihood of loan repayment by classifying applicants based on various features.

## 🧠 Objective

To build a classification model that determines whether a loan application should be **approved** or **rejected** based on applicant details.

## 📁 Dataset

The dataset used: `loan_detection.csv`

### Features include:
- Gender
- Marital Status
- Education
- Applicant Income
- Loan Amount
- Credit History
- Property Area
- And more...

### Target:
- `Loan_Status`: Y (Approved), N (Rejected)

## 🛠️ ML Workflow

1. Data Loading & Inspection  
2. Missing Value Handling  
3. Feature Engineering  
4. Label Encoding  
5. Train-Test Split  
6. Model Training (Logistic Regression, Random Forest)  
7. Model Evaluation (Accuracy, Confusion Matrix)

## 🔍 Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib (for model serialization)

## 📈 Model Performance

| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression| 80%      |
| Random Forest      | 83%      |

## 💾 File Structure

```bash
Loan_Classification/
├── Loan_Classification_Project.ipynb
├── loan_detection.csv
├── model.pkl
└── README.md

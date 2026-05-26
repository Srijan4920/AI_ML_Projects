# Disease Prediction Using Machine Learning

An interactive, clinical-grade medical diagnostic assistant powered by a machine learning soft-voting ensemble model. The system processes patient symptoms, classifies the data across 38 distinct diseases, and generates diagnostic reports with severity warnings and recommended clinical precautions.

---

## 📁 Project Structure

*   [Notebook_Disease_Prediction_Using_Machine_Learning_.ipynb](file:///d:/Disease%20Prediction/Notebook_Disease_Prediction_Using_Machine_Learning_.ipynb): Development notebook containing pipeline experiments, cross-validation runs, and documentation for each step.
*   [improved_disease_dataset.csv](file:///d:/Disease%20Prediction/improved_disease_dataset.csv): The baseline dataset containing 2,000 patient records across 10 symptom columns and 38 disease target classes.
*   [app.py](file:///d:/Disease%20Prediction/app.py): Interactive Streamlit web application providing a graphical user intake, consensus prediction outputs, probability charts, and diagnostic reports.

---

## 🛠️ Installation & Setup

1.  **Prerequisites**:
    Ensure you have Python 3.8+ installed on your system.

2.  **Install Required Libraries**:
    Open your terminal or PowerShell and install the required dependencies:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn streamlit fpdf2
    ```

3.  **Run the Streamlit Web Application**:
    Execute the following command in your terminal to start the interactive dashboard:
    ```powershell
    streamlit run "d:\Disease Prediction\app.py"
    ```
    Once started, the application will automatically launch in your default browser at `http://localhost:8501`.

---

## 🧠 Machine Learning Design & Pipeline

### 1. Data Leakage Resolution
Oversampling (to address class imbalances) is applied dynamically **inside** each cross-validation fold using `imblearn.pipeline.Pipeline`. This prevents duplicate rows from leaking between the training and validation splits, giving a statistically honest estimate of the models' performance.

### 2. Soft-Voting Ensemble Classifier
The core diagnostic engine combines the predictions of four distinct classifiers:
*   **Support Vector Machine (SVM)** (with probability estimation enabled)
*   **Random Forest Classifier**
*   **Gradient Boosting Classifier**
*   **Gaussian Naive Bayes**

The models are combined in a `VotingClassifier` utilizing a soft-voting scheme, which averages predicted probability distributions across all 38 disease classes to make a final consensus decision.

---

## ✨ Application Features

*   **Clinical Intake Checklist**: Symptoms are categorized into collapsible clinical sections (`Systemic`, `Neurological`, `Gastrointestinal`, etc.) for clean and intuitive selection.
*   **Disease Database**: Displays detailed description summaries and actionable precautions for predicted diseases.
*   **Severity Warnings**: Diagnoses are color-coded: **High** (Red/Urgent emergency alerts), **Medium** (Orange/Physician scheduling), and **Low** (Green/General care).
*   **Probability Charts**: Renders visual bar charts of the top 5 most probable diseases.
*   **PDF Diagnostic Report Exporter**: Generates and downloads a clean, formatted PDF document detailing the patient's symptoms, diagnoses, and medical instructions using the `fpdf2` library in-memory.

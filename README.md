# Stroke Predictor

Predict stroke risk from health/demographic features.  
This repo includes:

-   **Exploration notebook** (CS97 project): multiple classical ML models + a PyTorch NN.
    
-   **Pipeline**: clean `scikit-learn` training script that saves a model + metadata.
    
-   **Interactive app**: a **Gradio** UI to try predictions in the browser.
    

> **Educational use only — not for clinical decisions.**

---

## 🚀 Features

- **Data Preprocessing**
  - Handles missing values with median imputation.
  - Encodes categorical variables.
  - Standardizes numerical features.
  - Balances the dataset to address class imbalance.

- **Machine Learning Models Implemented**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Neural Network (PyTorch)

- **Model Evaluation**
  - Accuracy, Precision, Recall, and F1 Score
  - Confusion Matrices with visualization
  - ROC Curves & AUC scores for model comparison
 
### Notebook (Exploration)

-   Median imputation, encoding, scaling
    
-   Models: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, PyTorch NN
    
-   Metrics: Accuracy, Precision/Recall/F1, Confusion Matrix, ROC-AUC

### Pipeline (Training script)

-   Clean `ColumnTransformer` (no leakage), stratified split
    
-   **Threshold tuning** (choose decision threshold to balance precision/recall)
    
-   Saves `model.joblib` + `metadata.json` (ROC-AUC, PR-AUC, chosen threshold, feature lists)
    
### App (Gradio)

-   Form inputs for all features (age, BMI, glucose, hypertension, etc.)
    
-   Returns probability + label; slider to change decision threshold
    
-   Uses the same saved model (no code changes needed when you retrain)

---

## 📊 Dataset
The dataset was sourced from [Kaggle’s Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset).  

- **Features include:**  
  - Age, Gender, BMI, Smoking Status  
  - Hypertension, Heart Disease  
  - Average Glucose Level  
  - Marital/Work/Residence Status  

- **Target:** `stroke` (1 = Stroke, 0 = No Stroke)

---

## ⚙️ Installation & Usage

### ▶️ Option A: Run in Colab (no setup required)

Click the badge at the top of this README and run all cells.

### ▶️ Option B: Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/AkhilByju/Stroke-Predictor.git
   cd Stroke-Predictor
   ```

2. (Optional) Create & activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook and open the file:

   ```bash
   jupyter notebook Stroke_Predictor_CS97.ipynb
   ```

## ▶️ How to run

### A) Explore in the notebook

```bash
jupyter notebook Stroke_Predictor_CS97.ipynb
# run cells to see EDA, model training, and plots

```
### B) Train a model (pipeline)

Use the trainer:

```bash
python train_pipeline.py

```

This will create:

-   `model.joblib` — the trained model (ignored by git)
    
-   `metadata.json` — metrics + chosen threshold (ignored by git)
    

### C) Launch the Gradio app

```bash
python app.py

```

Open the printed URL (default: [http://127.0.0.1:7860](http://127.0.0.1:7860/)).  
The app reads `model.joblib` + `metadata.json` and uses the saved **threshold** as the default.

---

## 📈 Results

* **Random Forest** and **Logistic Regression** achieved strong performance compared to other models.
* **Neural Network** implementation demonstrates the potential of deep learning but requires tuning for optimal performance.

**Evaluation includes:**

* Confusion Matrices
* ROC Curves with AUC Scores
* Comparative Accuracy

---

## 🔮 Future Improvements

* Hyperparameter optimization using GridSearchCV / RandomizedSearch
* Handling class imbalance with oversampling or weighted loss functions
* Model interpretability using SHAP or LIME
* Deployment as a **web app** (Flask/FastAPI + React)
* Cross-validation and ensemble methods for robustness

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and **not for clinical use**.
The models and predictions should **not** be used for real medical decision-making.

---

## 👨‍💻 Author

**Akhil Byju**
Undergraduate Student, UCLA (Computer Science)
Interests: Machine Learning, Artificial Intelligence


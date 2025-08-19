# Stroke-Predictor

This project predicts whether an individual is at risk of having a stroke using machine learning models trained on health and demographic data. It explores classical ML algorithms as well as a neural network built in PyTorch.

The project was created as part of a CS97 Machine Learning course project.

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

5. Run all cells to preprocess data, train models, and view evaluation results.

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
Undergraduate Student, UCLA (MCDB + Applied Mathematics, Pre-Med)
Interests: Machine Learning, Bioengineering, Neuroscience


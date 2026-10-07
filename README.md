# ANN Classification: Customer Churn Prediction

An end-to-end deep learning project that uses an **Artificial Neural Network (ANN)** to predict whether a bank customer will leave (churn), plus a second ANN that estimates customer salary. Both models are served through interactive **Streamlit** web apps.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-yellow)
![License](https://img.shields.io/badge/License-GPL--3.0-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Overview

Customer churn is expensive: retaining an existing customer is usually cheaper than acquiring a new one. This project builds a binary classifier that estimates the probability that a customer will leave the bank, based on demographic and account information.

The project covers the full machine learning workflow:

1. Data cleaning and feature engineering
2. Categorical encoding and feature scaling
3. ANN design, training, and evaluation
4. Hyperparameter tuning with grid search
5. Experiment tracking with TensorBoard
6. Deployment of the trained models as Streamlit apps

## Features

- **Churn classification:** outputs a churn probability and a likely / not likely verdict (threshold 0.5).
- **Salary regression:** a second ANN predicts a customer's estimated salary.
- **Hyperparameter tuning:** `GridSearchCV` with `scikeras` over neurons, layers, and epochs.
- **Early stopping** on validation loss to prevent overfitting.
- **TensorBoard logging** for training and validation curves.
- **Reusable preprocessing:** the fitted label encoder, one-hot encoder, and scaler are saved as `.pkl` files so the apps apply the exact same transformations as training.
- **Interactive UI** built with Streamlit.

## Dataset

The project uses the **Churn Modelling** dataset (`Churn_Modelling.csv`): 10,000 bank customers with 14 columns.

| Feature | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Has a credit card (0/1) |
| `IsActiveMember` | Active member (0/1) |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target:** 1 if the customer churned, else 0 |

About 20% of customers in the dataset churned. `RowNumber`, `CustomerId`, and `Surname` are dropped as non-predictive identifiers.

### Preprocessing

- `Gender` is label encoded.
- `Geography` is one-hot encoded.
- Data is split 80/20 into train and test sets (`random_state=42`).
- Features are standardized with `StandardScaler` (fit on the training set only).

## Model Architecture

### Churn classifier

```
Input (12 features)
   ↓
Dense (64, ReLU)
   ↓
Dense (32, ReLU)
   ↓
Dense (1, Sigmoid)   → churn probability
```

- **Parameters:** 2,945
- **Optimizer:** Adam (learning rate 0.01)
- **Loss:** Binary cross-entropy
- **Callbacks:** EarlyStopping (patience 10, restores best weights), TensorBoard

### Salary regressor

```
Input (12 features)
   ↓
Dense (64, ReLU)
   ↓
Dense (32, ReLU)
   ↓
Dense (1)            → predicted salary
```

- **Optimizer:** Adam
- **Loss / metric:** Mean absolute error
- **Callbacks:** EarlyStopping, TensorBoard

## Results

| Model | Metric | Result |
|---|---|---|
| Churn classifier | Validation accuracy | **~85.6%** |
| Churn classifier (grid search) | Best 3-fold CV accuracy | **85.65%** (128 neurons, 1 hidden layer, 50 epochs) |
| Salary regressor | Test MAE | ~$50,263 |

**Hyperparameter search space:** neurons `[16, 32, 64, 128]` × layers `[1, 2]` × epochs `[50, 100]` (16 combinations, 3-fold cross-validation, 48 fits).

> **Note on the salary model:** `EstimatedSalary` in this dataset is spread almost uniformly between $0 and $200K and has little relationship with the other features. A naive model that always predicts the median salary already scores an MAE of about $49.7K, so the regressor is best viewed as a demonstration of the regression workflow rather than a production-grade predictor.

## Project Structure

```
ANN-Classification-Churn/
├── app.py                          # Streamlit app: churn prediction
├── streamlit_regression.py         # Streamlit app: salary prediction
├── experiments.ipynb               # Data prep, ANN training, TensorBoard
├── hyperparametertuningann.ipynb   # GridSearchCV hyperparameter tuning
├── salaryregression.ipynb          # ANN regression model
├── prediction.ipynb                # Inference on a sample customer
├── Churn_Modelling.csv             # Dataset
├── model.keras                     # Trained churn classifier
├── regression_model.keras          # Trained salary regressor
├── label_encoder_gender.pkl        # Fitted gender encoder
├── onehot_encoder_geo.pkl          # Fitted geography encoder
├── scaler.pkl                      # Fitted StandardScaler
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/atul-aiml/ANN-Classification-Churn.git
cd ANN-Classification-Churn

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the churn prediction app

```bash
streamlit run app.py
```

Enter the customer's geography, gender, age, balance, credit score, salary, tenure, number of products, and card / activity status. The app returns the churn probability and a verdict.

### Run the salary prediction app

```bash
streamlit run streamlit_regression.py
```

### Retrain the models

Open the notebooks in Jupyter or VS Code and run them in this order:

1. `experiments.ipynb` trains the churn classifier and saves `model.keras`, the encoders, and the scaler.
2. `hyperparametertuningann.ipynb` runs the grid search.
3. `salaryregression.ipynb` trains the salary regressor and saves `regression_model.keras`.

### View training curves in TensorBoard

```bash
tensorboard --logdir logs/fit
```

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow / Keras
- **Machine learning utilities:** scikit-learn, SciKeras
- **Data handling:** pandas, NumPy
- **Web app:** Streamlit
- **Experiment tracking:** TensorBoard

## Future Improvements

- Address class imbalance (about 20% churn) with class weights or SMOTE, and report precision, recall, F1, and ROC-AUC alongside accuracy.
- Save a separate scaler for each model (for example `scaler_churn.pkl` and `scaler_salary.pkl`), since the two models use different feature sets.
- Add dropout and batch normalization, and compare against tree-based baselines such as XGBoost.
- Add model explainability with SHAP.
- Deploy the app to Streamlit Community Cloud.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Author

**Atul Choudhary**

- GitHub: [@atul-aiml](https://github.com/atul-aiml)
- LinkedIn: _add your profile link here_

If you found this project useful, consider giving it a star.

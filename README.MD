# 🏥 Medical Insurance Claim Prediction

An end-to-end machine learning system that predicts medical insurance claim amounts from patient demographic, lifestyle, and health data. The project covers the full ML lifecycle — exploratory data analysis, data cleaning, feature engineering, model selection, hyperparameter tuning, explainability, and deployment as both a Streamlit app and a Flask REST-style web app.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Explainability (SHAP)](#explainability-shap)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API / Input Schema](#api--input-schema)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Insurance providers need accurate, explainable estimates of expected claim costs to price policies fairly and manage risk. This project builds a regression pipeline that predicts a policyholder's **claim amount** from features such as age, BMI, smoking status, blood pressure, hereditary disease, and occupation.

Key goals:

- Build a clean, leak-free preprocessing pipeline
- Benchmark multiple regression algorithms objectively
- Tune the best-performing model
- Explain predictions with SHAP (not just report metrics)
- Ship the model behind two different deployment interfaces

---

## Dataset

**File:** `Data/medical_insurance_claims.csv`

| Feature | Description |
|---|---|
| `age` | Age of the insured |
| `sex` | Biological sex |
| `weight` | Weight (kg) |
| `bmi` | Body Mass Index |
| `hereditary_diseases` | Presence/type of hereditary disease |
| `no_of_dependents` | Number of dependents |
| `smoker` | Smoking status (0/1) |
| `city` | City of residence |
| `bloodpressure` | Blood pressure reading |
| `diabetes` | Diabetic status (0/1) |
| `regular_ex` | Regular exercise habit (0/1) |
| `job_title` | Occupation |
| `claim` | **Target** — medical insurance claim amount |

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Univariate distributions (`histplot` with KDE) for all numeric features
- Boxplots per numeric feature to visually flag outliers
- Correlation heatmap across numeric variables
- Pairwise scatter matrix (`pairplot`)
- Categorical feature distributions (`countplot`)

### 2. Data Cleaning

- Stripped whitespace from column headers
- Missing value imputation: `age` and `bmi` filled with column mean
- Duplicate rows detected and dropped
- Full statistical profiling (`describe`, `value_counts`) for both numeric and categorical columns

### 3. Outlier Handling

- IQR method (1.5×IQR rule) applied to `age`, `weight`, `bmi`, and `bloodpressure`
- Rows outside bounds on **any** of these columns removed
- Outlier impact quantified (% of rows affected) before removal

### 4. Preprocessing Pipeline

Built with `ColumnTransformer` to prevent train/test leakage:

| Branch | Steps |
|---|---|
| **Categorical** | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")` |
| **Numerical** | `SimpleImputer(mean)` → `StandardScaler` |

Fit only on training data; test data transformed using the fitted preprocessor.

### 5. Feature Selection

- `SelectKBest` with `r_regression` scoring
- Swept `k ∈ {5, 10, 20, 30, 40, 50, 75, 100}` and benchmarked each with an `XGBRegressor` to find the point of diminishing returns
- Final model uses **k = 30** selected features

### 6. Model Benchmarking

Nine regressors trained and compared out of the box:

`LinearRegression`, `SGDRegressor`, `SVR`, `DecisionTreeRegressor`, `KNeighborsRegressor`, `RandomForestRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`, `XGBRegressor`

### 7. Hyperparameter Tuning

- `RandomizedSearchCV` per model (`n_iter=10`, `cv=3`, scoring = `neg_mean_absolute_error`)
- Custom search space defined per algorithm (regularization, depth, learning rate, ensemble size, etc.)
- Best estimator retained per model for final evaluation

### 8. Evaluation

Final comparison on a held-out test set using **R², MAE, MSE, RMSE**.

### 9. Explainability

SHAP `TreeExplainer` applied to the best model (Gradient Boosting) to surface global feature importance and per-feature impact direction.

### 10. Deployment

Final production pipeline (`preprocessor → SelectKBest → GradientBoostingRegressor`) serialized with `pickle` and served via:

- A **Streamlit** form-based UI
- A **Flask** web app with an HTML form and server-side prediction endpoint

---

## Modeling Approach

The final production model is a single scikit-learn `Pipeline` — not just a bare estimator — so preprocessing and feature selection travel with the model and can't drift out of sync with the code that trained it:

```python
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("select", SelectKBest(score_func=r_regression, k=30)),
    ("model", GradientBoostingRegressor(
        max_depth=7,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=200,
        random_state=42,
        subsample=0.8
    ))
])
```

---

## Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Gradient Boosting** | **0.964** | 771.88 | 2322.47 |
| XGBoost | 0.963 | 850.99 | 2368.60 |
| KNN | 0.961 | 432.03 | 2413.02 |
| Random Forest | 0.953 | 969.79 | 2657.02 |
| Decision Tree | 0.946 | 525.82 | 2836.67 |
| AdaBoost | 0.795 | 3653.87 | 5544.80 |
| Linear Regression | 0.762 | 3817.63 | 5968.79 |
| SGD Regressor | 0.762 | 3814.91 | 5968.85 |
| SVR | 0.724 | 3350.74 | 6428.24 |

**Gradient Boosting** was selected as the final model, balancing the highest R² with strong error metrics. An Actual-vs-Predicted scatter plot (with a reference diagonal) is used to visually confirm calibration across the claim range.

> Note: KNN's lower MAE despite a slightly lower R² than XGBoost highlights why multiple metrics — not just R² — are reported.

---

## Explainability (SHAP)

Rather than treating the model as a black box, SHAP values are computed on the test set to show:

- **Global feature importance** (summary/bar plot) — which features drive claim predictions most
- **Directionality** — whether a feature (e.g., `smoker`, `bmi`, `age`) pushes predictions up or down
- Feature names are mapped back from the `ColumnTransformer` output via `get_feature_names_out()`, so SHAP plots are human-readable rather than showing encoded column indices

---

## Tech Stack

| Category | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn`, `xgboost` |
| Explainability | `shap` |
| Model persistence | `pickle` |
| Deployment | `streamlit`, `flask` |

---

## Project Structure

```
medical-insurance-claim-prediction/
│
├── Data/
│   └── medical_insurance_claims.csv
│
├── notebooks/
│   └── claim_prediction_pipeline.ipynb
│
├── models/
│   └── gradient_boosting_claim_pipeline.pkl
│
├── app_streamlit.py       # Streamlit UI
├── app_flask.py           # Flask app
├── templates/
│   └── index.html         # Flask front-end
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Pince11feb2005/medical-insurance-claim-prediction-ml.git
cd medical-insurance-claim-prediction-ml
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Suggested `requirements.txt`:**

```
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
shap
streamlit
flask
```

---

## Usage

### Run the Streamlit app

```bash
streamlit run app_streamlit.py
```

### Run the Flask app

```bash
python app_flask.py
```

Then visit `http://127.0.0.1:5000/`

### Retrain the model

Open and run the notebook end-to-end, then confirm the pickle file is regenerated:

```python
import pickle
pickle.dump(model, open("gradient_boosting_claim_pipeline.pkl", "wb"))
```

---

## API / Input Schema

Both apps expect the following fields per prediction request:

```json
{
  "age": 39.5,
  "sex": "female",
  "weight": 41,
  "bmi": 27.6,
  "hereditary_diseases": "NoDisease",
  "no_of_dependents": 0,
  "smoker": 0,
  "city": "Minot",
  "bloodpressure": 58,
  "diabetes": 1,
  "regular_ex": 0,
  "job_title": "Dancer"
}
```

**Response:**

```json
{
  "predicted_claim": 4231.87
}
```

---

## Future Improvements

- Add cross-validated confidence intervals to predictions
- Containerize with Docker for consistent deployment
- Add automated tests for the preprocessing pipeline
- Expand SHAP explainability to per-prediction (local) explanations in the UI

---

## License

This project is available under the MIT License.
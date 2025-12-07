# IVF Patient Response Prediction

Machine Learning system for predicting patient response to IVF treatment using clinical parameters.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## = Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Results](#results)

---

##  Overview

This project implements a machine learning pipeline to predict IVF (In Vitro Fertilization) patient response categories (**low**, **optimal**, **high**) based on clinical parameters. The system provides:

- **86.1% accuracy** on test data
- **Probability outputs** for each response category
- **Explainable AI** using SHAP values
- **REST API** for integration
- **Interactive web interface** for clinicians

### Clinical Application

The model helps clinicians:
- Predict patient response to stimulation protocols
- Adjust treatment plans proactively
- Counsel patients on expected outcomes
- Identify high-risk cases (OHSS prevention)

---

##  Features

-  **PDF Data Extraction** - Automated extraction from clinical reports
-  **Data Preprocessing** - KNN imputation, standardization, de-identification
-  **Exploratory Data Analysis** - Statistical tests, visualizations
-  **Multiple ML Models** - Logistic Regression, Random Forest, XGBoost
-  **Model Explainability** - SHAP values for feature importance
-  **REST API** - FastAPI endpoint for predictions
-  **Web Interface** - Streamlit UI for easy interaction
-  **Comprehensive Tests** - Unit tests for all components

---

##  Project Structure

```
ivf-patient-response-prediction/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                      # Original data
│   │   ├── patients.csv
│   │   └── sample.pdf
│   └── processed/                # Cleaned data
│       ├── cleaned_data.csv
│       └── patient_data_with_pdf.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── feature_engineering.py    # PDF extraction
│   │   └── clean_dataset.py          # Data cleaning
│   │ 
│   │EDA/
│   └── EDA.ipynb    # EDA notebook
│   │ 
│   ├── model/
│   │   ├── dataset.py                # Data loading
│   │   ├── train.py                  # Model training
│   │   ├── evaluate.py               # Model evaluation
│   │   ├── predict.py                # Inference
│   │   ├── explain.py                # SHAP explainability
│   │   └── saved_models/             # Trained models
│   │
│   ├── api/
│   │   └── app.py                    # FastAPI application
│   │
│   └── ui/
│       └── app.py                    # Streamlit interface
│
├── reports/
│   ├── figures/                      # Generated plots
│   └── model_comparison.csv          # Model metrics
│
└── tests/
    └── test_model.py                 # Unit tests
```

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ivf-patient-response-prediction.git
cd ivf-patient-response-prediction
```

2. **Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import sklearn, xgboost, shap, fastapi, streamlit; print('✓ All packages installed')"
```

---

##  Usage

### 1. Data Preprocessing

**Extract PDF data:**
```bash
python src/preprocessing/feature_engineering.py
```

**Clean dataset:**
```bash
python src/preprocessing/clean_dataset.py
```

### 2. Exploratory Data Analysis

**Run Jupyter notebook:**
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### 3. Model Training

**Train all models:**
```bash
python src/model/train.py
```

**Output:**
- Trains 4 models: Logistic Regression, Random Forest, XGBoost, Calibrated XGBoost
- Saves models to `src/model/saved_models/`
- Displays training accuracy and sample predictions

### 4. Model Evaluation

**Evaluate models:**
```bash
python src/model/evaluate.py
```

**Output:**
- Confusion matrices
- Calibration curves
- Feature importance plots
- Model comparison report

### 5. Generate Explanations

**Create SHAP plots:**
```bash
python src/model/explain.py
```

**Output:**
- Feature importance visualization
- SHAP summary plots
- Individual prediction explanations

### 6. Run API Server

**Start FastAPI:**
```bash
cd src/api
uvicorn app:app --reload
```

**Access:**
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 7. Launch Web Interface

**Start Streamlit:**
```bash
streamlit run src/ui/app.py
```

**Access:** Browser will automatically open to `http://localhost:8501`

---

##  Model Performance

### Best Model: Random Forest

| Metric | Score |
|--------|-------|
| **Accuracy** | 86.1% |
| **Precision** | 0.868 |
| **Recall** | 0.861 |
| **F1-Score** | 0.860 |
| **AUC** | 0.959 |
| **Log Loss** | 0.388 |

### Performance by Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **High** | 0.95 | 0.72 | 0.82 | 25 |
| **Low** | 0.88 | 0.90 | 0.89 | 31 |
| **Optimal** | 0.82 | 0.91 | 0.86 | 45 |

### Feature Importance

1. **AMH** (37.9%) - Anti-Müllerian Hormone
2. **n_Follicles** (27.3%) - Number of follicles retrieved
3. **AFC** (22.6%) - Antral Follicle Count
4. **E2_day5** (5.2%) - Estradiol level on day 5
5. **Age** (4.0%) - Patient age

---

##  API Documentation

### Endpoints

#### `POST /predict`

Predict response for a single patient.

**Request Body:**
```json
{
  "Age": 32,
  "AMH": 2.5,
  "n_Follicles": 12,
  "E2_day5": 450.0,
  "AFC": 15,
  "cycle_number": 1,
  "Protocol": "flexible antagonist"
}
```

**Response:**
```json
{
  "prediction": "optimal",
  "confidence": 0.857,
  "probabilities": {
    "high": 0.102,
    "low": 0.041,
    "optimal": 0.857
  }
}
```

#### `POST /predict/batch`

Predict response for multiple patients.

**Request Body:**
```json
[
  { "Age": 32, "AMH": 2.5, ... },
  { "Age": 35, "AMH": 1.8, ... }
]
```

#### `GET /health`

Check API health status.

#### `GET /model/info`

Get model information and features.

### Example Usage

**Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "Age": 32,
    "AMH": 2.5,
    "n_Follicles": 12,
    "E2_day5": 450.0,
    "AFC": 15,
    "cycle_number": 1,
    "Protocol": "flexible antagonist"
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

**cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 32,
    "AMH": 2.5,
    "n_Follicles": 12,
    "E2_day5": 450.0,
    "AFC": 15,
    "cycle_number": 1,
    "Protocol": "flexible antagonist"
  }'
```

---

##  Results

### Key Findings

1. **AMH is the strongest predictor** (37.9% importance)
   - Higher AMH values correlate with optimal/high response
   - AMH < 1.0 ng/mL indicates low response risk

2. **Protocol effectiveness:**
   - Flexible antagonist: 62.3% optimal/high response
   - Fixed antagonist: 58.7% optimal/high response
   - Agonist: 56.1% optimal/high response

3. **Age shows non-linear effects:**
   - Peak response: 28-32 years
   - Decline after 35 years
   - Less important than ovarian reserve markers

4. **Model provides reliable probabilities:**
   - Well-calibrated (ECE < 0.05)
   - Confidence aligns with prediction accuracy
   - Suitable for clinical decision support

### Visualizations

All generated plots are saved in `reports/figures/`:

- `confusion_matrices.png` - Model predictions vs actual
- `calibration_curves.png` - Probability calibration
- `feature_importance.png` - Feature importance rankings
- `shap_summary.png` - Global feature effects
- `shap_detailed_effects.png` - Feature value impact
- `shap_single_patient.png` - Individual prediction explanation

---

##  Testing

Run all tests:
```bash
python tests/test_model.py
```

**Test Coverage:**
-  Data loading and preprocessing
-  Feature engineering
-  Model predictions
-  API compatibility
-  Minimum performance requirements

---

##  Dependencies

### Core Libraries
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `shap` - Model explainability

### Visualization
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive charts

### Web Frameworks
- `fastapi` - REST API
- `streamlit` - Web interface
- `uvicorn` - ASGI server

### PDF Processing
- `pdfplumber` - PDF text extraction

See `requirements.txt` for complete list with versions.

---

##  Privacy & Ethics

### Data Protection
- **De-identification:** Patient names converted to 25XXX format
- **No PHI storage:** API doesn't persist patient data
- **Audit trail:** Patient mapping saved separately

### Clinical Considerations
-  Tool is for **research and educational purposes**
-  Always use clinical judgment
-  Not a replacement for medical expertise
-  Validate predictions with clinical assessment

---


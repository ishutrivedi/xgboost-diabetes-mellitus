# XGBoost Diabetes Mellitus Prediction

A machine learning project that predicts diabetes mellitus in ICU patients using XGBoost classifier with comprehensive hyperparameter tuning, developed for the WiDS (Women in Data Science) Datathon 2021.

## 🏥 Domain Overview

### What is Diabetes Mellitus?
Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. In ICU settings, diabetes can significantly impact patient outcomes, making early detection and prediction crucial for:

- **Risk Assessment**: Identifying high-risk patients for targeted interventions
- **Resource Allocation**: Optimizing ICU bed management and staffing
- **Treatment Planning**: Enabling proactive diabetes management protocols
- **Outcome Improvement**: Reducing complications and mortality rates

### Clinical Significance
This project addresses a critical healthcare challenge by leveraging ICU patient data to predict diabetes mellitus occurrence. Early prediction enables:
- Preventive care measures
- Optimized treatment protocols
- Reduced healthcare costs
- Improved patient outcomes

## 🔬 Technical Implementation

### Dataset
- **Source**: WiDS Datathon 2021 Dataset
- **Training Data**: `TrainingWiDS2021.csv` - ICU patient records with diabetes labels
- **Test Data**: `UnlabeledWiDS2021.csv` - Unlabeled patient records for prediction
- **Target Variable**: `diabetes_mellitus` (binary classification)

### Data Preprocessing Pipeline

#### 1. **Data Cleaning**
- Duplicate row removal
- Feature selection based on clinical relevance and data quality

#### 2. **Categorical Data Handling**
- One-hot encoding for categorical variables:
  - `ethnicity`: Patient ethnic background
  - `icu_type`: Type of ICU unit
  - `gender`: Patient gender
  - `icu_admit_source`: Source of ICU admission
  - `icu_stay_type`: Type of ICU stay
  - `hospital_admit_source`: Hospital admission source

#### 3. **Missing Value Imputation**
Multiple imputation strategies tested:
- **Mean imputation**: ROC-AUC 0.824
- **Median imputation**: ROC-AUC 0.820
- **Most frequent imputation**: ROC-AUC 0.827
- **Constant imputation**: ROC-AUC 0.822 (selected)

#### 4. **Feature Engineering**
- Feature importance analysis using XGBoost
- Selection of top 192 non-zero importance features
- Dimensionality reduction from original feature set

### Model Architecture

#### XGBoost Classifier Configuration
```python
XGBClassifier(
    learning_rate=0.1,
    n_estimators=1500,
    max_depth=3,
    min_child_weight=2,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    reg_alpha=0.00001,
    use_label_encoder=False
)
```

#### Hyperparameter Tuning Process

**1. Tree Structure Optimization**
- `max_depth`: Tuned to 3 (prevents overfitting)
- `min_child_weight`: Optimized to 2

**2. Regularization Parameters**
- `gamma`: Set to 0 (no minimum split loss)
- `reg_alpha`: Fine-tuned to 0.00001 (L1 regularization)

**3. Sampling Parameters**
- `subsample`: 0.8 (80% of samples per tree)
- `colsample_bytree`: 0.8 (80% of features per tree)

**4. Learning Parameters**
- `learning_rate`: 0.1 (balanced learning speed)
- `n_estimators`: 1500 (sufficient iterations with early stopping)

### Model Performance

#### Evaluation Metrics
- **Primary Metric**: ROC-AUC Score (Area Under ROC Curve)
- **Secondary Metric**: Accuracy Score
- **Validation Strategy**: 80-20 train-validation split

#### Achieved Performance
- **Training Accuracy**: ~85.4%
- **ROC-AUC Score**: ~0.87-0.89
- **Cross-validation**: 5-fold CV with early stopping

### Advanced Techniques

#### Feature Importance Analysis
- Automated feature selection based on XGBoost importance scores
- Retention of 192 most important features from original dataset
- Elimination of zero-importance features to reduce noise

#### Model Validation
- Custom `modelfit()` function with cross-validation
- Early stopping mechanism (90 rounds) to prevent overfitting
- Grid search for optimal hyperparameter combinations

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost
```

### Required Libraries
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
```

### Usage

1. **Data Loading**
```python
df = pd.read_csv("TrainingWiDS2021.csv")
test = pd.read_csv("UnlabeledWiDS2021.csv")
```

2. **Run the Complete Pipeline**
Execute the Jupyter notebook `xgboost-with-hyper-parameter-tuning.ipynb` which includes:
- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Prediction generation

3. **Generate Predictions**
The model outputs predictions in `submission.csv` format with:
- `encounter_id`: Patient encounter identifier
- `diabetes_mellitus`: Predicted probability of diabetes

## 📊 Project Structure

```
xgboost-diabetes-mellitus/
├── README.md                                    # Project documentation
├── xgboost-with-hyper-parameter-tuning.ipynb  # Main analysis notebook
└── submission.csv                              # Generated predictions (after running)
```

## 🔍 Key Features

### Data Science Methodology
- **Systematic Approach**: Structured pipeline from data loading to prediction
- **Feature Engineering**: Intelligent feature selection and importance analysis
- **Model Optimization**: Comprehensive hyperparameter tuning
- **Validation Strategy**: Robust cross-validation and performance metrics

### Clinical Relevance
- **ICU-Specific**: Tailored for intensive care unit patient data
- **Multi-Modal Features**: Incorporates demographic, clinical, and administrative data
- **Interpretable Results**: Feature importance provides clinical insights

### Technical Excellence
- **Scalable Architecture**: Efficient handling of large healthcare datasets
- **Production-Ready**: Clean, documented code suitable for deployment
- **Performance Optimized**: Balanced accuracy and computational efficiency

## 🎯 Results and Impact

This project demonstrates the successful application of machine learning in healthcare, achieving:
- High predictive accuracy for diabetes mellitus in ICU patients
- Clinically relevant feature identification
- Scalable methodology applicable to similar healthcare prediction tasks

The model can assist healthcare professionals in:
- Early diabetes risk identification
- Proactive patient care planning
- Resource optimization in ICU settings
- Evidence-based clinical decision making

## 🏆 Competition Context

Developed for the **WiDS Datathon 2021**, this project showcases:
- Advanced machine learning techniques in healthcare
- Rigorous data science methodology
- Real-world clinical application potential
- Comprehensive model validation and tuning

## 📈 Future Enhancements

Potential improvements and extensions:
- Integration of additional clinical features
- Ensemble methods combining multiple algorithms
- Real-time prediction capabilities
- Clinical validation studies
- Integration with electronic health records (EHR) systems

---

*This project demonstrates the power of machine learning in healthcare, providing a robust foundation for diabetes prediction in critical care settings.*
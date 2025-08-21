# Healthcare Premium Prediction System

A machine learning-powered web application that predicts healthcare insurance premiums based on various personal and health factors. The system uses age-segmented models to provide more accurate predictions for different demographic groups.

## 🚀 Features

- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Age-Segmented Prediction**: Separate models optimized for young adults (≤25) and general population (>25)
- **Comprehensive Risk Assessment**: Includes medical history, genetic risk, and lifestyle factors
- **Real-time Predictions**: Instant premium calculations based on user inputs

## 🌐 Live Demo

**Try the application now:** [Healthcare Premium Predictor](https://healthcare-premium-prediction-27.streamlit.app/)

_Experience the full functionality of the healthcare premium prediction system with our hosted Streamlit application. No installation required!_

## 📊 Prediction Factors

The system considers the following factors for premium calculation:

### Personal Information

- **Age**: Primary factor for model selection and risk assessment
- **Gender**: Male/Female
- **Marital Status**: Married/Unmarried
- **Number of Dependants**: 0-20 dependants
- **Income**: Annual income in lakhs (INR)

### Health & Risk Factors

- **BMI Category**: Normal, Overweight, Obesity, Underweight
- **Medical History**:
  - No Disease
  - Diabetes
  - High Blood Pressure
  - Heart Disease
  - Thyroid
  - Multiple conditions combinations
- **Genetic Risk**: Scale of 0-5
- **Smoking Status**: No Smoking, Occasional, Regular

### Insurance & Employment

- **Insurance Plan**: Bronze, Silver, Gold
- **Employment Status**: Salaried, Self-Employed, Freelancer
- **Region**: Northwest, Southeast, Northeast, Southwest

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib
- **Development Environment**: Jupyter Notebooks

## 📁 Project Structure

```
├── app/
│   ├── main.py                 # Streamlit web application
│   ├── prediction_helper.py    # ML prediction logic and preprocessing
│   └── artifacts/              # Trained models and scalers
│       ├── model_young.joblib  # Model for age ≤ 25
│       ├── model_rest.joblib   # Model for age > 25
│       ├── scaler_young.joblib # Scaler for ≤ 25
│       └── scaler_rest.joblib  # Scaler for age > 25
├── artifacts/                  # Model artifacts
├── data_segmentation.ipynb     # Data analysis and segmentation
├── healthcare_premium_prediction.ipynb      # Main model development
├── healthcare_premium_prediction_young.ipynb    # age ≤ 25 model
├── healthcare_premium_prediction_rest.ipynb     # age > 25 population model
├── healthcare_premium_prediction_young_with_gr.ipynb  # age ≤ 25 model with genetic risk
├── healthcare_premium_prediction_rest_with_gr.ipynb   # age > 25 model with genetic risk

```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Abi27052000/HealthCare-Premium-Prediction.git
   cd HealthCare-Premium-Prediction
   ```

2. **Install required packages**

   ```bash
   pip install streamlit pandas numpy scikit-learn joblib statsmodels matplotlib seaborn xgboost ipykernel
   ```

3. **Run the application**

   ```bash
   cd app
   streamlit run ./main.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8501`

## 💡 How It Works

### Model Architecture

1. **Data Preprocessing**:

   - Categorical variables are one-hot encoded and label encoded
   - Numerical features are scaled using StandardScaler
   - Medical history is converted to normalized risk scores

2. **Age-Based Model Selection**:

   - **Young Model** (Age ≤ 25): Optimized for younger people with different risk patterns
   - **General Model** (Age > 25): Trained on broader population data

3. **Risk Scoring System**:

   - Medical conditions are assigned weighted risk scores
   - Multiple conditions are combined for comprehensive risk assessment
   - Scores are normalized to 0-1 scale

4. **Prediction Pipeline**:
   - Input validation and preprocessing
   - Feature engineering and scaling
   - Model selection based on age
   - Premium prediction and formatting

### Key Features of the Prediction System

- **Normalized Risk Scoring**: Medical history is converted to numerical risk scores
- **Feature Engineering**: Categorical variables are properly encoded
- **Scalable Architecture**: Easy to retrain with new data
- **Model Versioning**: Separate models for different demographics

## 📈 Model Performance

The system uses separate models for different age groups to improve prediction accuracy:

- **Young Adult Model**: Specialized for ages 18-25
- **General Population Model**: Covers ages 26-100

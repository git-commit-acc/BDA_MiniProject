# ClimateScope - Climate Change Trends and Rainfall Prediction

![ClimateScope](https://img.shields.io/badge/ClimateScope-Big%20Data%20Project-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

## 📋 Project Overview

ClimateScope India is a comprehensive big data analytics project that analyzes historical climate data to predict rainfall patterns for 2025 across Indian states and districts. Using machine learning techniques and interactive visualizations, this project provides insights into climate trends and supports data-driven decision making for agriculture, policy, and research.

## 👥 Team Members

- **Ajinkya** (G24AI1046)
- **Ramashankar Mishra** (G24AI1056)

## 🎯 Key Features

- **Data Integration**: Merges multiple CSV files from various sources
- **Machine Learning**: Random Forest model for rainfall prediction
- **Interactive Dashboard**: Streamlit-based web application
- **Predictive Analytics**: 2025 rainfall predictions for all districts
- **Risk Assessment**: Identifies drought and flood risk areas
- **Performance Metrics**: Model evaluation with MAE, RMSE, and R² scores

## 🛠️ Technology Stack

### Technologies Used
- **Backend**: Python 3.9+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest)
- **Visualization**: Plotly, Plotly Express
- **Deployment**: GitHub Actions, Streamlit Cloud

## 🚀 Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Step-by-step Installation

1. **Extract the project files**
   ```bash
   unzip climatescope-project.zip
   cd climatescope-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place your CSV files (10-12 files) in the `data/` folder
   - Ensure CSV files have columns: State, District, Date, Year, Month, Avg_rainfall

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser and go to `http://localhost:8501`

## 📊 Data Format

Your CSV files should follow this format:

```csv
State,District,Date,Year,Month,Avg_rainfall
Karnataka,Dakshina Kannada,1/3/2024,2024,1,0.899774132
Kerala,Idukki,1/3/2024,2024,1,0.271220996
```

## 🔧 Usage Guide

### 1. Data Loading
- Navigate to "Data Loading" page
- Click "Load and Merge Data" to process all CSV files

### 2. Model Training
- Go to "Model Training" page
- Click "Train Rainfall Prediction Model"

### 3. Generate Predictions
- Visit "2025 Predictions" page
- Click "Generate 2025 Predictions"

### 4. Visualizations
- Access "Visualizations" page for interactive charts

### 5. Insights
- Check "Insights" page for key findings and risk assessment

## 🤖 Machine Learning Model

### Algorithm: Random Forest Regressor
- **Estimators**: 100 trees
- **Max Depth**: 10
- **Features**: State, District, Year, Month, Season, Monsoon, Previous Year Rainfall

## 📈 Key Deliverables

1. **Interactive Dashboard** - Multi-page Streamlit application
2. **Machine Learning Model** - Random Forest for rainfall prediction
3. **Complete Codebase** - GitHub repository with documentation
4. **Benchmarking Study** - Model performance evaluation

## 🎨 Visualizations

- Top 10 highest/lowest rainfall predictions
- Monthly rainfall distribution
- State-wise rainfall comparison
- Risk assessment charts

## 🔮 Future Enhancements

- Real-time weather API integration
- Advanced deep learning models
- Geospatial mapping
- Mobile-responsive design

---

**Course**: Big Data Management  
**Project Status**: Production Ready 🚀
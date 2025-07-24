# ClimateScope - Climate Change Trends and Rainfall Prediction
# Big Data Management Course Project

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import glob
import os
from datetime import datetime, timedelta
import joblib

warnings.filterwarnings('ignore')

class ClimateScope:
    def __init__(self):
        self.data = None
        self.model = None
        self.le_state = LabelEncoder()
        self.le_district = LabelEncoder()
        self.predictions_2025 = None
        
    def load_and_merge_data(self, data_folder='data'):
        """Load and merge all CSV files from data folder"""
        try:
            # Get all CSV files from data folder
            csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
            
            if not csv_files:
                st.error("No CSV files found in data folder!")
                return None
            
            # Read and combine all CSV files
            dataframes = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    dataframes.append(df)
                    st.info(f"Loaded {file}: {len(df)} records")
                except Exception as e:
                    st.warning(f"Error reading {file}: {str(e)}")
            
            # Combine all dataframes
            if dataframes:
                combined_data = pd.concat(dataframes, ignore_index=True)
                
                # Clean and standardize column names
                combined_data.columns = combined_data.columns.str.strip()
                
                # Ensure required columns exist
                required_columns = ['State', 'District', 'Date', 'Year', 'Month', 'Avg_rainfall']
                missing_columns = [col for col in required_columns if col not in combined_data.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    return None
                
                # Data cleaning
                combined_data['Avg_rainfall'] = pd.to_numeric(combined_data['Avg_rainfall'], errors='coerce')
                combined_data = combined_data.dropna(subset=['Avg_rainfall'])
                combined_data = combined_data[combined_data['Avg_rainfall'] >= 0]  # Remove negative rainfall
                
                # Convert Date to datetime
                combined_data['Date'] = pd.to_datetime(combined_data['Date'], errors='coerce')
                
                self.data = combined_data
                st.success(f"Successfully merged {len(csv_files)} files. Total records: {len(combined_data)}")
                return combined_data
            else:
                st.error("No valid data found!")
                return None
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # Encode categorical variables
        features_df['State_encoded'] = self.le_state.fit_transform(features_df['State'])
        features_df['District_encoded'] = self.le_district.fit_transform(features_df['District'])
        
        # Create additional time-based features
        features_df['Season'] = features_df['Month'].apply(self.get_season)
        features_df['Is_Monsoon'] = features_df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
        
        # Create lag features (previous year's rainfall)
        features_df = features_df.sort_values(['State', 'District', 'Year', 'Month'])
        features_df['Prev_Year_Rainfall'] = features_df.groupby(['State', 'District', 'Month'])['Avg_rainfall'].shift(1)
        
        # Fill missing lag features with mean
        features_df['Prev_Year_Rainfall'] = features_df['Prev_Year_Rainfall'].fillna(features_df['Avg_rainfall'].mean())
        
        return features_df
    
    def get_season(self, month):
        """Map month to season"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Summer
        elif month in [6, 7, 8, 9]:
            return 2  # Monsoon
        else:
            return 3  # Post-Monsoon
    
    def train_model(self):
        """Train Random Forest model for rainfall prediction"""
        if self.data is None:
            st.error("No data available for training!")
            return None
        
        # Prepare features
        features_df = self.prepare_features(self.data)
        
        # Select features for training
        feature_columns = ['State_encoded', 'District_encoded', 'Year', 'Month', 
                          'Season', 'Is_Monsoon', 'Prev_Year_Rainfall']
        
        X = features_df[feature_columns]
        y = features_df['Avg_rainfall']
        
        # Remove rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        joblib.dump(self.model, 'rainfall_model.pkl')
        joblib.dump(self.le_state, 'state_encoder.pkl')
        joblib.dump(self.le_district, 'district_encoder.pkl')
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'feature_importance': dict(zip(feature_columns, self.model.feature_importances_))
        }
    
    def predict_2025_rainfall(self):
        """Predict rainfall for 2025"""
        if self.model is None:
            st.error("Model not trained yet!")
            return None
        
        # Get unique states and districts from training data
        unique_combinations = self.data[['State', 'District']].drop_duplicates()
        
        predictions = []
        
        for _, row in unique_combinations.iterrows():
            state = row['State']
            district = row['District']
            
            # Get historical data for this location
            location_data = self.data[(self.data['State'] == state) & 
                                    (self.data['District'] == district)]
            
            if len(location_data) == 0:
                continue
            
            # Calculate average previous year rainfall for each month
            monthly_avg = location_data.groupby('Month')['Avg_rainfall'].mean()
            
            for month in range(1, 13):
                try:
                    state_encoded = self.le_state.transform([state])[0]
                    district_encoded = self.le_district.transform([district])[0]
                    
                    prev_rainfall = monthly_avg.get(month, location_data['Avg_rainfall'].mean())
                    
                    features = [[
                        state_encoded,
                        district_encoded,
                        2025,
                        month,
                        self.get_season(month),
                        1 if month in [6, 7, 8, 9] else 0,
                        prev_rainfall
                    ]]
                    
                    prediction = self.model.predict(features)[0]
                    prediction = max(0, prediction)  # Ensure non-negative rainfall
                    
                    predictions.append({
                        'State': state,
                        'District': district,
                        'Month': month,
                        'Predicted_Rainfall': prediction
                    })
                    
                except ValueError:
                    # Skip if state/district not in training data
                    continue
        
        self.predictions_2025 = pd.DataFrame(predictions)
        return self.predictions_2025
    
    def create_visualizations(self):
        """Create interactive visualizations"""
        if self.predictions_2025 is None:
            st.error("No predictions available!")
            return
        
        # Calculate annual rainfall for each location
        annual_rainfall = self.predictions_2025.groupby(['State', 'District'])['Predicted_Rainfall'].sum().reset_index()
        annual_rainfall = annual_rainfall.sort_values('Predicted_Rainfall', ascending=False)
        
        # Top 10 highest predicted rainfall locations
        top_10_highest = annual_rainfall.head(10)
        
        # Top 10 lowest predicted rainfall locations
        top_10_lowest = annual_rainfall.tail(10)
        
        # Visualization 1: Top 10 Highest Rainfall Predictions
        fig1 = px.bar(
            top_10_highest, 
            x='Predicted_Rainfall', 
            y=[f"{row['District']}, {row['State']}" for _, row in top_10_highest.iterrows()],
            orientation='h',
            title='Top 10 Districts with Highest Predicted Rainfall in 2025',
            labels={'Predicted_Rainfall': 'Annual Rainfall (mm)', 'y': 'District, State'},
            color='Predicted_Rainfall',
            color_continuous_scale='Blues'
        )
        fig1.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        
        # Visualization 2: Top 10 Lowest Rainfall Predictions
        fig2 = px.bar(
            top_10_lowest, 
            x='Predicted_Rainfall', 
            y=[f"{row['District']}, {row['State']}" for _, row in top_10_lowest.iterrows()],
            orientation='h',
            title='Top 10 Districts with Lowest Predicted Rainfall in 2025',
            labels={'Predicted_Rainfall': 'Annual Rainfall (mm)', 'y': 'District, State'},
            color='Predicted_Rainfall',
            color_continuous_scale='Reds'
        )
        fig2.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        
        # Visualization 3: Monthly Rainfall Distribution
        monthly_dist = self.predictions_2025.groupby('Month')['Predicted_Rainfall'].mean().reset_index()
        fig3 = px.line(
            monthly_dist, 
            x='Month', 
            y='Predicted_Rainfall',
            title='Average Monthly Rainfall Distribution Across India - 2025',
            labels={'Predicted_Rainfall': 'Average Rainfall (mm)', 'Month': 'Month'},
            markers=True
        )
        fig3.update_layout(height=400)
        
        # Visualization 4: State-wise Rainfall Distribution
        state_rainfall = self.predictions_2025.groupby('State')['Predicted_Rainfall'].sum().reset_index()
        state_rainfall = state_rainfall.sort_values('Predicted_Rainfall', ascending=False)
        
        fig4 = px.bar(
            state_rainfall, 
            x='State', 
            y='Predicted_Rainfall',
            title='State-wise Total Predicted Rainfall in 2025',
            labels={'Predicted_Rainfall': 'Total Rainfall (mm)', 'State': 'State'},
            color='Predicted_Rainfall',
            color_continuous_scale='Viridis'
        )
        fig4.update_layout(height=500, xaxis_tickangle=-45)
        
        return fig1, fig2, fig3, fig4
    
    def generate_insights(self):
        """Generate insights from predictions"""
        if self.predictions_2025 is None:
            return "No predictions available for insights generation."
        
        annual_rainfall = self.predictions_2025.groupby(['State', 'District'])['Predicted_Rainfall'].sum().reset_index()
        
        insights = []
        insights.append(f"📊 **Total Locations Analyzed**: {len(annual_rainfall)}")
        insights.append(f"💧 **Average Annual Rainfall Predicted**: {annual_rainfall['Predicted_Rainfall'].mean():.2f} mm")
        insights.append(f"🏆 **Highest Rainfall Expected**: {annual_rainfall['Predicted_Rainfall'].max():.2f} mm in {annual_rainfall.loc[annual_rainfall['Predicted_Rainfall'].idxmax(), 'District']}, {annual_rainfall.loc[annual_rainfall['Predicted_Rainfall'].idxmax(), 'State']}")
        insights.append(f"🏜️ **Lowest Rainfall Expected**: {annual_rainfall['Predicted_Rainfall'].min():.2f} mm in {annual_rainfall.loc[annual_rainfall['Predicted_Rainfall'].idxmin(), 'District']}, {annual_rainfall.loc[annual_rainfall['Predicted_Rainfall'].idxmin(), 'State']}")
        
        # Monsoon analysis
        monsoon_months = self.predictions_2025[self.predictions_2025['Month'].isin([6, 7, 8, 9])]
        monsoon_rainfall = monsoon_months.groupby(['State', 'District'])['Predicted_Rainfall'].sum().mean()
        insights.append(f"🌧️ **Average Monsoon Rainfall (Jun-Sep)**: {monsoon_rainfall:.2f} mm")
        
        return "\n".join(insights)

# Initialize session state
def init_session_state():
    if 'climate_scope' not in st.session_state:
        st.session_state.climate_scope = ClimateScope()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'predictions_generated' not in st.session_state:
        st.session_state.predictions_generated = False

def main():
    st.set_page_config(
        page_title="ClimateScope - Rainfall Prediction 2025",
        page_icon="🌧️",
        layout="wide"
    )
    
    st.title("🌧️ ClimateScope - Climate Change Trends and Rainfall Prediction")
    st.markdown("### Big Data Management Course Project")
    st.markdown("**Team**: Ajinkya (G24AI1046), Ramashankar Mishra (G24AI1056)")
    
    # Initialize session state
    init_session_state()
    
    # Get ClimateScope instance from session state
    climate_scope = st.session_state.climate_scope
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Data Loading", "Model Training", "2025 Predictions", "Visualizations", "Insights"])
    
    # Status indicators in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:**")
    st.sidebar.markdown(f"📁 Data Loaded: {'✅' if st.session_state.data_loaded else '❌'}")
    st.sidebar.markdown(f"🤖 Model Trained: {'✅' if st.session_state.model_trained else '❌'}")
    st.sidebar.markdown(f"🔮 Predictions Generated: {'✅' if st.session_state.predictions_generated else '❌'}")
    
    if page == "Data Loading":
        st.header("📁 Data Loading and Processing")
        
        if st.button("Load and Merge Data"):
            with st.spinner("Loading and merging data files..."):
                data = climate_scope.load_and_merge_data()
                
                if data is not None:
                    st.session_state.data_loaded = True
                    st.subheader("Data Overview")
                    st.write(f"**Total Records**: {len(data)}")
                    st.write(f"**Date Range**: {data['Year'].min()} - {data['Year'].max()}")
                    st.write(f"**States**: {data['State'].nunique()}")
                    st.write(f"**Districts**: {data['District'].nunique()}")
                    
                    st.subheader("Sample Data")
                    st.dataframe(data.head(10))
                    
                    st.subheader("Data Statistics")
                    st.write(data['Avg_rainfall'].describe())
                else:
                    st.session_state.data_loaded = False
    
    elif page == "Model Training":
        st.header("🤖 Machine Learning Model Training")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first!")
        else:
            if st.button("Train Rainfall Prediction Model"):
                with st.spinner("Training Random Forest model..."):
                    results = climate_scope.train_model()
                    
                    if results:
                        st.session_state.model_trained = True
                        st.success("Model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Absolute Error", f"{results['mae']:.3f}")
                        with col2:
                            st.metric("R² Score", f"{results['r2']:.3f}")
                        with col3:
                            st.metric("RMSE", f"{np.sqrt(results['mse']):.3f}")
                        
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame(
                            list(results['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='Feature', y='Importance',
                                   title='Feature Importance in Rainfall Prediction')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.session_state.model_trained = False
    
    elif page == "2025 Predictions":
        st.header("🔮 Rainfall Predictions for 2025")
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
        else:
            if st.button("Generate 2025 Predictions"):
                with st.spinner("Generating predictions for 2025..."):
                    predictions = climate_scope.predict_2025_rainfall()
                    
                    if predictions is not None:
                        st.session_state.predictions_generated = True
                        st.success(f"Generated predictions for {len(predictions)} location-month combinations!")
                        
                        # Show sample predictions
                        st.subheader("Sample Predictions")
                        st.dataframe(predictions.head(20))
                        
                        # Download predictions
                        csv = predictions.to_csv(index=False)
                        st.download_button(
                            label="Download Full Predictions as CSV",
                            data=csv,
                            file_name="rainfall_predictions_2025.csv",
                            mime="text/csv"
                        )
                    else:
                        st.session_state.predictions_generated = False
    
    elif page == "Visualizations":
        st.header("📊 Interactive Visualizations")
        
        if not st.session_state.predictions_generated:
            st.warning("Please generate 2025 predictions first!")
        else:
            with st.spinner("Creating visualizations..."):
                fig1, fig2, fig3, fig4 = climate_scope.create_visualizations()
                
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig3, use_container_width=True)
                st.plotly_chart(fig4, use_container_width=True)
    
    elif page == "Insights":
        st.header("💡 Key Insights")
        
        if not st.session_state.predictions_generated:
            st.warning("Please generate 2025 predictions first!")
        else:
            insights = climate_scope.generate_insights()
            st.markdown(insights)
            
            # Additional analysis
            st.subheader("Detailed Analysis")
            
            annual_rainfall = climate_scope.predictions_2025.groupby(['State', 'District'])['Predicted_Rainfall'].sum().reset_index()
            
            # Risk assessment
            low_rainfall_threshold = annual_rainfall['Predicted_Rainfall'].quantile(0.25)
            high_rainfall_threshold = annual_rainfall['Predicted_Rainfall'].quantile(0.75)
            
            drought_risk = annual_rainfall[annual_rainfall['Predicted_Rainfall'] < low_rainfall_threshold]
            flood_risk = annual_rainfall[annual_rainfall['Predicted_Rainfall'] > high_rainfall_threshold]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏜️ Drought Risk Areas")
                st.write(f"Districts with rainfall < {low_rainfall_threshold:.2f} mm")
                st.dataframe(drought_risk[['State', 'District', 'Predicted_Rainfall']].head(10))
            
            with col2:
                st.subheader("🌊 Flood Risk Areas")
                st.write(f"Districts with rainfall > {high_rainfall_threshold:.2f} mm")
                st.dataframe(flood_risk[['State', 'District', 'Predicted_Rainfall']].head(10))

if __name__ == "__main__":
    main()
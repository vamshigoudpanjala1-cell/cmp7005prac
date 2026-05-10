import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set page config
st.set_page_config(layout="wide", page_title="Beijing Air Quality Analysis")

# --- Load Models and Scaler ---
@st.cache_resource
def load_artifacts():
    artifacts = {}
    # Load models, including the 'needs_scaling' flag
    model_dir = 'models'
    artifacts['lr'] = joblib.load(os.path.join(model_dir, 'linear_regression.joblib'))
    artifacts['dt'] = joblib.load(os.path.join(model_dir, 'decision_tree.joblib'))
    artifacts['rf'] = joblib.load(os.path.join(model_dir, 'random_forest.joblib'))
    artifacts['knn'] = joblib.load(os.path.join(model_dir, 'knn.joblib'))
    artifacts['scaler'] = joblib.load('scaler.joblib')
    artifacts['feature_columns'] = joblib.load('feature_columns.joblib')
    artifacts['model_comparison_df'] = pd.read_csv('model_comparison.csv').set_index('Model')
    artifacts['df_merged'] = pd.read_csv('df_merged.csv')
    return artifacts

artifacts = load_artifacts()

# Extract individual components
lr_model_data = artifacts['lr']
dt_model_data = artifacts['dt']
rf_model_data = artifacts['rf']
knn_model_data = artifacts['knn']
scaler = artifacts['scaler']
feature_columns = artifacts['feature_columns']
model_comparison_df = artifacts['model_comparison_df']
df_merged = artifacts['df_merged']

# --- Streamlit UI ---
st.title("Beijing Air Quality Analysis and Prediction")
st.markdown("An interactive dashboard for exploring air quality data and predicting PM2.5 levels using various machine learning models.")

# Navigation
page = st.sidebar.radio("Navigation", ["Data Exploration", "PM2.5 Prediction", "Model Performance"])

if page == "Data Exploration":
    st.header("Data Overview")
    st.write("Displaying a sample of the cleaned, merged dataset. This includes urban and suburban stations, various pollutants, and meteorological data.")

    # Display a sample of the original df_merged
    st.subheader("Sample Data")
    st.dataframe(df_merged.head())
    st.write("*(Note: Full dataset too large to display, showing only the head)*")

    st.subheader("Correlation Heatmap")
    st.image('correlation_heatmap.png', caption='Correlation Heatmap of Air Pollutants and Meteorological Variables')
    st.write("The heatmap shows strong positive correlations among primary pollutants (PM2.5, PM10, CO, NO2) indicating common emission sources. Wind Speed (WSPM) shows a distinct negative correlation with PM2.5, demonstrating its role in pollutant dispersion.")

elif page == "PM2.5 Prediction":
    st.header("Predict PM2.5 Levels")
    st.write("Enter the current environmental conditions to get a PM2.5 prediction.")

    # Input features (simplified for demonstration, add more as needed)
    with st.form("prediction_form"):
        st.subheader("Input Current Conditions")
        col1, col2, col3 = st.columns(3)
        with col1:
            SO2 = st.slider("SO2 (µg/m³)", 0.0, 300.0, 10.0)
            NO2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 40.0)
            CO = st.slider("CO (µg/m³)", 0.0, 5000.0, 800.0)
        with col2:
            O3 = st.slider("O3 (µg/m³)", 0.0, 300.0, 50.0)
            TEMP = st.slider("Temperature (°C)", -20.0, 40.0, 15.0)
            PRES = st.slider("Pressure (hPa)", 980.0, 1040.0, 1010.0)
        with col3:
            DEWP = st.slider("Dew Point (°C)", -30.0, 30.0, 5.0)
            RAIN = st.slider("Rain (mm)", 0.0, 50.0, 0.0)
            WSPM = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)

        # Categorical inputs
        st.subheader("Categorical Features")
        col_cat1, col_cat2, col_cat3 = st.columns(3)
        with col_cat1:
            wd = st.selectbox("Wind Direction", ['N', 'ENE', 'ESE', 'E', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW'])
        with col_cat2:
            Station_Type = st.selectbox("Station Type", ['Urban', 'Suburban'])
        with col_cat3:
            season = st.selectbox("Season", ['Spring', 'Summer', 'Autumn', 'Winter'])

        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
        is_weekend = 1 if day_of_week >= 5 else 0

        model_choice = st.selectbox("Select Model for Prediction", list(model_comparison_df.index))

        submitted = st.form_submit_button("Predict PM2.5")

    if submitted:
        input_data = {
            'SO2': SO2, 'NO2': NO2, 'CO': CO, 'O3': O3, 'TEMP': TEMP,
            'PRES': PRES, 'DEWP': DEWP, 'RAIN': RAIN, 'WSPM': WSPM,
            'day_of_week': day_of_week, 'is_weekend': is_weekend,
            'wd': wd, 'Station_Type': Station_Type, 'season': season
        }

        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical features, aligning with training columns
        input_encoded = pd.get_dummies(input_df, columns=['wd', 'Station_Type', 'season'], drop_first=True)

        # Ensure all feature_columns from training are present, fill missing with 0
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_columns] # Reorder columns to match training

        # Select the chosen model and its scaling requirement
        model_mapping = {
            'Linear Regression': lr_model_data,
            'Decision Tree': dt_model_data,
            'Random Forest': rf_model_data,
            'KNN (k=10)': knn_model_data
        }
        selected_model_info = model_mapping[model_choice]
        model = selected_model_info['model']
        needs_scaling = selected_model_info['needs_scaling']

        if needs_scaling:
            input_scaled = scaler.transform(input_encoded)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_encoded)[0]

        st.subheader(f"Predicted PM2.5 (using {model_choice}):")
        st.success(f"{prediction:.2f} µg/m³")

elif page == "Model Performance":
    st.header("Machine Learning Model Performance")
    st.write("Comparison of different regression models for PM2.5 prediction.")

    st.subheader("Evaluation Metrics")
    st.dataframe(model_comparison_df)

    st.subheader("Metric Comparison Plots")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [('MAE', 'lower is better'),
               ('RMSE', 'lower is better'),
               ('R²', 'higher is better')]
    colors = ['#4C72B0', '#DD8452', '#55A467', '#C44E52'] # Example colors

    for ax, (m, note) in zip(axes, metrics):
        bars = ax.bar(model_comparison_df.index, model_comparison_df[m], color=colors)
        ax.set_title(f'{m} ({note})')
        ax.set_ylabel(m)
        ax.tick_params(axis='x', rotation=20)
        for b, v in zip(bars, model_comparison_df[m]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f'{v}', ha='center', va='bottom', fontsize=9)
    st.pyplot(fig)

    st.subheader("Random Forest Feature Importance")
    st.image('rf_feature_importance.png', caption='Top 15 features driving PM2.5 prediction') # Assuming this is saved
    st.write("CO dominates feature importance, followed by NO2 and Dew Point. This highlights the strong influence of co-emitted pollutants and meteorological conditions on PM2.5 levels.")




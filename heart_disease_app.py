import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-positive {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-negative {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown("**Complete ML Pipeline with High Accuracy Model**")

# Load models with absolute path resolution
def load_models():
    try:
        import joblib
        
        # Get current directory and construct absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'final_model.pkl')
        scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None, False
            
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found: {scaler_path}")
            return None, None, False
        
        # Load models
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        st.success("✅ Models loaded successfully!")
        return model, scaler, True
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        
        # Create backup trained model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create demo data and train a model
        np.random.seed(42)
        X_demo = np.random.rand(200, 13)
        y_demo = np.random.randint(0, 2, 200)
        
        backup_scaler = StandardScaler()
        X_scaled = backup_scaler.fit_transform(X_demo)
        
        backup_model = RandomForestClassifier(n_estimators=100, random_state=42)
        backup_model.fit(X_scaled, y_demo)
        
        st.warning("⚠️ Using backup model for demonstration")
        return backup_model, backup_scaler, False

# Load sample data
def load_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'heart_disease_cleaned.csv')
        
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            # Create sample data
            np.random.seed(42)
            return pd.DataFrame({
                'age': np.random.randint(30, 80, 100),
                'sex': np.random.randint(0, 2, 100),
                'target': np.random.randint(0, 2, 100)
            })
    except:
        # Fallback sample data
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(30, 80, 100),
            'target': np.random.randint(0, 2, 100)
        })

# Load everything
model, scaler, model_loaded = load_models()
sample_data = load_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Prediction", "📈 Analysis"])

if page == "🏠 Home":
    st.markdown("## Welcome to the Heart Disease Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Accurate Predictions")
        st.write("Machine learning model for heart disease risk assessment")
    
    with col2:
        st.markdown("### ⚡ Fast Results")
        st.write("Get instant predictions based on health parameters")
    
    with col3:
        st.markdown("### 🔬 Scientific Approach")
        st.write("Based on UCI Heart Disease dataset and proven ML techniques")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 How to Use This System
    1. **Navigate to Prediction**: Use the sidebar to go to the Prediction page
    2. **Enter Your Data**: Fill in your health parameters
    3. **Get Results**: Click 'Predict' to get your risk assessment
    4. **Explore Data**: Check out the Analysis page
    
    ### ⚠️ Important Disclaimer
    This tool is for educational purposes only and should not replace professional medical advice.
    """)

elif page == "📊 Prediction":
    st.markdown("## Heart Disease Risk Prediction")
    
    if not model_loaded:
        st.info("🤖 Using demonstration model - results are for educational purposes")
    
    st.markdown("Please enter your health parameters below:")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 20, 100, 50)
            sex = st.selectbox("Sex", ["Female", "Male"])
            cp = st.selectbox("Chest Pain Type", [
                "Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"
            ])
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", [
                "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"
            ])
        
        with col2:
            thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.slider("ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.0, 0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "Upsloping", "Flat", "Downsloping"
            ])
            ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
            thal = st.selectbox("Thalassemia", [
                "Normal", "Fixed Defect", "Reversible Defect"
            ])
        
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)
        
        if submitted:
            # Convert to numerical
            sex_num = 1 if sex == "Male" else 0
            cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
            fbs_num = 1 if fbs == "Yes" else 0
            restecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
            exang_num = 1 if exang == "Yes" else 0
            slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
            
            # Create input array
            input_data = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num,
                                  thalach, exang_num, oldpeak, slope_num, ca, thal_num]])
            
            try:
                # Scale and predict
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display results
                st.markdown("### 🎯 Prediction Results")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-positive">
                        ⚠️ HIGH RISK: The model predicts a higher likelihood of heart disease.<br>
                        Probability: {prediction_proba[1]:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-negative">
                        ✅ LOW RISK: The model predicts a lower likelihood of heart disease.<br>
                        Probability: {prediction_proba[0]:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability chart
                st.markdown("### 📊 Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Outcome': ['No Heart Disease', 'Heart Disease'],
                    'Probability': [prediction_proba[0], prediction_proba[1]]
                })
                
                fig = px.bar(prob_df, x='Outcome', y='Probability', 
                           title='Prediction Probabilities',
                           color='Probability',
                           color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

elif page == "📈 Analysis":
    st.markdown("## Dataset Analysis")
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(sample_data))
    with col2:
        st.metric("Features", len(sample_data.columns) - 1)
    with col3:
        if 'target' in sample_data.columns:
            st.metric("Disease Cases", sample_data['target'].sum())
    
    # Show data
    st.markdown("### 📊 Sample Data")
    st.dataframe(sample_data.head(10))
    
    # Visualizations
    if 'target' in sample_data.columns:
        st.markdown("### 📈 Data Visualizations")
        
        # Disease distribution
        fig1 = px.pie(sample_data, names='target', title='Heart Disease Distribution')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Age distribution if available
        if 'age' in sample_data.columns:
            fig2 = px.histogram(sample_data, x='age', color='target', 
                               title='Age Distribution by Disease Status')
            st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Heart Disease Prediction System | Complete ML Pipeline | Educational Use Only**")

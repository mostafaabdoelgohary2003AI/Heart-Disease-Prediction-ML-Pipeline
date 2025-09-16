import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
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

# Main title
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)

# Load models with error handling for Streamlit Cloud
@st.cache_resource
def load_models():
    try:
        import joblib
        model = joblib.load('models/final_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler, True
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        # Create a dummy model for demo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        dummy_model = RandomForestClassifier(random_state=42)
        dummy_scaler = StandardScaler()
        # Fit with dummy data
        dummy_X = np.random.rand(100, 13)
        dummy_y = np.random.randint(0, 2, 100)
        dummy_scaler.fit(dummy_X)
        dummy_model.fit(dummy_scaler.transform(dummy_X), dummy_y)
        return dummy_model, dummy_scaler, False

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('data/heart_disease_cleaned.csv')
    except:
        # Create sample data for demo
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(30, 80, 100),
            'sex': np.random.randint(0, 2, 100),
            'cp': np.random.randint(0, 4, 100),
            'trestbps': np.random.randint(90, 180, 100),
            'chol': np.random.randint(150, 350, 100),
            'fbs': np.random.randint(0, 2, 100),
            'restecg': np.random.randint(0, 3, 100),
            'thalach': np.random.randint(80, 200, 100),
            'exang': np.random.randint(0, 2, 100),
            'oldpeak': np.random.uniform(0, 6, 100),
            'slope': np.random.randint(0, 3, 100),
            'ca': np.random.randint(0, 4, 100),
            'thal': np.random.randint(0, 3, 100),
            'target': np.random.randint(0, 2, 100)
        })

# Load models and data
model, scaler, model_loaded = load_models()
sample_data = load_sample_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Prediction", "📈 Data Analysis", "ℹ️ About"])

if page == "🏠 Home":
    st.markdown("## Welcome to the Heart Disease Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Accurate Predictions")
        st.write("Our machine learning models achieve high accuracy in predicting heart disease risk.")
    
    with col2:
        st.markdown("### ⚡ Fast Results") 
        st.write("Get instant predictions based on your health parameters.")
    
    with col3:
        st.markdown("### 🔬 Scientific Approach")
        st.write("Based on the UCI Heart Disease dataset and proven ML techniques.")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 How to Use This System
    1. Navigate to Prediction page using the sidebar
    2. Enter your health parameters 
    3. Get your heart disease risk assessment
    4. Explore the data analysis section
    
    ### ⚠️ Important Disclaimer
    This tool is for educational purposes only and should not replace professional medical advice.
    """)

elif page == "📊 Prediction":
    st.markdown("## Heart Disease Risk Prediction")
    
    if not model_loaded:
        st.warning("⚠️ Using demo model. Real model training in progress.")
    
    st.markdown("Please enter your health parameters below:")
    
    # Create input form
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
            # Convert inputs to numerical
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
                        ⚠️ HIGH RISK: Higher likelihood of heart disease.<br>
                        Probability: {prediction_proba[1]:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-negative">
                        ✅ LOW RISK: Lower likelihood of heart disease.<br>
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
                st.info("🤖 Demo mode - please check model files.")

elif page == "📈 Data Analysis":
    st.markdown("## Dataset Analysis")
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(sample_data))
    with col2:
        st.metric("Features", len(sample_data.columns) - 1)
    with col3:
        st.metric("Heart Disease Cases", sample_data['target'].sum())
    with col4:
        st.metric("Healthy Cases", len(sample_data) - sample_data['target'].sum())
    
    # Sample data
    st.markdown("### 🔍 Sample Data")
    st.dataframe(sample_data.head(10))
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    # Target distribution
    fig1 = px.pie(sample_data, names='target', title='Heart Disease Distribution')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Age distribution
    fig2 = px.histogram(sample_data, x='age', color='target', 
                       title='Age Distribution by Heart Disease Status')
    st.plotly_chart(fig2, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("## About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This Heart Disease Prediction System implements multiple machine learning algorithms 
    to predict heart disease likelihood based on clinical parameters.
    
    ### 🔬 Machine Learning Pipeline
    - **Data Preprocessing**: Cleaning and scaling UCI Heart Disease dataset
    - **Feature Engineering**: PCA and feature selection techniques
    - **Model Training**: Logistic Regression, Decision Trees, Random Forest, SVM
    - **Clustering Analysis**: K-Means and Hierarchical clustering
    - **Optimization**: Hyperparameter tuning with GridSearchCV
    
    ### 📊 Performance
    - High accuracy in heart disease prediction
    - Balanced precision and recall
    - Robust cross-validation scores
    
    ### ⚠️ Disclaimer
    This tool is for educational purposes only. Always consult healthcare professionals.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Heart Disease Prediction System | Built with Streamlit | Educational Use Only"
    "</div>", 
    unsafe_allow_html=True
)

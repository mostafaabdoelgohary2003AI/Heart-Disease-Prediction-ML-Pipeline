import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin-bottom: 1rem;
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

# Load models and data (with error handling)
@st.cache_resource
def load_models():
    try:
        # Try to load the final model (will be created after hyperparameter tuning)
        model = joblib.load('models/final_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler, True
    except:
        # If final model doesn't exist, create a placeholder
        st.warning("⚠️ Final model not found. Please run the training notebooks first.")
        return None, None, False

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv('data/heart_disease_cleaned.csv')
    except:
        # Create sample data if file doesn't exist
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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Prediction", "📈 Data Analysis", "ℹ️ About"])

if page == "🏠 Home":
    st.markdown('<h2 class="sub-header">Welcome to the Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate Predictions</h3>
            <p>Our machine learning models achieve high accuracy in predicting heart disease risk.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast Results</h3>
            <p>Get instant predictions based on your health parameters.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Scientific Approach</h3>
            <p>Based on the UCI Heart Disease dataset and proven ML techniques.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 How to Use This System
    
    1. **Navigate to Prediction**: Use the sidebar to go to the Prediction page
    2. **Enter Your Data**: Fill in your health parameters using the input fields
    3. **Get Results**: Click 'Predict' to get your heart disease risk assessment
    4. **Explore Data**: Check out the Data Analysis page to understand the dataset
    
    ### 🏥 Health Parameters We Analyze
    
    - **Age**: Your current age
    - **Sex**: Gender (Male/Female)
    - **Chest Pain Type**: Type of chest pain experienced
    - **Blood Pressure**: Resting blood pressure
    - **Cholesterol**: Serum cholesterol level
    - **And more...**
    
    ### ⚠️ Important Disclaimer
    
    This tool is for educational purposes only and should not replace professional medical advice. 
    Always consult with healthcare professionals for medical decisions.
    """)

elif page == "📊 Prediction":
    st.markdown('<h2 class="sub-header">Heart Disease Risk Prediction</h2>', unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("❌ Model not available. Please run the training notebooks first.")
        st.stop()
    
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
            # Convert categorical inputs to numerical
            sex_num = 1 if sex == "Male" else 0
            cp_num = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
            fbs_num = 1 if fbs == "Yes" else 0
            restecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
            exang_num = 1 if exang == "Yes" else 0
            slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
            
            # Create input array with ALL 13 features in correct order
            # Order: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_data = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]])
            
            # Scale the input data (if scaler is available)
            if scaler:
                input_scaled = scaler.transform(input_data)
            else:
                input_scaled = input_data
            
            # Make prediction (placeholder since model might not be loaded)
            try:
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
                
                # Show probability breakdown
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
                st.error(f"Error making prediction: {str(e)}")
                # Show placeholder result
                st.info("🤖 This is a demo prediction. Please train the models first for real predictions.")

elif page == "📈 Data Analysis":
    st.markdown('<h2 class="sub-header">Dataset Analysis</h2>', unsafe_allow_html=True)
    
    # Display dataset info
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(sample_data))
    with col2:
        st.metric("Features", len(sample_data.columns) - 1)
    with col3:
        st.metric("Heart Disease Cases", sample_data['target'].sum())
    with col4:
        st.metric("Healthy Cases", len(sample_data) - sample_data['target'].sum())
    
    # Show sample data
    st.markdown("### 🔍 Sample Data")
    st.dataframe(sample_data.head(10))
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    # Target distribution
    fig1 = px.pie(sample_data, names='target', title='Heart Disease Distribution',
                  labels={'target': 'Heart Disease', 0: 'No Disease', 1: 'Disease'})
    st.plotly_chart(fig1, use_container_width=True)
    
    # Age distribution by target
    fig2 = px.histogram(sample_data, x='age', color='target', 
                       title='Age Distribution by Heart Disease Status',
                       labels={'target': 'Heart Disease'})
    st.plotly_chart(fig2, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlations")
    corr_matrix = sample_data.corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     title="Feature Correlation Heatmap")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This Heart Disease Prediction System is a comprehensive machine learning project that implements 
    multiple classification algorithms to predict the likelihood of heart disease based on various 
    health parameters.
    
    ### 🔬 Machine Learning Pipeline
    
    Our system includes:
    
    1. **Data Preprocessing**: Cleaning, scaling, and preparing the UCI Heart Disease dataset
    2. **Feature Engineering**: 
       - Principal Component Analysis (PCA) for dimensionality reduction
       - Feature selection using Random Forest importance, RFE, and statistical tests
    3. **Model Training**: Multiple algorithms including:
       - Logistic Regression
       - Decision Trees
       - Random Forest
       - Support Vector Machines
    4. **Model Optimization**: Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
    5. **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
    
    ### 📊 Dataset Information
    
    **Source**: UCI Heart Disease Dataset
    **Samples**: 303 patients
    **Features**: 13 clinical features
    **Target**: Binary classification (Heart Disease: Yes/No)
    
    ### 🏥 Clinical Features
    
    - **age**: Age in years
    - **sex**: Gender (1 = male, 0 = female)
    - **cp**: Chest pain type (4 values)
    - **trestbps**: Resting blood pressure
    - **chol**: Serum cholesterol in mg/dl
    - **fbs**: Fasting blood sugar > 120 mg/dl
    - **restecg**: Resting electrocardiographic results
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slope**: Slope of the peak exercise ST segment
    - **ca**: Number of major vessels (0-3) colored by fluoroscopy
    - **thal**: Thalassemia type
    
    ### 🛠️ Technology Stack
    
    - **Backend**: Python, Scikit-learn, Pandas, NumPy
    - **Frontend**: Streamlit
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Deployment**: Streamlit Cloud, Ngrok
    
    ### ⚠️ Important Notes
    
    - This system is for educational and research purposes only
    - It should not be used as a substitute for professional medical advice
    - Always consult healthcare professionals for medical decisions
    - The predictions are based on statistical patterns in historical data
    
    ### 👨‍💻 Development
    
    This project demonstrates a complete machine learning workflow from data preprocessing 
    to model deployment, showcasing best practices in:
    
    - Data science methodology
    - Model selection and evaluation
    - Feature engineering techniques
    - Web application development
    - Model deployment strategies
    
    ### 📈 Performance
    
    Our models achieve competitive performance metrics:
    - High accuracy in heart disease prediction
    - Balanced precision and recall
    - Robust cross-validation scores
    - Optimized hyperparameters for best performance
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ using Python and Streamlit**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Heart Disease Prediction System | Built with Streamlit | For Educational Purposes Only"
    "</div>", 
    unsafe_allow_html=True
)

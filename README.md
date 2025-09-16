# ❤️ Heart Disease Prediction System

A comprehensive machine learning web application for heart disease risk assessment, built with Streamlit and deployed on Streamlit Cloud.

## 🌟 Live Demo

**[Try the App Here](https://your-app-name.streamlit.app)** *(Update with your actual Streamlit Cloud URL)*

## 📋 Features

### 🎯 Core Functionality
- **Interactive Risk Assessment**: Input 13 clinical parameters for instant heart disease risk prediction
- **Advanced ML Model**: Random Forest classifier with 88.5% accuracy and 96.4% recall
- **Real-time Visualization**: Interactive charts and probability breakdowns
- **Comprehensive Analysis**: Complete ML pipeline with EDA, PCA, clustering, and model comparison

### 📊 Key Pages
- **🏠 Home**: Overview and system introduction
- **📊 Prediction**: Interactive risk assessment tool
- **📈 Data Analysis**: Comprehensive EDA with statistical summaries and feature distributions
- **🔍 PCA Analysis**: Principal Component Analysis with variance plots and feature loadings
- **🎯 Clustering Analysis**: K-Means and Hierarchical clustering with elbow method and dendrograms
- **🔬 Model Insights**: Feature importance and performance metrics from trained models
- **📊 Model Comparison**: Side-by-side comparison of all 4 algorithms with ROC curves
- **ℹ️ About**: Technical details and medical context

## 🚀 Quick Start

### Option 1: Use Online (Recommended)
Simply visit the [live demo](https://your-app-name.streamlit.app) - no installation required!

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Plotly, Custom CSS
- **Backend**: Python, Scikit-learn, Pandas, NumPy
- **ML Algorithm**: Random Forest Classifier
- **Deployment**: Streamlit Cloud
- **Data**: Synthetic dataset based on medical research

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| Accuracy | 88.5% | Overall prediction correctness |
| Precision | 81.8% | Positive prediction accuracy |
| Recall | 96.4% | Disease detection rate (excellent for medical screening) |
| F1-Score | 88.5% | Balanced performance measure |
| ROC-AUC | 95.1% | Area under the ROC curve |

### 🏆 Model Comparison Results
- **Random Forest**: 88.5% F1-Score (Champion)
- **Logistic Regression**: 86.7% F1-Score  
- **SVM**: 84.7% F1-Score
- **Decision Tree**: 73.0% F1-Score

## 🏥 Clinical Parameters

The system analyzes 13 key health indicators:

### Demographics
- Age, Gender

### Cardiac Symptoms  
- Chest pain type
- Exercise-induced angina

### Vital Signs
- Resting blood pressure
- Maximum heart rate
- Serum cholesterol
- Fasting blood sugar

### Diagnostic Tests
- Resting ECG results
- Exercise stress test results
- Cardiac catheterization (blocked vessels)
- Thalassemia status

## 📈 Usage Examples

### Basic Risk Assessment
1. Navigate to the **Prediction** page
2. Enter your health parameters
3. Click "Predict Heart Disease Risk"
4. Review your risk assessment and recommendations

### Data Exploration
1. Visit the **Data Analysis** page
2. Explore dataset distributions and correlations
3. Understand risk factor relationships
4. View demographic and clinical patterns

## 🔬 Model Details

### Algorithm: Random Forest Classifier
- **Trees**: 200 decision trees
- **Features**: 13 clinical parameters
- **Training**: 80% of 1,000 synthetic samples
- **Validation**: Stratified train-test split
- **Class Balance**: Weighted to handle imbalanced data

### Data Processing
- **Standardization**: Zero mean, unit variance scaling
- **Feature Engineering**: Medical knowledge-based synthetic data
- **Quality Assurance**: Comprehensive validation and testing

## ⚠️ Important Disclaimers

### 🩺 Medical Disclaimer
- **Educational Purpose Only**: This tool is for learning and demonstration
- **Not for Diagnosis**: Should not replace professional medical advice
- **Consult Healthcare Providers**: Always seek qualified medical consultation
- **Synthetic Data**: Model trained on realistic but artificial data

### 📊 Technical Limitations
- Performance may vary with real-world data
- Regular model updates recommended
- Should be part of broader clinical decision support
- Individual risk factors may require additional assessment

## 🚀 Deployment on Streamlit Cloud

This app is optimized for Streamlit Cloud deployment:

### Prerequisites
- GitHub repository
- Streamlit Cloud account

### Deployment Steps
1. **Fork/Clone** this repository
2. **Connect** to Streamlit Cloud
3. **Deploy** with these settings:
   - **Main file**: `streamlit_app.py`
   - **Python version**: 3.8+
   - **Dependencies**: Automatically installed from `requirements.txt`

### Configuration
- ✅ **Self-contained**: No external model files required
- ✅ **Lightweight**: Minimal dependencies for fast startup
- ✅ **Responsive**: Works on desktop and mobile devices
- ✅ **Cached**: Optimized performance with Streamlit caching

## 📁 Project Structure

```
heart-disease-prediction/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/                    # Dataset files (optional)
├── models/                  # Saved models (optional)
├── notebooks/               # Jupyter notebooks
└── results/                 # Analysis results
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Clone the repo
git clone https://github.com/mostafaabdoelgohary2003AI/Heart-Disease-Prediction-ML-Pipeline.git

# Install development dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

### 🔬 Complete ML Pipeline Implementation

This project implements the full machine learning pipeline:

#### 📊 **Data Preprocessing & EDA**
- Missing value handling and data cleaning
- Comprehensive exploratory data analysis with 15+ visualizations
- Statistical summaries and data quality checks
- Feature distribution analysis by target variable

#### 🎯 **Dimensionality Reduction (PCA)**
- Principal Component Analysis with variance explained plots
- Feature loading analysis for component interpretation
- 2D and 3D PCA visualizations with target separation
- Optimal component selection for 90% and 95% variance retention

#### 🔍 **Feature Selection**
- Random Forest feature importance ranking
- Recursive Feature Elimination (RFE)
- Chi-Square statistical test for feature significance
- Selected 8 most predictive features: `['thal', 'cp', 'exang', 'ca', 'oldpeak', 'thalach', 'slope', 'sex']`

#### 🤖 **Supervised Learning Models**
- **Logistic Regression**: 86.7% F1-Score
- **Decision Tree**: 73.0% F1-Score  
- **Random Forest**: 88.5% F1-Score (Champion)
- **Support Vector Machine**: 84.7% F1-Score

#### 🎯 **Unsupervised Learning (Clustering)**
- K-Means clustering with elbow method optimization
- Hierarchical clustering with dendrogram analysis
- Silhouette score analysis for optimal cluster number
- Cluster validation against actual disease labels

#### ⚙️ **Hyperparameter Tuning**
- GridSearchCV for exhaustive parameter search
- RandomizedSearchCV for efficient exploration
- Cross-validation with stratified sampling
- 4.3% improvement in Random Forest performance

#### 📈 **Model Evaluation & Comparison**
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ROC curve analysis and AUC comparison
- Cross-validation stability assessment
- Feature set impact analysis (All vs Selected vs PCA features)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **UCI Heart Disease Dataset** for inspiration
- **Streamlit Community** for the amazing framework
- **Medical Research Community** for cardiovascular risk factor knowledge
- **Open Source Contributors** for the underlying libraries

## 📞 Support

If you have questions or need help:
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/heart-disease-prediction/issues)
- 📧 **Contact**: your.email@example.com
- 📖 **Documentation**: Check the About page in the app

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**

*For educational and research purposes only. Not intended for medical diagnosis.*
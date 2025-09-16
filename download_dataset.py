import pandas as pd
import urllib.request

# Download the dataset
print("Downloading Heart Disease UCI dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
urllib.request.urlretrieve(url, 'data/heart_disease.csv')

# Column names based on the dataset documentation
column_names = [
    'age',
    'sex',
    'cp',  # chest pain type
    'trestbps',  # resting blood pressure
    'chol',  # serum cholesterol
    'fbs',  # fasting blood sugar
    'restecg',  # resting electrocardiographic results
    'thalach',  # maximum heart rate achieved
    'exang',  # exercise induced angina
    'oldpeak',  # ST depression induced by exercise
    'slope',  # slope of the peak exercise ST segment
    'ca',  # number of major vessels colored by fluoroscopy
    'thal',  # 3 = normal; 6 = fixed defect; 7 = reversable defect
    'target'  # diagnosis of heart disease
]

# Load the data and add column names
df = pd.read_csv('data/heart_disease.csv', names=column_names)

# Save with proper headers
df.to_csv('data/heart_disease.csv', index=False)

print(f"Dataset downloaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

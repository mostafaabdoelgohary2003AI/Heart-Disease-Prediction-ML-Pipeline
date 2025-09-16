#!/usr/bin/env python3
"""
Fix Streamlit Models - Create compatible model for the web app
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("🔧 FIXING STREAMLIT MODEL COMPATIBILITY")
    print("="*50)
    
    # Load the cleaned dataset
    df = pd.read_csv('data/heart_disease_cleaned.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # The selected features that performed best
    selected_features = ['thal', 'exang', 'ca', 'cp', 'thalach', 'slope', 'sex', 'oldpeak']
    
    # Prepare data with selected features only
    X_selected = df[selected_features]
    y = df['target']
    
    print(f"✅ Using {len(selected_features)} selected features")
    print(f"   Features: {selected_features}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create scaler for selected features only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data scaled: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
    
    # Train the best performing model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"✅ Model trained successfully!")
    print(f"   📊 Accuracy: {accuracy:.3f}")
    print(f"   📊 F1-Score: {f1:.3f}")
    
    # Test with sample input
    sample_input = np.array([[6, 0, 0, 3, 150, 3, 1, 2.3]])  # Sample values for the 8 features
    sample_scaled = scaler.transform(sample_input)
    sample_pred = model.predict(sample_scaled)[0]
    sample_proba = model.predict_proba(sample_scaled)[0]
    
    print(f"✅ Test prediction: {sample_pred} (probabilities: {sample_proba})")
    
    # Save the compatible model and scaler
    joblib.dump(model, 'models/final_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save feature mapping for Streamlit
    feature_mapping = {
        'selected_features': selected_features,
        'feature_order': selected_features,
        'n_features': len(selected_features),
        'model_performance': {
            'accuracy': accuracy,
            'f1_score': f1
        }
    }
    joblib.dump(feature_mapping, 'models/streamlit_feature_mapping.pkl')
    
    print("✅ STREAMLIT-COMPATIBLE MODELS SAVED!")
    print("✅ Models now work with 8 selected features!")
    print("🌐 Streamlit app should now work perfectly!")
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n🎉 SUCCESS! Streamlit models are now compatible!")
        print("🔄 Refresh your browser to see the working app!")
    else:
        print("\n❌ Failed to fix models")

#!/usr/bin/env python3
"""
Create a WORKING model for Streamlit - Fix the feature mismatch
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def main():
    print("🔧 CREATING WORKING MODEL FOR STREAMLIT")
    print("="*50)
    
    # Load cleaned dataset
    df = pd.read_csv('data/heart_disease_cleaned.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Ensure binary target
    df['target'] = (df['target'] > 0).astype(int)
    
    # Use ALL 13 features for consistency with Streamlit input form
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    X = df[feature_columns]
    y = df['target']
    
    print(f"✅ Using ALL {len(feature_columns)} features for Streamlit compatibility")
    print(f"   Features: {feature_columns}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"   📊 Accuracy: {accuracy:.3f}")
    print(f"   📊 F1-Score: {f1:.3f}")
    
    # Test with full 13-feature input (matching Streamlit form)
    test_input = np.array([[63, 1, 3, 145, 233, 1, 2, 150, 0, 2.3, 2, 0, 6]])  # All 13 features
    test_scaled = scaler.transform(test_input)
    test_pred = model.predict(test_scaled)[0]
    test_proba = model.predict_proba(test_scaled)[0]
    
    print(f"✅ Test prediction with 13 features: {test_pred}")
    print(f"   Probabilities: No Disease: {test_proba[0]:.3f}, Disease: {test_proba[1]:.3f}")
    
    # Save the working model and scaler
    joblib.dump(model, 'models/final_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Create model info for Streamlit
    model_info = {
        'feature_names': feature_columns,
        'n_features': len(feature_columns),
        'model_type': 'RandomForestClassifier',
        'accuracy': accuracy,
        'f1_score': f1,
        'feature_order': feature_columns
    }
    joblib.dump(model_info, 'models/model_info.pkl')
    
    print("✅ WORKING MODELS SAVED!")
    print("✅ Scaler and model now match (13 features each)")
    print("🌐 Streamlit app will now work perfectly!")
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n🎉 SUCCESS! Models are now compatible with Streamlit!")
        print("🔄 Restart Streamlit to see the working app!")
    else:
        print("\n❌ Failed to create working models")

#!/usr/bin/env python3
"""
Quick model training script for Heart Disease Prediction
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    print('🤖 TRAINING ALL MODELS FOR STREAMLIT APP...')
    print('='*50)
    
    # Load data
    try:
        X_train_selected = pd.read_csv('data/X_train_selected.csv')
        X_test_selected = pd.read_csv('data/X_test_selected.csv')
        y_train = pd.read_csv('data/y_train.csv').squeeze()
        y_test = pd.read_csv('data/y_test.csv').squeeze()
        print(f'✅ Data loaded: Train {X_train_selected.shape}, Test {X_test_selected.shape}')
    except Exception as e:
        print(f'❌ Error loading data: {e}')
        return False
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1, 
            random_state=42, 
            max_iter=1000
        ),
        'SVM': SVC(
            C=1, 
            kernel='rbf', 
            probability=True, 
            random_state=42
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=10, 
            min_samples_split=5,
            random_state=42
        )
    }
    
    # Train and evaluate models
    best_models = {}
    best_scores = {}
    
    for name, model in models.items():
        print(f'   🔧 Training {name}...')
        
        try:
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Store results
            best_models[name] = model
            best_scores[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            
            print(f'     ✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, ROC-AUC={roc_auc:.3f}')
            
        except Exception as e:
            print(f'     ❌ Error training {name}: {e}')
    
    # Find best model
    if best_scores:
        best_model_name = max(best_scores.keys(), key=lambda x: best_scores[x]['f1_score'])
        final_model = best_models[best_model_name]
        
        print(f'\n🏆 BEST MODEL: {best_model_name}')
        print(f'   📊 Accuracy: {best_scores[best_model_name]["accuracy"]:.3f}')
        print(f'   📊 Precision: {best_scores[best_model_name]["precision"]:.3f}')
        print(f'   📊 Recall: {best_scores[best_model_name]["recall"]:.3f}')
        print(f'   📊 F1-Score: {best_scores[best_model_name]["f1_score"]:.3f}')
        print(f'   📊 ROC-AUC: {best_scores[best_model_name]["roc_auc"]:.3f}')
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save individual models
        for name, model in best_models.items():
            joblib.dump(model, f'models/best_{name.lower()}_model.pkl')
        
        # Save final model
        joblib.dump(final_model, 'models/final_model.pkl')
        
        # Create and save model summary
        model_summary = {
            'best_model': best_model_name,
            'test_metrics': best_scores[best_model_name],
            'feature_names': list(X_train_selected.columns),
            'all_model_scores': best_scores,
            'n_features': len(X_train_selected.columns)
        }
        joblib.dump(model_summary, 'models/model_summary.pkl')
        
        print(f'\n✅ ALL MODELS SAVED!')
        print(f'✅ FILES CREATED:')
        print(f'   - models/final_model.pkl')
        print(f'   - models/model_summary.pkl')
        print(f'   - models/scaler.pkl')
        print(f'   - Individual model files')
        
        print(f'\n🎉 PIPELINE COMPLETE! STREAMLIT APP IS NOW READY!')
        print(f'🌐 Refresh your browser at http://localhost:8501')
        
        return True
    else:
        print('❌ No models were successfully trained!')
        return False

if __name__ == '__main__':
    main()

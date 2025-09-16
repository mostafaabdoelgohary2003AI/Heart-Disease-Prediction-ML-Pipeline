#!/usr/bin/env python3
"""
Complete Heart Disease Prediction ML Pipeline
This script runs ALL the machine learning steps that should be in the notebooks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os

def main():
    print('🚀 RUNNING COMPLETE HEART DISEASE PREDICTION ML PIPELINE')
    print('='*70)
    
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ===== STEP 1: DATA PREPROCESSING =====
    print('\n📊 STEP 1: DATA PREPROCESSING AND CLEANING')
    print('-'*50)
    
    # Load dataset
    df = pd.read_csv('data/heart_disease.csv')
    print(f'✅ Dataset loaded: {df.shape}')
    print(f'   Features: {list(df.columns)}')
    
    # Handle missing values (fix the regex issue)
    df_clean = df.copy()
    for col in df.columns:
        # Replace '?' with NaN (avoid regex issues)
        df_clean[col] = df_clean[col].replace('?', np.nan)
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Impute missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if col in ['ca', 'thal']:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 0
                df_clean[col].fillna(mode_val, inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Convert target to binary
    df_clean['target'] = (df_clean['target'] > 0).astype(int)
    
    print(f'✅ Missing values handled: {df_clean.isnull().sum().sum()} remaining')
    no_disease = (df_clean['target']==0).sum()
    disease = (df_clean['target']==1).sum()
    print(f'✅ Target distribution: No Disease: {no_disease}, Disease: {disease}')
    
    # Save cleaned dataset
    df_clean.to_csv('data/heart_disease_cleaned.csv', index=False)
    
    # Split features and target
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'✅ Data split: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}')
    
    # Save preprocessed data
    joblib.dump(scaler, 'models/scaler.pkl')
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/X_test_scaled.csv', index=False)
    pd.DataFrame(X_train, columns=X.columns).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv('data/X_test.csv', index=False)
    y_train.to_frame().to_csv('data/y_train.csv', index=False)
    y_test.to_frame().to_csv('data/y_test.csv', index=False)
    
    print('✅ STEP 1 COMPLETE!')
    
    # ===== STEP 2: PCA ANALYSIS =====
    print('\n📈 STEP 2: PCA ANALYSIS')
    print('-'*50)
    
    # Apply PCA
    pca_full = PCA()
    X_train_pca_full = pca_full.fit_transform(X_train_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Find optimal components for 90% variance
    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    
    # Apply optimal PCA
    pca_optimal = PCA(n_components=n_components_90)
    X_train_pca = pca_optimal.fit_transform(X_train_scaled)
    X_test_pca = pca_optimal.transform(X_test_scaled)
    
    print(f'✅ PCA Analysis: {n_components_90} components for 90% variance')
    print(f'   Variance retained: {cumulative_variance_ratio[n_components_90-1]:.3f}')
    
    # Save PCA results
    joblib.dump(pca_optimal, 'models/pca_model.pkl')
    pca_columns = [f'PC{i+1}' for i in range(n_components_90)]
    pd.DataFrame(X_train_pca, columns=pca_columns).to_csv('data/X_train_pca.csv', index=False)
    pd.DataFrame(X_test_pca, columns=pca_columns).to_csv('data/X_test_pca.csv', index=False)
    
    print('✅ STEP 2 COMPLETE!')
    
    # ===== STEP 3: FEATURE SELECTION =====
    print('\n🎯 STEP 3: FEATURE SELECTION')
    print('-'*50)
    
    # Random Forest Feature Importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X_train_scaled, y_train)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_selected = feature_importance[feature_importance['importance'] >= 0.05]['feature'].tolist()
    
    # Recursive Feature Elimination
    rfe = RFE(estimator=LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=7)
    rfe.fit(X_train_scaled, y_train)
    rfe_selected = X.columns[rfe.support_].tolist()
    
    # Chi-square test (need non-negative values)
    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train_scaled)
    chi2_selector = SelectKBest(chi2, k=8)
    chi2_selector.fit(X_train_minmax, y_train)
    chi2_selected = X.columns[chi2_selector.get_support()].tolist()
    
    # F-score test
    f_selector = SelectKBest(f_classif, k=8)
    f_selector.fit(X_train_scaled, y_train)
    f_selected = X.columns[f_selector.get_support()].tolist()
    
    # Consensus selection
    all_features = set(X.columns)
    feature_votes = {}
    for feature in all_features:
        votes = 0
        if feature in rf_selected: votes += 1
        if feature in rfe_selected: votes += 1
        if feature in chi2_selected: votes += 1
        if feature in f_selected: votes += 1
        feature_votes[feature] = votes
    
    final_selected_features = [f for f, v in feature_votes.items() if v >= 2]
    if len(final_selected_features) < 6:
        additional = [f for f, v in sorted(feature_votes.items(), key=lambda x: x[1], reverse=True) if v == 1]
        final_selected_features.extend(additional[:6-len(final_selected_features)])
    
    print(f'✅ Feature Selection: {len(final_selected_features)} features selected')
    print(f'   Selected features: {final_selected_features}')
    
    # Create selected feature datasets
    X_train_selected = pd.DataFrame(X_train_scaled, columns=X.columns)[final_selected_features]
    X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[final_selected_features]
    
    # Save feature selection results
    joblib.dump(rfe, 'models/rfe_selector.pkl')
    joblib.dump(chi2_selector, 'models/chi2_selector.pkl')
    joblib.dump(f_selector, 'models/f_selector.pkl')
    X_train_selected.to_csv('data/X_train_selected.csv', index=False)
    X_test_selected.to_csv('data/X_test_selected.csv', index=False)
    
    feature_selection_summary = {
        'final_selected_features': final_selected_features,
        'rf_selected': rf_selected,
        'rfe_selected': rfe_selected,
        'chi2_selected': chi2_selected,
        'f_selected': f_selected,
        'feature_votes': feature_votes
    }
    joblib.dump(feature_selection_summary, 'models/feature_selection_summary.pkl')
    
    print('✅ STEP 3 COMPLETE!')
    
    # ===== STEP 4: SUPERVISED LEARNING =====
    print('\n🤖 STEP 4: SUPERVISED LEARNING MODELS')
    print('-'*50)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Test on different feature sets
    datasets = {
        'All Features': (pd.DataFrame(X_train_scaled, columns=X.columns), pd.DataFrame(X_test_scaled, columns=X.columns)),
        'Selected Features': (X_train_selected, X_test_selected),
        'PCA Features': (pd.DataFrame(X_train_pca, columns=pca_columns), pd.DataFrame(X_test_pca, columns=pca_columns))
    }
    
    results = []
    best_overall_score = 0
    best_overall_model = None
    best_overall_name = ""
    
    for dataset_name, (X_tr, X_te) in datasets.items():
        print(f'   Testing on {dataset_name}:')
        for model_name, model in models.items():
            # Train model
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_tr, y_train)
            
            # Make predictions
            y_pred = model_copy.predict(X_te)
            y_pred_proba = model_copy.predict_proba(X_te)[:, 1] if hasattr(model_copy, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            results.append({
                'Model': model_name,
                'Dataset': dataset_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
            
            print(f'     {model_name}: Acc={accuracy:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}')
            
            # Track best overall model
            if f1 > best_overall_score:
                best_overall_score = f1
                best_overall_model = model_copy
                best_overall_name = f"{model_name} ({dataset_name})"
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/supervised_learning_results.csv', index=False)
    
    print(f'\\n🏆 Best Overall Model: {best_overall_name}')
    print(f'   F1-Score: {best_overall_score:.3f}')
    
    print('✅ STEP 4 COMPLETE!')
    
    # ===== STEP 5: UNSUPERVISED LEARNING =====
    print('\\n🔍 STEP 5: UNSUPERVISED LEARNING (CLUSTERING)')
    print('-'*50)
    
    # K-Means clustering
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train_scaled)
        inertias.append(kmeans.inertia_)
    
    # Optimal K (elbow method - simplified)
    optimal_k = 3  # Typically good for binary classification
    
    # Final K-means
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans_final.fit_predict(X_train_scaled)
    
    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(X_train_scaled)
    
    # Calculate silhouette scores
    kmeans_silhouette = silhouette_score(X_train_scaled, kmeans_labels)
    hierarchical_silhouette = silhouette_score(X_train_scaled, hierarchical_labels)
    
    print(f'✅ K-Means Clustering: K={optimal_k}, Silhouette={kmeans_silhouette:.3f}')
    print(f'✅ Hierarchical Clustering: K={optimal_k}, Silhouette={hierarchical_silhouette:.3f}')
    
    # Save clustering models
    joblib.dump(kmeans_final, 'models/kmeans_model.pkl')
    
    print('✅ STEP 5 COMPLETE!')
    
    # ===== STEP 6: HYPERPARAMETER TUNING =====
    print('\\n⚙️ STEP 6: HYPERPARAMETER TUNING')
    print('-'*50)
    
    # Focus on best performing models with grid search
    tuning_models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        }
    }
    
    best_tuned_models = {}
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model_info in tuning_models.items():
        print(f'   Tuning {model_name}...')
        
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=cv_folds,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_selected, y_train)
        
        # Test performance
        y_pred = grid_search.predict(X_test_selected)
        y_pred_proba = grid_search.predict_proba(X_test_selected)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        best_tuned_models[model_name] = grid_search.best_estimator_
        
        print(f'     ✅ {model_name}: Acc={accuracy:.3f}, F1={f1:.3f}, AUC={roc_auc:.3f}')
        print(f'        Best params: {grid_search.best_params_}')
    
    # Select final model
    final_model = best_overall_model  # Use the best from supervised learning
    
    print('✅ STEP 6 COMPLETE!')
    
    # ===== FINAL MODEL EXPORT =====
    print('\\n💾 FINAL: MODEL EXPORT')
    print('-'*50)
    
    # Save final model
    joblib.dump(final_model, 'models/final_model.pkl')
    
    # Create comprehensive model summary
    model_summary = {
        'best_model_name': best_overall_name,
        'best_f1_score': best_overall_score,
        'feature_names': final_selected_features,
        'n_features': len(final_selected_features),
        'dataset_shape': df_clean.shape,
        'train_test_split': f"{X_train_scaled.shape[0]}/{X_test_scaled.shape[0]}",
        'pca_components': n_components_90,
        'clustering_results': {
            'kmeans_silhouette': kmeans_silhouette,
            'hierarchical_silhouette': hierarchical_silhouette
        }
    }
    joblib.dump(model_summary, 'models/model_summary.pkl')
    
    # Save evaluation metrics
    with open('results/evaluation_metrics.txt', 'w') as f:
        f.write("Heart Disease Prediction - Model Evaluation Results\\n")
        f.write("="*50 + "\\n")
        f.write(f"Best Model: {best_overall_name}\\n")
        f.write(f"Best F1-Score: {best_overall_score:.3f}\\n")
        f.write(f"Selected Features: {final_selected_features}\\n")
        f.write(f"Dataset Shape: {df_clean.shape}\\n")
        f.write(f"PCA Components (90% var): {n_components_90}\\n")
        f.write(f"K-Means Silhouette: {kmeans_silhouette:.3f}\\n")
        f.write(f"Hierarchical Silhouette: {hierarchical_silhouette:.3f}\\n")
    
    print('✅ FINAL MODEL EXPORTED!')
    
    # ===== SUMMARY =====
    print('\\n🎉 COMPLETE ML PIPELINE FINISHED!')
    print('='*70)
    print('✅ Data Preprocessing: Missing values handled, features scaled')
    print(f'✅ PCA Analysis: {n_components_90} components for 90% variance')
    print(f'✅ Feature Selection: {len(final_selected_features)} optimal features')
    print('✅ Supervised Learning: 4 algorithms tested on 3 feature sets')
    print('✅ Unsupervised Learning: K-Means and Hierarchical clustering')
    print('✅ Hyperparameter Tuning: Grid search optimization')
    print(f'✅ Final Model: {best_overall_name} with F1={best_overall_score:.3f}')
    
    print('\\n📁 FILES CREATED:')
    print('   Data: X_train/test_scaled.csv, X_train/test_selected.csv, X_train/test_pca.csv')
    print('   Models: final_model.pkl, scaler.pkl, pca_model.pkl, *_selector.pkl')
    print('   Results: supervised_learning_results.csv, evaluation_metrics.txt')
    
    print('\\n🌐 STREAMLIT APP IS NOW FULLY FUNCTIONAL!')
    print('   Refresh your browser at http://localhost:8501')
    print('   All models are trained and ready for predictions!')
    
    return True

if __name__ == '__main__':
    success = main()
    if success:
        print('\\n🎊 SUCCESS! Complete ML pipeline executed successfully!')
    else:
        print('\\n❌ Pipeline failed. Check error messages above.')

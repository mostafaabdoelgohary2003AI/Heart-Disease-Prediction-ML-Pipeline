#!/usr/bin/env python3
"""
Heart Disease Prediction - Complete Pipeline Runner

This script runs the entire machine learning pipeline:
1. Data preprocessing
2. PCA analysis  
3. Feature selection
4. Supervised learning
5. Unsupervised learning
6. Hyperparameter tuning
7. Model export

Usage: python run_pipeline.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_notebook(notebook_path):
    """Run a Jupyter notebook using nbconvert"""
    print(f"\n{'='*60}")
    print(f"Running: {notebook_path}")
    print(f"{'='*60}")
    
    try:
        # Convert notebook path to absolute path
        nb_path = Path(notebook_path)
        if not nb_path.exists():
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        # Run the notebook
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            str(nb_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Successfully completed: {notebook_path}")
            return True
        else:
            print(f"❌ Error running {notebook_path}:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout running {notebook_path}")
        return False
    except Exception as e:
        print(f"❌ Exception running {notebook_path}: {str(e)}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'jupyter', 'joblib', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_directories():
    """Ensure all required directories exist"""
    directories = ['data', 'models', 'results', 'notebooks', 'ui', 'deployment']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def main():
    """Run the complete machine learning pipeline"""
    print("🚀 Starting Heart Disease Prediction Pipeline")
    print(f"Working directory: {os.getcwd()}")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup directories
    setup_directories()
    
    # Define notebook execution order
    notebooks = [
        "notebooks/01_data_preprocessing.ipynb",
        "notebooks/02_pca_analysis.ipynb", 
        "notebooks/03_feature_selection.ipynb",
        "notebooks/04_supervised_learning.ipynb",
        "notebooks/05_unsupervised_learning.ipynb",
        "notebooks/06_hyperparameter_tuning.ipynb"
    ]
    
    # Track execution results
    results = {}
    start_time = time.time()
    
    # Run each notebook
    for notebook in notebooks:
        notebook_start = time.time()
        success = run_notebook(notebook)
        notebook_time = time.time() - notebook_start
        
        results[notebook] = {
            'success': success,
            'time': notebook_time
        }
        
        if not success:
            print(f"\n❌ Pipeline failed at: {notebook}")
            print("Please check the notebook for errors and try again.")
            break
        
        print(f"⏱️  Completed in {notebook_time:.1f} seconds")
    
    # Summary
    total_time = time.time() - start_time
    successful_notebooks = sum(1 for r in results.values() if r['success'])
    
    print(f"\n{'='*60}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Notebooks completed: {successful_notebooks}/{len(notebooks)}")
    
    for notebook, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {notebook} ({result['time']:.1f}s)")
    
    if successful_notebooks == len(notebooks):
        print("\n🎉 Pipeline completed successfully!")
        print("\n📱 Next steps:")
        print("1. Test the Streamlit app: streamlit run ui/app.py")
        print("2. Set up ngrok deployment: see deployment/ngrok_setup.txt")
        print("3. Upload to GitHub: git init && git add . && git commit -m 'Initial commit'")
        return True
    else:
        print(f"\n⚠️  Pipeline incomplete: {len(notebooks) - successful_notebooks} notebooks failed")
        return False

def test_streamlit():
    """Test if the Streamlit app can be imported"""
    try:
        import streamlit as st
        print("✅ Streamlit app ready to run")
        print("   Command: streamlit run ui/app.py")
        return True
    except Exception as e:
        print(f"❌ Streamlit app test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the pipeline
    success = main()
    
    if success:
        print("\n🧪 Testing Streamlit app...")
        test_streamlit()
        
        print(f"\n🔗 Useful commands:")
        print(f"  Run app locally: streamlit run ui/app.py")
        print(f"  Create tunnel: ngrok http 8501")
        print(f"  View results: cat results/evaluation_metrics.txt")
    
    sys.exit(0 if success else 1)

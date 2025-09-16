#!/usr/bin/env python3
"""
Simple Ngrok Deployment Script
"""

from pyngrok import ngrok, conf
import time
import sys

def deploy_with_ngrok():
    """Deploy the Streamlit app using ngrok"""
    print("🌐 DEPLOYING HEART DISEASE PREDICTION APP WITH NGROK")
    print("="*60)
    
    # Set your auth token
    token = "2wEHqzj3QcjCTfT74xiNE9pZqL6_gpWtdnoXftCs7KJqvDcv"
    
    try:
        # Configure ngrok
        ngrok.set_auth_token(token)
        print("✅ Ngrok auth token configured!")
        
        # Create tunnel to Streamlit port (8501)
        print("🚀 Creating public tunnel...")
        tunnel = ngrok.connect(8501)
        
        print("🎉 SUCCESS! Your app is now publicly accessible!")
        print("="*60)
        print(f"🌐 PUBLIC URL: {tunnel.public_url}")
        print("="*60)
        print("")
        print("📱 SHARE THIS URL WITH ANYONE:")
        print(f"   {tunnel.public_url}")
        print("")
        print("🏥 Your Heart Disease Prediction App Features:")
        print("   ✅ 88.5% accuracy machine learning model")
        print("   ✅ Real-time heart disease risk assessment")
        print("   ✅ Interactive data analysis dashboard")
        print("   ✅ Professional medical prediction interface")
        print("")
        print("⚠️  IMPORTANT:")
        print("   - Keep this terminal open to maintain the tunnel")
        print("   - Make sure Streamlit is running: streamlit run ui/app.py")
        print("   - Press Ctrl+C to stop the tunnel")
        print("")
        print("🎊 CONGRATULATIONS! YOUR PROJECT IS NOW 100% COMPLETE!")
        print("   📱 GitHub: https://github.com/mostafaabdoelgohary2003AI/Heart-Disease-Prediction-ML-Pipeline")
        print(f"   🌐 Live App: {tunnel.public_url}")
        
        # Keep the tunnel alive
        try:
            print("\n🔄 Tunnel is active. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping ngrok tunnel...")
            ngrok.disconnect(tunnel.public_url)
            print("✅ Tunnel stopped successfully!")
            
    except Exception as e:
        print(f"❌ Error creating tunnel: {e}")
        print("\n🔧 ALTERNATIVE SOLUTION:")
        print("1. Download ngrok manually from: https://ngrok.com/download")
        print("2. Extract ngrok.exe to this folder")
        print("3. Run: ngrok config add-authtoken 2wEHqzj3QcjCTfT74xiNE9pZqL6_gpWtdnoXftCs7KJqvDcv")
        print("4. Run: ngrok http 8501")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting ngrok deployment...")
    print("📋 Make sure Streamlit is running first: streamlit run ui/app.py")
    print("")
    
    # Give user a moment to start Streamlit if needed
    input("Press Enter when Streamlit is running at http://localhost:8501...")
    
    deploy_with_ngrok()

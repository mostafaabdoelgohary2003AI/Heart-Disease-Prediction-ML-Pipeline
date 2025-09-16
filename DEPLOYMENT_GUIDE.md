# 🚀 Complete Deployment Guide

## 🌐 **NGROK DEPLOYMENT SETUP**

### **Step 1: Install Ngrok**
1. **Download Ngrok**:
   - Go to: https://ngrok.com/download
   - Download the Windows version
   - Extract `ngrok.exe` to this project folder

2. **Get Auth Token**:
   - Sign up at: https://ngrok.com (free account)
   - Go to: https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy your authtoken

3. **Configure Ngrok**:
   ```bash
   ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
   ```

### **Step 2: Deploy the App**
```bash
# Start Streamlit app (if not already running)
streamlit run ui/app.py

# In a new terminal, create public tunnel
ngrok http 8501
```

### **Step 3: Access Your Public App**
- Ngrok will provide a public URL like: `https://abc123.ngrok.io`
- Share this URL with anyone to access your heart disease prediction app!

---

## 📱 **ALTERNATIVE DEPLOYMENT OPTIONS**

### **Option 1: Streamlit Cloud (Free & Easy)**
1. Upload project to GitHub (see below)
2. Go to: https://share.streamlit.io/
3. Connect GitHub repo
4. Deploy automatically!

### **Option 2: Heroku (Free Tier)**
```bash
# Create Procfile
echo "web: streamlit run ui/app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-heart-disease-app
git push heroku main
```

---

## 🎯 **QUICK DEPLOYMENT CHECKLIST**

- [ ] Install ngrok
- [ ] Get auth token from ngrok.com
- [ ] Configure: `ngrok config add-authtoken YOUR_TOKEN`
- [ ] Run: `streamlit run ui/app.py`
- [ ] Run: `ngrok http 8501`
- [ ] Share the public URL!

Your app will be live and accessible worldwide! 🌍

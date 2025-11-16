# Deployment Guide: GitHub & Streamlit Cloud

This guide will walk you through uploading your project to GitHub and deploying it to Streamlit Cloud for public access.

---

## Part 1: Upload to GitHub

### Step 1: Create a GitHub Account (if you don't have one)

1. Go to [github.com](https://github.com)
2. Click "Sign up"
3. Follow the registration process

### Step 2: Create a New Repository

1. Log in to GitHub
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `heart-disease-prediction` (or your preferred name)
   - **Description**: "Machine Learning model for heart disease prediction using Logistic Regression and Random Forest"
   - **Visibility**: Choose **Public** (required for free Streamlit Cloud)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### Step 3: Initialize Git in Your Project

Open your terminal/command prompt in your project directory and run:

```bash
# Navigate to your project directory
cd C:\Users\hp\OneDrive\Desktop\projects\heart_disease_model

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Heart Disease Prediction Model"

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git

# Rename main branch (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: You'll be prompted for your GitHub username and password. For password, use a **Personal Access Token** (see below).

### Step 4: Create a Personal Access Token (for authentication)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **"Generate new token (classic)"**
3. Give it a name: "Streamlit Deployment"
4. Select scopes: Check **"repo"** (full control of private repositories)
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't see it again!)
7. Use this token as your password when pushing to GitHub

---

## Part 2: Deploy to Streamlit Cloud

### Step 1: Sign up for Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign up"**
3. Sign in with your **GitHub account** (this links your GitHub to Streamlit)

### Step 2: Deploy Your App

1. After signing in, click **"New app"**
2. Fill in the deployment form:
   - **Repository**: Select `YOUR_USERNAME/heart-disease-prediction`
   - **Branch**: `main` (or `master`)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g., `heart-disease-prediction`)
3. Click **"Deploy"**

### Step 3: Wait for Deployment

- Streamlit will automatically:
  - Install dependencies from `requirements.txt`
  - Run your `app.py`
  - Deploy your app
- This usually takes 2-5 minutes
- You'll see a live URL like: `https://heart-disease-prediction.streamlit.app`

### Step 4: Access Your Public App

Once deployed, your app will be:
- ✅ Publicly accessible via the URL
- ✅ Automatically updated when you push changes to GitHub
- ✅ Free to use (with Streamlit Cloud limitations)

---

## Part 3: Important Configuration Files

### Ensure Your `requirements.txt` is Complete

Make sure your `requirements.txt` includes all dependencies:

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
streamlit>=1.28.0
seaborn>=0.12.0
joblib>=1.2.0
```

### Create `packages.txt` (Optional - for system packages)

If you need system packages, create `packages.txt` in the root:

```txt
# Usually not needed for this project
```

### Create `.streamlit/config.toml` (Optional - for app configuration)

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
```

---

## Part 4: Troubleshooting

### Common Issues

#### 1. **ModuleNotFoundError**
- **Solution**: Ensure all dependencies are in `requirements.txt`
- Check that imports use correct paths (e.g., `from src import ...`)

#### 2. **FileNotFoundError: dataset.csv**
- **Solution**: Make sure `data/raw/dataset.csv` is committed to GitHub
- Check that paths in `config.py` are relative, not absolute

#### 3. **App Crashes on Startup**
- **Solution**: Check Streamlit Cloud logs (click "Manage app" → "Logs")
- Ensure all file paths are relative to the project root
- Verify that `config.py` paths work correctly

#### 4. **Import Errors**
- **Solution**: Ensure `src/__init__.py` exists
- Check that imports in `app.py` use `from src import ...`

#### 5. **Deployment Takes Too Long**
- **Solution**: This is normal for first deployment (2-5 minutes)
- Subsequent updates are faster (usually < 1 minute)

---

## Part 5: Updating Your App

After making changes to your code:

```bash
# Make your changes to files

# Stage changes
git add .

# Commit changes
git commit -m "Update: [describe your changes]"

# Push to GitHub
git push origin main
```

Streamlit Cloud will **automatically detect** the changes and redeploy your app (usually within 1-2 minutes).

---

## Part 6: Best Practices

### 1. **Keep Your Repository Clean**
- Don't commit large files (>100MB)
- Use `.gitignore` to exclude unnecessary files
- Don't commit sensitive data (API keys, passwords)

### 2. **Test Locally First**
```bash
streamlit run app.py
```
Always test your app locally before pushing to GitHub.

### 3. **Monitor Your App**
- Check Streamlit Cloud dashboard regularly
- Monitor app usage and performance
- Review logs if issues occur

### 4. **Documentation**
- Keep `README.md` updated
- Add comments to your code
- Document any special requirements

---

## Quick Reference Commands

```bash
# Initialize and push to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-prediction.git
git push -u origin main

# Update your app
git add .
git commit -m "Update description"
git push origin main

# Check status
git status
git log
```

---

## Summary Checklist

Before deploying, ensure:

- [ ] All code is working locally
- [ ] `requirements.txt` is complete and up-to-date
- [ ] `data/raw/dataset.csv` is in the repository
- [ ] `.gitignore` is properly configured
- [ ] `README.md` is informative
- [ ] All imports use correct paths (`from src import ...`)
- [ ] GitHub repository is created and public
- [ ] Code is pushed to GitHub
- [ ] Streamlit Cloud account is connected to GitHub
- [ ] App is deployed on Streamlit Cloud

---

## Need Help?

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Streamlit Community**: https://discuss.streamlit.io/
- **GitHub Help**: https://docs.github.com/

---

**Congratulations!** 🎉 Once deployed, your app will be publicly accessible and automatically update with each GitHub push!


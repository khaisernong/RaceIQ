# Deployment Instructions for RaceIQ

## Streamlit Cloud Deployment

### Prerequisites
1. GitHub account with RaceIQ repository
2. Streamlit Cloud account (sign up at https://streamlit.io/cloud)

### Steps to Deploy

1. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Repository: `khaisernong/RaceIQ`
   - Branch: `main`
   - Main file path: `src/ui/dashboard.py`
   - App URL: Choose your custom URL (e.g., `raceiq-analytics`)

3. **Configure Dataset**
   - **Important**: The Dataset folder is NOT included in the repository (too large)
   - You have two options:

   **Option A: Upload Dataset to GitHub**
   - Create a separate repository for the dataset
   - Update the dataset path in `dashboard.py` to point to the new location

   **Option B: Use Sample Data** (Recommended for demo)
   - The app will use the provided sample data from Barber and COTA tracks
   - For full functionality, you'll need to provide the complete dataset

4. **Advanced Settings** (Optional)
   - Python version: 3.10+
   - Secrets: Add any API keys if needed (none required currently)

5. **Deploy**
   - Click "Deploy!"
   - Wait 2-3 minutes for deployment to complete

### Post-Deployment

Your app will be live at: `https://[your-app-name].streamlit.app`

Example: `https://raceiq-analytics.streamlit.app`

### Local Testing Before Deployment

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run Streamlit locally
streamlit run src/ui/dashboard.py
```

### Troubleshooting

**Issue**: "Dataset not found"
- **Solution**: Upload dataset to a cloud storage (GitHub LFS, AWS S3, etc.) and update the path

**Issue**: "Module not found"
- **Solution**: Ensure all dependencies are in `requirements.txt`

**Issue**: "Memory limit exceeded"
- **Solution**: Streamlit Cloud has resource limits. Consider:
  - Implementing data caching with `@st.cache_data`
  - Loading only necessary data
  - Using data sampling for large datasets

### Alternative Deployment Options

1. **Heroku**
   - Add `Procfile`: `web: streamlit run src/ui/dashboard.py --server.port $PORT`
   - Add `setup.sh` for Streamlit configuration
   - Deploy via Heroku CLI

2. **AWS EC2**
   - Launch EC2 instance
   - Install Python and dependencies
   - Run Streamlit with SSL (nginx + certbot)
   - Open port 8501 in security group

3. **Docker**
   - Create Dockerfile
   - Build image: `docker build -t raceiq .`
   - Run container: `docker run -p 8501:8501 raceiq`

### Configuration Files

All configuration is set in `.streamlit/config.toml`:
- Theme colors (Toyota Red)
- Server settings
- Browser preferences

---

**Ready to deploy!** 🚀

For questions or issues, contact: khaisernong@github.com

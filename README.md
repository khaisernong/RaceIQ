# 🏁 RaceIQ - Real-Time Racing Intelligence & Strategy Platform

**Toyota Gazoo Racing Hackathon 2025**  
**Category**: Real-Time Analytics

## 🎯 Project Overview

RaceIQ is an intelligent real-time analytics system that provides race engineers and drivers with actionable insights during live race conditions. The platform enables data-driven decisions for optimal pit stop timing, tire degradation monitoring, performance gap analysis, and weather-responsive strategy adjustments.

## ✨ Key Features

- **Pit Stop Optimizer**: Calculate optimal pit windows based on tire degradation, fuel consumption, and track position
- **Tire Strategy Intelligence**: Real-time tire degradation monitoring and lap time fade prediction
- **Race Situation Awareness**: Live position tracking with gap analysis and sector-by-sector performance comparison
- **Performance Anomaly Detection**: Driver consistency monitoring and vehicle performance degradation alerts
- **Weather-Responsive Strategy**: Track temperature impact analysis and grip level predictions

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Navigate to project directory
cd raceiq

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Link Dataset

```powershell
# Create symlink to Dataset folder
New-Item -ItemType SymbolicLink -Path "data\raw" -Target "..\Dataset"
```

### 3. Run Dashboard

```powershell
streamlit run src/ui/dashboard.py
```

## 📊 Dataset

The project uses GR Cup Series racing data from multiple tracks:
- Barber Motorsports Park
- Circuit of the Americas (COTA)
- Indianapolis Motor Speedway
- Road America
- Sebring International Raceway
- Sonoma Raceway
- Virginia International Raceway (VIR)

Each track includes:
- Race results and standings
- Lap-by-lap sector analysis
- Weather conditions
- Lap timing data
- High-frequency telemetry

## 🏗️ Project Structure

```
raceiq/
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Predictive models (tire degradation, lap time)
│   ├── analytics/         # Analytics modules (pit optimizer, gap analyzer)
│   ├── ui/                # Streamlit dashboard
│   └── utils/             # Utility functions
├── data/
│   ├── raw/               # Raw CSV files (symlink to Dataset)
│   └── processed/         # Processed data
├── models/                # Saved model files
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🛠️ Technology Stack

- **Python 3.10+**
- **Data Processing**: pandas, numpy
- **Visualization**: Streamlit, Plotly, matplotlib
- **Machine Learning**: scikit-learn, XGBoost
- **Optimization**: scipy

## 📈 Analytics Approach

### Tire Degradation Model
- Polynomial regression with environmental factors
- Predicts lap time increase as tires age
- Factors: tire age, track temperature, air temperature

### Lap Time Prediction
- Exponential Weighted Moving Average (EWMA)
- Trend analysis for pace monitoring
- Confidence intervals for predictions

### Pit Stop Optimization
- Multi-objective optimization
- Considers tire life, track position, gaps, fuel requirements
- Dynamic programming for optimal timing

### Anomaly Detection
- Statistical process control
- Z-score based outlier detection
- Sector-specific performance monitoring

## 🎯 Success Metrics

- **Prediction Accuracy**: RMSE < 0.5 seconds for lap time predictions
- **Response Time**: < 5 seconds from data to insight
- **System Latency**: Dashboard refresh rate 1-2 seconds
- **Model Performance**: R² > 0.85 for degradation models

## 📝 Development Status

- [x] Project structure setup
- [x] Requirements defined
- [ ] Data loader implementation
- [ ] Data preprocessing pipeline
- [ ] Feature engineering
- [ ] Tire degradation model
- [ ] Streamlit dashboard
- [ ] Testing and validation

## 👥 Team

University of Tsukuba Malaysia  
Toyota Gazoo Racing Hackathon 2025

## 📄 License

This project is developed for the Toyota Gazoo Racing Hackathon 2025.

## 🔗 Links

- [Hackathon Page](https://hackthetrack.devpost.com/)
- [Documentation](./docs/)
- [Pre-Prompt Documentation](../preprompt-01-overview.md)

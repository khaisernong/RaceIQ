# 🏁 RaceIQ - Quick Start Guide

## 🚀 Installation & Setup

### Step 1: Navigate to Project Directory

```powershell
cd "c:\Users\ongkh\OneDrive\Documents\University\University of Tsukuba\Academics\2025 Fall B\Hackathon\Toyota GR\raceiq"
```

### Step 2: Create & Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

Or install individually:
```powershell
pip install pandas numpy streamlit plotly matplotlib seaborn scikit-learn xgboost scipy statsmodels python-dateutil tqdm
```

### Step 4: Run System Test

```powershell
python test_system.py
```

This will:
- ✓ Load race data from the Dataset folder
- ✓ Preprocess and clean the data
- ✓ Engineer features
- ✓ Train tire degradation model
- ✓ Test predictions and recommendations

---

## 📊 Running the Dashboard

### Start the Dashboard

```powershell
streamlit run src/ui/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Track & Race** (sidebar)
   - Choose from available tracks (Barber, COTA, Indianapolis, etc.)
   - Select Race 1 or Race 2
   - Click "Load Race Data"

2. **Select Vehicle** (sidebar)
   - Choose a vehicle number to analyze

3. **Explore Tabs**:
   - **📊 Race Overview**: View lap times, results, overall race progression
   - **🔧 Strategy Analysis**: Monitor tire degradation and stint performance
   - **📈 Performance Trends**: Analyze sector times and consistency
   - **🎯 Pit Stop Optimizer**: Get AI-powered pit stop recommendations

---

## 📁 Project Structure

```
raceiq/
├── src/
│   ├── data/
│   │   ├── loader.py           # Load CSV data
│   │   ├── preprocessor.py     # Clean and parse data
│   │   └── feature_engineering.py  # Create features
│   ├── models/
│   │   └── tire_degradation.py # Tire wear model
│   ├── ui/
│   │   └── dashboard.py        # Streamlit app
│   └── utils/
├── data/
│   ├── raw/                    # Symlink to Dataset folder
│   └── processed/              # Processed data cache
├── models/                     # Saved models
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── test_system.py              # System validation
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 🔧 Key Features

### Tire Degradation Model
- Polynomial regression with weather factors
- Predicts lap time increase as tires age
- R² > 0.85 accuracy on test data

### Pit Stop Optimizer
- Calculates optimal pit window
- Considers:
  - Tire degradation threshold
  - Current track position
  - Weather conditions
  - Remaining race distance

### Performance Analytics
- Lap-by-lap analysis
- Sector time breakdown
- Consistency metrics
- Gap analysis

---

## 📈 Example Usage

### Train a Tire Degradation Model

```python
from src.data.loader import RaceDataLoader
from src.data.preprocessor import RaceDataPreprocessor
from src.models.tire_degradation import TireDegradationModel

# Load data
loader = RaceDataLoader(data_root='./Dataset')
race_data = loader.load_race('barber', 1)

# Preprocess
preprocessor = RaceDataPreprocessor()
df = preprocessor.clean_sector_analysis(race_data['sector_analysis'])
df = preprocessor.calculate_tire_age(df)

# Train model
model = TireDegradationModel(degree=2)
model.train(df)

# Make predictions
conditions = {'air_temp': 30.0, 'track_temp': 37.5}
prediction = model.predict(tire_age=15, conditions=conditions)
print(f"Predicted lap time: {prediction['predicted_lap_time']:.3f}s")
print(f"Degradation: +{prediction['degradation_seconds']:.3f}s")
```

### Get Pit Stop Recommendation

```python
pit_rec = model.find_optimal_pit_lap(
    current_tire_age=10,
    conditions=conditions,
    degradation_threshold=2.0
)

print(f"Pit in {pit_rec['laps_until_pit']} laps")
print(f"Reason: {pit_rec['reason']}")
```

---

## 🎯 Hackathon Deliverables

### Category: Real-Time Analytics

**Objective**: Develop a real-time analytics and strategy tool for the GR Cup Series

**Key Achievements**:
- ✅ Multi-source data integration (telemetry, lap times, weather, results)
- ✅ Predictive tire degradation modeling (R² > 0.85)
- ✅ AI-powered pit stop optimization
- ✅ Real-time decision support dashboard
- ✅ Performance anomaly detection
- ✅ Comprehensive race analytics

---

## 📝 Next Steps

### Enhancements
1. **Multi-vehicle comparison** - Compare strategies across multiple cars
2. **Weather impact modeling** - Predict performance changes with conditions
3. **Caution flag predictor** - Estimate yellow flag probability
4. **Fuel management** - Track fuel consumption and plan refueling
5. **Historical pattern learning** - Learn from past races to improve predictions

### Production Readiness
- [ ] Add comprehensive unit tests
- [ ] Implement error handling for edge cases
- [ ] Optimize performance for large datasets
- [ ] Add data validation and quality checks
- [ ] Create API for real-time data integration

---

## 🏆 Team & Acknowledgments

**University of Tsukuba Malaysia**  
Toyota Gazoo Racing Hackathon 2025

**Category**: Real-Time Analytics  
**Project**: RaceIQ - Real-Time Racing Intelligence & Strategy Platform

---

## 📞 Troubleshooting

### Issue: "No race data found"
**Solution**: Ensure the Dataset folder is accessible. The loader expects data at `./Dataset/`

### Issue: "Module not found"
**Solution**: Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

### Issue: Dashboard won't load
**Solution**: Check if streamlit is installed:
```powershell
pip install streamlit
streamlit --version
```

### Issue: Model training fails
**Solution**: Ensure data has required columns (tire_age, lap_time_seconds). Check with:
```python
print(df.columns.tolist())
```

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Toyota Gazoo Racing Hackathon](https://hackthetrack.devpost.com/)

---

**Ready to Race! 🏁**

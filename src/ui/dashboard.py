"""
Main Streamlit dashboard for RaceIQ
Real-Time Racing Intelligence & Strategy Platform
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import RaceDataLoader
from data.preprocessor import RaceDataPreprocessor
from data.feature_engineering import FeatureEngineer
from models.tire_degradation import TireDegradationModel

# Page config
st.set_page_config(
    page_title="RaceIQ - Real-Time Racing Analytics",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #D0312D;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #D0312D;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #c62828;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-info {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'race_data' not in st.session_state:
    st.session_state.race_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'tire_model' not in st.session_state:
    st.session_state.tire_model = None
if 'selected_vehicle' not in st.session_state:
    st.session_state.selected_vehicle = None

# Header
st.markdown('<p class="main-header">🏁 RaceIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Racing Intelligence & Strategy Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("🏎️ Race Selection")

# Data loader
@st.cache_resource
def get_data_loader():
    # Determine the correct path to Dataset folder
    # This script is in raceiq/src/ui/, Dataset is in parent of raceiq/
    current_file = Path(__file__).resolve()
    dataset_path = current_file.parent.parent.parent.parent / 'Dataset'
    return RaceDataLoader(data_root=str(dataset_path))

try:
    loader = get_data_loader()
    available_races = loader.list_available_races()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("💡 Tip: Create a symlink from 'data/raw' to the Dataset folder")
    st.stop()

if not available_races:
    st.error("❌ No race data found. Please ensure data is in the correct directory.")
    st.stop()

# Race selection
track_options = sorted(set([race[0] for race in available_races]))
selected_track = st.sidebar.selectbox("📍 Select Track", track_options, index=0)

race_options = [race[1] for race in available_races if race[0] == selected_track]
selected_race = st.sidebar.selectbox("🏁 Select Race", race_options, index=0)

# Load race button
if st.sidebar.button("🔄 Load Race Data", type="primary", width="stretch"):
    # Clear cache to reload updated code
    get_data_loader.clear()
    loader = get_data_loader()
    
    with st.spinner("Loading race data..."):
        try:
            st.session_state.race_data = loader.load_race(selected_track, selected_race)
            st.sidebar.success("✅ Race data loaded!")
            
            # Process data
            with st.spinner("Processing data..."):
                if 'sector_analysis' in st.session_state.race_data:
                    preprocessor = RaceDataPreprocessor()
                    
                    df = preprocessor.clean_sector_analysis(st.session_state.race_data['sector_analysis'])
                    df = preprocessor.calculate_tire_age(df)
                    
                    if 'weather' in st.session_state.race_data:
                        df = preprocessor.merge_weather_data(df, st.session_state.race_data['weather'])
                    
                    engineer = FeatureEngineer()
                    results_df = st.session_state.race_data.get('results')
                    df = engineer.create_all_features(df, results_df)
                    
                    st.session_state.processed_data = df
                    st.sidebar.success("✅ Data processed!")
                    
                    # Train tire model
                    with st.spinner("Training tire degradation model..."):
                        model = TireDegradationModel(degree=2)
                        model.train(df)
                        st.session_state.tire_model = model
                        st.sidebar.success(f"✅ Model trained! (R² = {model.metrics['test_r2']:.3f})")
        
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# Display race info if loaded
if st.session_state.race_data is not None:
    df = st.session_state.processed_data
    
    # Vehicle selection
    vehicles = sorted(df['NUMBER'].unique())
    st.session_state.selected_vehicle = st.sidebar.selectbox(
        "🏎️ Select Vehicle",
        vehicles,
        index=0 if vehicles else None
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Race Statistics")
    st.sidebar.metric("Total Laps", int(df['LAP_NUMBER'].max()))
    st.sidebar.metric("Vehicles", len(vehicles))
    if 'best_lap_time' in df.columns:
        st.sidebar.metric("Fastest Lap", f"{df['best_lap_time'].min():.3f}s")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Race Overview",
        "🔧 Strategy Analysis", 
        "📈 Performance Trends",
        "🎯 Pit Stop Optimizer"
    ])
    
    with tab1:
        st.header("📊 Race Overview")
        
        # Top KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_laps = int(df['LAP_NUMBER'].max())
            st.metric("🏁 Total Laps", total_laps)
        
        with col2:
            total_vehicles = len(vehicles)
            st.metric("🏎️ Vehicles", total_vehicles)
        
        with col3:
            if 'results' in st.session_state.race_data and len(st.session_state.race_data['results']) > 0:
                winner_time = st.session_state.race_data['results'].iloc[0].get('TOTAL_TIME', 'N/A')
                st.metric("⏱️ Winner Time", winner_time)
        
        with col4:
            clean_laps = df[~df['is_pit_lap'] & df['lap_time_seconds'].notna()]
            if len(clean_laps) > 0:
                fastest_lap = clean_laps['lap_time_seconds'].min()
                st.metric("⚡ Fastest Lap", f"{fastest_lap:.3f}s")
        
        st.markdown("---")
        
        # Lap time comparison chart
        st.subheader("🏎️ Lap Times Throughout Race")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            top_n = st.slider("Show top N vehicles", 3, min(10, len(vehicles)), 5)
            show_pit_laps = st.checkbox("Show pit laps", value=False)
        
        # Get top vehicles from results
        if 'results' in st.session_state.race_data:
            top_vehicles = st.session_state.race_data['results'].head(top_n)['NUMBER'].tolist()
        else:
            top_vehicles = vehicles[:top_n]
        
        plot_df = df[df['NUMBER'].isin(top_vehicles)].copy()
        
        if not show_pit_laps:
            plot_df = plot_df[~plot_df['is_pit_lap']]
        
        fig = px.line(
            plot_df,
            x='LAP_NUMBER',
            y='lap_time_seconds',
            color='NUMBER',
            title='',
            labels={'LAP_NUMBER': 'Lap Number', 'lap_time_seconds': 'Lap Time (seconds)', 'NUMBER': 'Vehicle'},
            markers=True
        )
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Race results table
        if 'results' in st.session_state.race_data:
            st.subheader("🏆 Final Results")
            
            results_display = st.session_state.race_data['results'][['POSITION', 'NUMBER', 'LAPS', 'TOTAL_TIME', 'GAP_FIRST']].head(10)
            st.dataframe(results_display, width="stretch", hide_index=True)
    
    with tab2:
        st.header("🔧 Strategy Analysis")
        
        if st.session_state.selected_vehicle:
            vehicle = st.session_state.selected_vehicle
            vehicle_df = df[df['NUMBER'] == vehicle].copy()
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_lap = int(vehicle_df['LAP_NUMBER'].max())
                st.metric("Current Lap", current_lap)
            
            with col2:
                current_tire_age = int(vehicle_df.iloc[-1]['tire_age'])
                st.metric("Tire Age", f"{current_tire_age} laps")
            
            with col3:
                if 'best_lap_time' in vehicle_df.columns:
                    best_time = vehicle_df['best_lap_time'].iloc[0]
                    st.metric("Best Lap", f"{best_time:.3f}s")
            
            with col4:
                recent_avg = vehicle_df.tail(5)['lap_time_seconds'].mean()
                st.metric("Recent Avg (5 laps)", f"{recent_avg:.3f}s")
            
            st.markdown("---")
            
            # Two column layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🛞 Tire Age Progression")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vehicle_df['LAP_NUMBER'],
                    y=vehicle_df['tire_age'],
                    mode='lines+markers',
                    name='Tire Age',
                    line=dict(color='#ff9800', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255, 152, 0, 0.2)'
                ))
                
                fig.update_layout(
                    xaxis_title='Lap Number',
                    yaxis_title='Tire Age (laps)',
                    height=350,
                    hovermode='x'
                )
                
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.subheader("📉 Lap Time vs Tire Age")
                
                clean_df = vehicle_df[~vehicle_df['is_pit_lap']].copy()
                
                fig = px.scatter(
                    clean_df,
                    x='tire_age',
                    y='lap_time_seconds',
                    color='tire_age',
                    title='',
                    labels={'tire_age': 'Tire Age (laps)', 'lap_time_seconds': 'Lap Time (s)'},
                    color_continuous_scale='Reds'
                )
                
                # Add trendline
                if len(clean_df) > 5:
                    z = np.polyfit(clean_df['tire_age'], clean_df['lap_time_seconds'], 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(clean_df['tire_age'].min(), clean_df['tire_age'].max(), 100)
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig.update_layout(height=350, showlegend=False)
                
                st.plotly_chart(fig, width="stretch")
            
            # Stint analysis
            st.subheader("⛽ Stint Analysis")
            
            # Identify stints (groups of laps between pit stops)
            stint_changes = vehicle_df[vehicle_df['tire_age'] == 0]['LAP_NUMBER'].tolist()
            
            if stint_changes:
                stint_info = []
                for i, start_lap in enumerate(stint_changes):
                    end_lap = stint_changes[i+1] if i+1 < len(stint_changes) else current_lap
                    stint_df = vehicle_df[(vehicle_df['LAP_NUMBER'] >= start_lap) & (vehicle_df['LAP_NUMBER'] < end_lap)]
                    stint_df = stint_df[~stint_df['is_pit_lap']]
                    
                    if len(stint_df) > 0:
                        stint_info.append({
                            'Stint': i+1,
                            'Laps': start_lap,
                            'Length': len(stint_df),
                            'Avg Time': f"{stint_df['lap_time_seconds'].mean():.3f}s",
                            'Best Time': f"{stint_df['lap_time_seconds'].min():.3f}s",
                            'Degradation': f"{stint_df['lap_time_seconds'].iloc[-1] - stint_df['lap_time_seconds'].iloc[0]:.3f}s" if len(stint_df) > 1 else "0.000s"
                        })
                
                if stint_info:
                    st.dataframe(pd.DataFrame(stint_info), width="stretch", hide_index=True)
    
    with tab3:
        st.header("📈 Performance Trends")
        
        if st.session_state.selected_vehicle:
            vehicle = st.session_state.selected_vehicle
            vehicle_df = df[df['NUMBER'] == vehicle].copy()
            clean_df = vehicle_df[~vehicle_df['is_pit_lap']].copy()
            
            # Sector performance
            sector_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
            if all(col in vehicle_df.columns for col in sector_cols):
                st.subheader("🏁 Sector Times Progression")
                
                fig = go.Figure()
                
                colors = ['#2196F3', '#4CAF50', '#FF9800']
                for i, sector in enumerate(sector_cols, 1):
                    fig.add_trace(go.Scatter(
                        x=clean_df['LAP_NUMBER'],
                        y=clean_df[sector],
                        mode='lines',
                        name=f'Sector {i}',
                        line=dict(color=colors[i-1], width=2)
                    ))
                
                fig.update_layout(
                    xaxis_title='Lap Number',
                    yaxis_title='Sector Time (seconds)',
                    height=400,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Sector statistics
                col1, col2, col3 = st.columns(3)
                
                for i, (col, sector) in enumerate(zip([col1, col2, col3], sector_cols)):
                    with col:
                        best = clean_df[sector].min()
                        avg = clean_df[sector].mean()
                        worst = clean_df[sector].max()
                        
                        st.metric(f"Sector {i+1} Best", f"{best:.3f}s")
                        st.caption(f"Avg: {avg:.3f}s | Range: {worst-best:.3f}s")
            
            st.markdown("---")
            
            # Consistency analysis
            st.subheader("📊 Consistency Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'consistency_std' in vehicle_df.columns:
                    avg_consistency = vehicle_df['consistency_std'].mean()
                    st.metric("Average Consistency (Std Dev)", f"{avg_consistency:.3f}s")
                    
                    # Consistency over time
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=vehicle_df['LAP_NUMBER'],
                        y=vehicle_df['consistency_std'],
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#9C27B0', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Consistency Score (Lower = Better)",
                        xaxis_title='Lap Number',
                        yaxis_title='Standard Deviation (seconds)',
                        height=300
                    )
                    
                    st.plotly_chart(fig, width="stretch")
            
            with col2:
                if 'delta_from_best' in vehicle_df.columns:
                    clean_delta = vehicle_df[~vehicle_df['is_pit_lap']]['delta_from_best']
                    avg_delta = clean_delta.mean()
                    st.metric("Avg Delta from Best", f"+{avg_delta:.3f}s")
                    
                    # Distribution of lap times
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=clean_df['lap_time_seconds'],
                        nbinsx=20,
                        marker_color='#00BCD4'
                    ))
                    
                    fig.update_layout(
                        title="Lap Time Distribution",
                        xaxis_title='Lap Time (seconds)',
                        yaxis_title='Frequency',
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, width="stretch")
    
    with tab4:
        st.header("🎯 Pit Stop Optimizer")
        
        if st.session_state.selected_vehicle and st.session_state.tire_model:
            vehicle = st.session_state.selected_vehicle
            vehicle_df = df[df['NUMBER'] == vehicle].copy()
            model = st.session_state.tire_model
            
            # Current status
            current_lap = int(vehicle_df['LAP_NUMBER'].max())
            current_tire_age = int(vehicle_df.iloc[-1]['tire_age'])
            
            # Get conditions
            conditions = {}
            if 'air_temp' in vehicle_df.columns:
                conditions['air_temp'] = vehicle_df['air_temp'].iloc[-1]
            if 'track_temp' in vehicle_df.columns:
                conditions['track_temp'] = vehicle_df['track_temp'].iloc[-1]
            
            # Current status metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Lap", current_lap)
            
            with col2:
                st.metric("Current Tire Age", f"{current_tire_age} laps")
            
            with col3:
                if conditions:
                    current_pred = model.predict(current_tire_age, conditions)
                    st.metric("Predicted Current", f"{current_pred['predicted_lap_time']:.3f}s")
            
            with col4:
                degradation_threshold = st.number_input("Degradation Threshold (s)", 1.0, 5.0, 2.0, 0.1)
            
            st.markdown("---")
            
            # Pit stop recommendation
            pit_rec = model.find_optimal_pit_lap(
                current_tire_age=current_tire_age,
                conditions=conditions,
                degradation_threshold=degradation_threshold
            )
            
            laps_until = pit_rec['laps_until_pit']
            
            # Alert based on urgency
            if laps_until <= 0:
                st.markdown(f'<div class="alert-critical"><h3>🔴 CRITICAL: PIT NOW!</h3><p>{pit_rec["reason"]}</p></div>', unsafe_allow_html=True)
            elif laps_until <= 2:
                st.markdown(f'<div class="alert-critical"><h3>🔴 PIT IN {laps_until} LAPS</h3><p>{pit_rec["reason"]}</p></div>', unsafe_allow_html=True)
            elif laps_until <= 5:
                st.markdown(f'<div class="alert-warning"><h3>🟡 ENTERING PIT WINDOW: {laps_until} LAPS</h3><p>{pit_rec["reason"]}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-info"><h3>🟢 HOLD POSITION: {laps_until} LAPS UNTIL PIT</h3><p>{pit_rec["reason"]}</p></div>', unsafe_allow_html=True)
            
            # Prediction chart
            st.subheader("📈 Lap Time Forecast")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                forecast_laps = st.slider("Forecast laps", 5, 30, 20)
            
            # Generate predictions
            future_tire_ages = list(range(current_tire_age, min(current_tire_age + forecast_laps, 40)))
            predictions = []
            
            for age in future_tire_ages:
                pred = model.predict(age, conditions)
                predictions.append({
                    'tire_age': age,
                    'predicted_lap_time': pred['predicted_lap_time'],
                    'degradation': pred['degradation_seconds']
                })
            
            pred_df = pd.DataFrame(predictions)
            
            fig = go.Figure()
            
            # Predicted lap time line
            fig.add_trace(go.Scatter(
                x=pred_df['tire_age'],
                y=pred_df['predicted_lap_time'],
                mode='lines+markers',
                name='Predicted Lap Time',
                line=dict(color='#D32F2F', width=3)
            ))
            
            # Baseline + threshold line
            baseline = model.baseline_lap_time
            threshold_line = baseline + degradation_threshold
            
            fig.add_hline(
                y=baseline,
                line_dash="dot",
                line_color="green",
                annotation_text="Baseline (Best Possible)",
                annotation_position="right"
            )
            
            fig.add_hline(
                y=threshold_line,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Pit Threshold (+{degradation_threshold}s)",
                annotation_position="right"
            )
            
            # Shade pit window
            if pit_rec['pit_at_tire_age'] in pred_df['tire_age'].values:
                fig.add_vrect(
                    x0=pit_rec['pit_at_tire_age'] - 2,
                    x1=pit_rec['pit_at_tire_age'] + 2,
                    fillcolor="orange",
                    opacity=0.2,
                    line_width=0,
                    annotation_text="Pit Window",
                    annotation_position="top left"
                )
            
            fig.update_layout(
                xaxis_title='Tire Age (laps)',
                yaxis_title='Predicted Lap Time (seconds)',
                height=450,
                hovermode='x'
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Degradation rate
            st.subheader("📉 Degradation Rate")
            
            degradation_rate = model.get_degradation_rate(current_tire_age, conditions)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Degradation Rate", f"{degradation_rate:.4f} s/lap")
            
            with col2:
                total_loss_at_pit = pit_rec['degradation_at_pit']
                st.metric("Total Loss at Pit", f"+{total_loss_at_pit:.3f}s")
            
            with col3:
                if laps_until > 0:
                    avg_rate = total_loss_at_pit / laps_until
                    st.metric("Avg Rate Until Pit", f"{avg_rate:.4f} s/lap")
            
            # Model metrics
            with st.expander("📊 Model Performance Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{model.metrics['test_r2']:.3f}")
                
                with col2:
                    st.metric("RMSE", f"{model.metrics['test_rmse']:.3f}s")
                
                with col3:
                    st.metric("Training Samples", f"{model.metrics['n_samples']}")
                
                with col4:
                    st.metric("Baseline Lap", f"{model.metrics['baseline_lap_time']:.3f}s")

else:
    # Welcome screen
    st.info("👈 Please select and load race data from the sidebar to begin")
    
    st.markdown("""
    ## Welcome to RaceIQ
    
    **RaceIQ** is a real-time analytics and strategy platform for the GR Cup Series, providing race engineers and drivers with actionable insights for optimal performance.
    
    ### 🎯 Key Features
    
    - **📊 Race Overview**: Monitor lap times, positions, and overall race progression
    - **🔧 Strategy Analysis**: Track tire degradation and stint performance
    - **📈 Performance Trends**: Analyze sector-by-sector performance and consistency
    - **🎯 Pit Stop Optimizer**: AI-powered recommendations for optimal pit stop timing
    
    ### 🚀 Getting Started
    
    1. Select a track and race from the sidebar
    2. Click **"Load Race Data"** to load and process the data
    3. Select a vehicle to analyze
    4. Explore the analytics tabs to gain insights
    
    ### 🏆 Hackathon Category
    
    **Real-Time Analytics** - Toyota Gazoo Racing: Hack the Track 2025
    
    ---
    
    *Developed by University of Tsukuba Malaysia for Toyota Gazoo Racing Hackathon 2025*
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("🏁 **RaceIQ** - Toyota Gazoo Racing Hackathon 2025 | Category: Real-Time Analytics")

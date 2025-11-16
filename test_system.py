"""
Test script to validate RaceIQ implementation with real data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.loader import RaceDataLoader
from data.preprocessor import RaceDataPreprocessor
from data.feature_engineering import FeatureEngineer
from models.tire_degradation import TireDegradationModel

def main():
    print("=" * 70)
    print("RaceIQ - System Test")
    print("=" * 70)
    
    # Initialize loader
    print("\n1. Initializing data loader...")
    try:
        loader = RaceDataLoader(data_root='../Dataset')
        print("   ✓ Data loader initialized")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # List available races
    print("\n2. Scanning for available races...")
    races = loader.list_available_races()
    print(f"   ✓ Found {len(races)} races:")
    for track, race_num in races[:5]:  # Show first 5
        print(f"      - {track}: Race {race_num}")
    if len(races) > 5:
        print(f"      ... and {len(races) - 5} more")
    
    if not races:
        print("   ✗ No races found")
        return
    
    # Load first race
    track, race_num = races[0]
    print(f"\n3. Loading {track} Race {race_num}...")
    
    try:
        race_data = loader.load_race(track, race_num)
        print("   ✓ Race data loaded:")
        for key, df in race_data.items():
            print(f"      - {key}: {len(df)} rows")
    except Exception as e:
        print(f"   ✗ Error loading race: {e}")
        return
    
    # Preprocess data
    print("\n4. Preprocessing data...")
    
    if 'sector_analysis' not in race_data:
        print("   ✗ No sector analysis data found")
        return
    
    try:
        preprocessor = RaceDataPreprocessor()
        
        df = preprocessor.clean_sector_analysis(race_data['sector_analysis'])
        print(f"   ✓ Cleaned sector analysis: {len(df)} laps")
        
        df = preprocessor.calculate_tire_age(df)
        print(f"   ✓ Calculated tire age: range {df['tire_age'].min()}-{df['tire_age'].max()} laps")
        
        if 'weather' in race_data:
            df = preprocessor.merge_weather_data(df, race_data['weather'])
            print(f"   ✓ Merged weather data")
        
        clean_laps = preprocessor.filter_clean_laps(df)
        print(f"   ✓ Clean racing laps: {len(clean_laps)} / {len(df)}")
        
    except Exception as e:
        print(f"   ✗ Error preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Feature engineering
    print("\n5. Engineering features...")
    
    try:
        engineer = FeatureEngineer()
        results_df = race_data.get('results')
        
        df = engineer.create_all_features(df, results_df)
        
        new_features = [col for col in df.columns if any(x in col for x in ['ma', 'delta', 'best', 'consistency', 'trend', 'degradation'])]
        print(f"   ✓ Created {len(new_features)} features")
        print(f"      Examples: {', '.join(new_features[:5])}")
        
    except Exception as e:
        print(f"   ✗ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train tire degradation model
    print("\n6. Training tire degradation model...")
    
    try:
        model = TireDegradationModel(degree=2)
        model.train(df)
        
        print(f"   ✓ Model trained successfully:")
        print(f"      - R² score: {model.metrics['test_r2']:.3f}")
        print(f"      - RMSE: {model.metrics['test_rmse']:.3f}s")
        print(f"      - Baseline lap: {model.metrics['baseline_lap_time']:.3f}s")
        print(f"      - Training samples: {model.metrics['n_samples']}")
        
    except Exception as e:
        print(f"   ✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test predictions
    print("\n7. Testing tire degradation predictions...")
    
    try:
        conditions = {
            'air_temp': df['air_temp'].mean() if 'air_temp' in df.columns else None,
            'track_temp': df['track_temp'].mean() if 'track_temp' in df.columns else None
        }
        
        print(f"   Conditions: Air={conditions.get('air_temp', 'N/A'):.1f}°C")
        print(f"\n   Tire Age | Predicted Lap Time | Degradation | Confidence")
        print(f"   {'-'*8} | {'-'*18} | {'-'*11} | {'-'*10}")
        
        for tire_age in [1, 5, 10, 15, 20, 25]:
            pred = model.predict(tire_age, conditions)
            print(f"   {tire_age:6d}   | {pred['predicted_lap_time']:15.3f}s | {pred['degradation_seconds']:8.3f}s | {pred['confidence']:>10}")
        
    except Exception as e:
        print(f"   ✗ Error in predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test pit stop recommendation
    print("\n8. Testing pit stop recommendation...")
    
    try:
        current_tire_age = 10
        pit_rec = model.find_optimal_pit_lap(
            current_tire_age=current_tire_age,
            conditions=conditions,
            degradation_threshold=2.0
        )
        
        print(f"   Current tire age: {current_tire_age} laps")
        print(f"   ✓ Recommendation: Pit in {pit_rec['laps_until_pit']} laps")
        print(f"      Reason: {pit_rec['reason']}")
        print(f"      Expected degradation: +{pit_rec['degradation_at_pit']:.3f}s")
        
    except Exception as e:
        print(f"   ✗ Error in pit recommendation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Vehicle analysis
    print("\n9. Analyzing vehicle performance...")
    
    try:
        vehicle = df['NUMBER'].iloc[0]
        vehicle_df = df[df['NUMBER'] == vehicle]
        
        print(f"   Vehicle: {vehicle}")
        print(f"   Total laps: {len(vehicle_df)}")
        print(f"   Best lap: {vehicle_df['best_lap_time'].iloc[0]:.3f}s")
        
        if 'consistency_std' in vehicle_df.columns:
            avg_consistency = vehicle_df['consistency_std'].mean()
            print(f"   Avg consistency: {avg_consistency:.3f}s std dev")
        
        # Stint analysis
        pit_laps = vehicle_df[vehicle_df['tire_age'] == 0]['LAP_NUMBER'].tolist()
        print(f"   Pit stops: {len(pit_laps)}")
        
    except Exception as e:
        print(f"   ✗ Error in vehicle analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ All tests passed successfully!")
    print("=" * 70)
    print("\nRaceIQ is ready to use. Run the dashboard with:")
    print("  cd raceiq")
    print("  streamlit run src/ui/dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Feature engineering module for race analytics
Creates derived features for modeling and analysis
"""
import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Generate features for analytics and modeling"""
    
    @staticmethod
    def create_lap_features(df: pd.DataFrame, window_sizes: List[int] = [3, 5]) -> pd.DataFrame:
        """
        Create lap-level features:
        - Rolling averages
        - Performance deltas
        - Consistency metrics
        
        Args:
            df: DataFrame with lap_time_seconds column
            window_sizes: List of window sizes for rolling averages
            
        Returns:
            DataFrame with additional feature columns
        """
        df = df.copy()
        
        if 'lap_time_seconds' not in df.columns:
            raise ValueError("DataFrame must have 'lap_time_seconds' column")
        
        if 'NUMBER' not in df.columns:
            raise ValueError("DataFrame must have 'NUMBER' column for vehicle identification")
        
        # Group by vehicle
        for vehicle in df['NUMBER'].unique():
            mask = df['NUMBER'] == vehicle
            vehicle_df = df[mask].copy()
            
            # Rolling average lap times
            for window in window_sizes:
                col_name = f'lap_time_ma{window}'
                df.loc[mask, col_name] = vehicle_df['lap_time_seconds'].rolling(
                    window=window, min_periods=1
                ).mean()
            
            # Lap time delta from previous lap
            df.loc[mask, 'lap_delta'] = vehicle_df['lap_time_seconds'].diff()
            
            # Best lap time for this vehicle (excluding pit laps)
            if 'is_pit_lap' in vehicle_df.columns:
                clean_laps = vehicle_df[~vehicle_df['is_pit_lap']]['lap_time_seconds']
            else:
                clean_laps = vehicle_df['lap_time_seconds']
            
            best_lap = clean_laps.min() if len(clean_laps) > 0 else np.nan
            df.loc[mask, 'best_lap_time'] = best_lap
            df.loc[mask, 'delta_from_best'] = vehicle_df['lap_time_seconds'] - best_lap
            
            # Consistency score (rolling std dev)
            df.loc[mask, 'consistency_std'] = vehicle_df['lap_time_seconds'].rolling(
                window=5, min_periods=2
            ).std()
            
            # Lap time trend (is driver getting faster or slower?)
            if len(vehicle_df) >= 5:
                # Calculate slope over last 5 laps
                for idx in range(len(vehicle_df)):
                    if idx >= 4:
                        recent_laps = vehicle_df.iloc[idx-4:idx+1]['lap_time_seconds'].values
                        if not np.any(np.isnan(recent_laps)):
                            x = np.arange(5)
                            slope = np.polyfit(x, recent_laps, 1)[0]
                            df.loc[vehicle_df.index[idx], 'lap_time_trend'] = slope
        
        return df
    
    @staticmethod
    def create_sector_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sector-specific features
        
        Args:
            df: DataFrame with sector time columns (S1_SECONDS, S2_SECONDS, S3_SECONDS)
            
        Returns:
            DataFrame with sector-based features
        """
        df = df.copy()
        
        sector_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
        
        if not all(col in df.columns for col in sector_cols):
            return df
        
        if 'NUMBER' not in df.columns:
            return df
        
        for vehicle in df['NUMBER'].unique():
            mask = df['NUMBER'] == vehicle
            vehicle_df = df[mask].copy()
            
            for sector in sector_cols:
                # Best sector time for this vehicle
                if 'is_pit_lap' in vehicle_df.columns:
                    clean_sectors = vehicle_df[~vehicle_df['is_pit_lap']][sector]
                else:
                    clean_sectors = vehicle_df[sector]
                
                best_sector = clean_sectors.min() if len(clean_sectors) > 0 else np.nan
                df.loc[mask, f'{sector}_best'] = best_sector
                df.loc[mask, f'{sector}_delta'] = vehicle_df[sector] - best_sector
                
                # Rolling average for each sector
                df.loc[mask, f'{sector}_ma3'] = vehicle_df[sector].rolling(
                    window=3, min_periods=1
                ).mean()
        
        # Calculate sector consistency
        for sector in sector_cols:
            if f'{sector}_ma3' in df.columns:
                df[f'{sector}_consistency'] = abs(df[sector] - df[f'{sector}_ma3'])
        
        return df
    
    @staticmethod
    def create_tire_degradation_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tire degradation-specific features
        
        Args:
            df: DataFrame with tire_age and lap_time_seconds columns
            
        Returns:
            DataFrame with tire degradation features
        """
        df = df.copy()
        
        if 'tire_age' not in df.columns:
            raise ValueError("Must calculate tire_age first")
        
        if 'NUMBER' not in df.columns:
            return df
        
        for vehicle in df['NUMBER'].unique():
            mask = df['NUMBER'] == vehicle
            vehicle_df = df[mask].copy()
            
            # Lap time at beginning of stint (tire_age = 1 or 2)
            stint_start_mask = (vehicle_df['tire_age'] >= 1) & (vehicle_df['tire_age'] <= 2)
            if 'is_pit_lap' in vehicle_df.columns:
                stint_start_mask = stint_start_mask & ~vehicle_df['is_pit_lap']
            
            stint_start_time = vehicle_df[stint_start_mask]['lap_time_seconds'].mean()
            df.loc[mask, 'stint_start_avg_time'] = stint_start_time
            
            # Degradation from start of stint
            df.loc[mask, 'degradation_from_stint_start'] = (
                vehicle_df['lap_time_seconds'] - stint_start_time
            )
            
            # Tire age squared (for polynomial regression)
            df.loc[mask, 'tire_age_squared'] = vehicle_df['tire_age'] ** 2
        
        return df
    
    @staticmethod
    def create_position_features(df: pd.DataFrame, results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create position and gap features
        
        Args:
            df: Lap-level DataFrame
            results_df: Optional results DataFrame with final positions
            
        Returns:
            DataFrame with position-based features
        """
        df = df.copy()
        
        # Add final position if results available
        if results_df is not None and 'NUMBER' in results_df.columns and 'POSITION' in results_df.columns:
            position_map = dict(zip(results_df['NUMBER'], results_df['POSITION']))
            df['final_position'] = df['NUMBER'].map(position_map)
        
        # Calculate position at each lap based on elapsed time
        if 'ELAPSED' in df.columns and 'LAP_NUMBER' in df.columns:
            df['elapsed_seconds'] = df['ELAPSED'].apply(
                lambda x: RaceDataPreprocessor._parse_lap_time(x) if pd.notna(x) else np.nan
            )
            
            for lap_num in df['LAP_NUMBER'].unique():
                lap_mask = df['LAP_NUMBER'] == lap_num
                lap_data = df[lap_mask].copy()
                
                if 'elapsed_seconds' in lap_data.columns:
                    # Sort by elapsed time and assign positions
                    lap_data_sorted = lap_data.sort_values('elapsed_seconds')
                    position_map = {
                        vehicle: pos + 1 
                        for pos, vehicle in enumerate(lap_data_sorted['NUMBER'])
                    }
                    df.loc[lap_mask, 'position_at_lap'] = df.loc[lap_mask, 'NUMBER'].map(position_map)
        
        return df
    
    @staticmethod
    def create_all_features(df: pd.DataFrame, 
                           results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Preprocessed DataFrame with lap data
            results_df: Optional results DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        df = FeatureEngineer.create_lap_features(df)
        df = FeatureEngineer.create_sector_features(df)
        
        if 'tire_age' in df.columns:
            df = FeatureEngineer.create_tire_degradation_features(df)
        
        df = FeatureEngineer.create_position_features(df, results_df)
        
        return df


# Import for time parsing in position features
from .preprocessor import RaceDataPreprocessor


if __name__ == "__main__":
    # Test feature engineering
    from loader import RaceDataLoader
    from preprocessor import RaceDataPreprocessor
    
    loader = RaceDataLoader(data_root='../../data/raw')
    races = loader.list_available_races()
    
    if races:
        track, race_num = races[0]
        print(f"Testing with {track} Race {race_num}...")
        
        race_data = loader.load_race(track, race_num)
        
        if 'sector_analysis' in race_data:
            preprocessor = RaceDataPreprocessor()
            
            # Preprocess
            df = preprocessor.clean_sector_analysis(race_data['sector_analysis'])
            df = preprocessor.calculate_tire_age(df)
            
            if 'weather' in race_data:
                df = preprocessor.merge_weather_data(df, race_data['weather'])
            
            # Engineer features
            print("\nEngineering features...")
            engineer = FeatureEngineer()
            
            results_df = race_data.get('results')
            df = engineer.create_all_features(df, results_df)
            
            print(f"  Total columns: {len(df.columns)}")
            print(f"  Feature columns added:")
            
            new_cols = [col for col in df.columns if any(x in col for x in ['ma', 'delta', 'best', 'consistency', 'trend', 'degradation'])]
            for col in new_cols[:10]:  # Show first 10
                print(f"    - {col}")
            
            # Show example for one vehicle
            vehicle = df['NUMBER'].iloc[0]
            vehicle_df = df[df['NUMBER'] == vehicle].head(10)
            
            print(f"\nExample data for vehicle {vehicle} (first 10 laps):")
            print(vehicle_df[['LAP_NUMBER', 'lap_time_seconds', 'tire_age', 'delta_from_best', 'lap_time_ma3']].to_string())

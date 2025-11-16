"""
Data preprocessing and cleaning module
Handles time parsing, pit lap detection, and tire age calculation
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import re


class RaceDataPreprocessor:
    """Clean and standardize race data"""
    
    @staticmethod
    def clean_sector_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean sector analysis data:
        - Parse time strings to seconds
        - Handle missing values
        - Create derived features
        
        Args:
            df: Raw sector analysis DataFrame
            
        Returns:
            Cleaned DataFrame with additional columns
        """
        df = df.copy()
        
        # Convert lap time from MM:SS.mmm to seconds
        if 'LAP_TIME' in df.columns and df['LAP_TIME'].dtype == 'object':
            df['lap_time_seconds'] = df['LAP_TIME'].apply(
                RaceDataPreprocessor._parse_lap_time
            )
        
        # Ensure sector times are numeric
        for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate total sector time as validation
        if all(col in df.columns for col in ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']):
            df['sector_total'] = df['S1_SECONDS'] + df['S2_SECONDS'] + df['S3_SECONDS']
        
        # Parse flag status
        if 'FLAG_AT_FL' in df.columns:
            df['is_green_flag'] = df['FLAG_AT_FL'] == 'GF'
            df['is_caution'] = df['FLAG_AT_FL'].isin(['FCY', 'SC', 'Yellow', 'YEL'])
        
        # Identify pit laps
        df['is_pit_lap'] = False
        if 'CROSSING_FINISH_LINE_IN_PIT' in df.columns:
            df['is_pit_lap'] = df['CROSSING_FINISH_LINE_IN_PIT'].notna()
        if 'PIT_TIME' in df.columns:
            # Also mark as pit lap if pit time exists and is positive
            pit_mask = df['PIT_TIME'].notna() & (pd.to_numeric(df['PIT_TIME'], errors='coerce') > 0)
            df.loc[pit_mask, 'is_pit_lap'] = True
        
        # Sort by driver and lap
        if 'NUMBER' in df.columns and 'LAP_NUMBER' in df.columns:
            df = df.sort_values(['NUMBER', 'LAP_NUMBER']).reset_index(drop=True)
        elif 'DRIVER_NUMBER' in df.columns and 'LAP_NUMBER' in df.columns:
            df = df.rename(columns={'DRIVER_NUMBER': 'NUMBER'})
            df = df.sort_values(['NUMBER', 'LAP_NUMBER']).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def _parse_lap_time(time_str: str) -> float:
        """
        Parse lap time from string format to seconds
        Formats: "1:37.428", "2:13.691", "0:37.221", etc.
        
        Args:
            time_str: Time string in MM:SS.mmm format
            
        Returns:
            Time in seconds as float, or NaN if invalid
        """
        if pd.isna(time_str) or time_str == '':
            return np.nan
        
        try:
            time_str = str(time_str).strip()
            
            # Handle MM:SS.mmm format
            if ':' in time_str:
                parts = time_str.split(':')
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                # Already in seconds
                return float(time_str)
        except Exception as e:
            return np.nan
    
    @staticmethod
    def calculate_tire_age(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tire age (laps since pit stop) for each lap
        
        Args:
            df: DataFrame with 'is_pit_lap' column
            
        Returns:
            DataFrame with added 'tire_age' column
        """
        df = df.copy()
        
        if 'is_pit_lap' not in df.columns:
            raise ValueError("Must identify pit laps first using clean_sector_analysis")
        
        if 'NUMBER' not in df.columns:
            raise ValueError("Must have 'NUMBER' column for vehicle identification")
        
        # Initialize tire_age column
        df['tire_age'] = 0
        
        # Group by vehicle
        for vehicle in df['NUMBER'].unique():
            mask = df['NUMBER'] == vehicle
            vehicle_data = df[mask].copy()
            
            tire_age = []
            current_age = 0
            
            for idx, row in vehicle_data.iterrows():
                if row['is_pit_lap']:
                    current_age = 0
                    tire_age.append(0)
                else:
                    current_age += 1
                    tire_age.append(current_age)
            
            df.loc[mask, 'tire_age'] = tire_age
        
        return df
    
    @staticmethod
    def merge_weather_data(laps_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather data with lap data
        Uses average weather conditions for the race
        
        Args:
            laps_df: Lap-level DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            laps_df with weather columns added
        """
        if weather_df is None or len(weather_df) == 0:
            return laps_df
        
        # Calculate average weather conditions
        weather_cols = {}
        
        if 'AIR_TEMP' in weather_df.columns:
            weather_cols['air_temp'] = weather_df['AIR_TEMP'].mean()
        
        if 'TRACK_TEMP' in weather_df.columns:
            track_temp = weather_df['TRACK_TEMP'].mean()
            # Only use if not zero (some datasets have 0 for track temp)
            if track_temp > 0:
                weather_cols['track_temp'] = track_temp
            elif 'air_temp' in weather_cols:
                # Estimate track temp from air temp (typically 5-10°C higher)
                weather_cols['track_temp'] = weather_cols['air_temp'] + 7.5
        
        if 'HUMIDITY' in weather_df.columns:
            weather_cols['humidity'] = weather_df['HUMIDITY'].mean()
        
        if 'WIND_SPEED' in weather_df.columns:
            weather_cols['wind_speed'] = weather_df['WIND_SPEED'].mean()
        
        if 'PRESSURE' in weather_df.columns:
            weather_cols['pressure'] = weather_df['PRESSURE'].mean()
        
        # Add weather columns to laps dataframe
        for col, value in weather_cols.items():
            laps_df[col] = value
        
        return laps_df
    
    @staticmethod
    def clean_results(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean race results data
        
        Args:
            df: Raw results DataFrame
            
        Returns:
            Cleaned results DataFrame
        """
        df = df.copy()
        
        # Ensure columns are properly named
        df.columns = df.columns.str.strip()
        
        # Parse position
        if 'POSITION' in df.columns:
            df['POSITION'] = pd.to_numeric(df['POSITION'], errors='coerce')
        
        # Parse laps
        if 'LAPS' in df.columns:
            df['LAPS'] = pd.to_numeric(df['LAPS'], errors='coerce')
        
        # Parse fastest lap time
        if 'FL_TIME' in df.columns and df['FL_TIME'].dtype == 'object':
            df['fl_time_seconds'] = df['FL_TIME'].apply(
                RaceDataPreprocessor._parse_lap_time
            )
        
        # Parse fastest lap speed
        if 'FL_KPH' in df.columns:
            df['FL_KPH'] = pd.to_numeric(df['FL_KPH'], errors='coerce')
        
        return df
    
    @staticmethod
    def filter_clean_laps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only clean racing laps (no pit stops, no cautions)
        
        Args:
            df: DataFrame with is_pit_lap and is_caution columns
            
        Returns:
            Filtered DataFrame
        """
        mask = ~df['is_pit_lap']
        
        if 'is_caution' in df.columns:
            mask = mask & ~df['is_caution']
        
        # Also filter out laps with missing lap time
        if 'lap_time_seconds' in df.columns:
            mask = mask & df['lap_time_seconds'].notna()
        
        return df[mask].copy()


if __name__ == "__main__":
    # Test the preprocessor
    from loader import RaceDataLoader
    
    loader = RaceDataLoader(data_root='../../data/raw')
    races = loader.list_available_races()
    
    if races:
        track, race_num = races[0]
        print(f"Testing with {track} Race {race_num}...")
        
        race_data = loader.load_race(track, race_num)
        
        if 'sector_analysis' in race_data:
            print("\nCleaning sector analysis...")
            preprocessor = RaceDataPreprocessor()
            
            clean_df = preprocessor.clean_sector_analysis(race_data['sector_analysis'])
            print(f"  Rows: {len(clean_df)}")
            print(f"  Columns: {list(clean_df.columns)}")
            
            clean_df = preprocessor.calculate_tire_age(clean_df)
            print(f"  Tire age calculated: min={clean_df['tire_age'].min()}, max={clean_df['tire_age'].max()}")
            
            if 'weather' in race_data:
                clean_df = preprocessor.merge_weather_data(clean_df, race_data['weather'])
                print(f"  Weather data merged")
            
            clean_laps = preprocessor.filter_clean_laps(clean_df)
            print(f"  Clean racing laps: {len(clean_laps)} / {len(clean_df)}")

"""
Tire degradation modeling module
Predicts lap time increase based on tire age and conditions
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TireDegradationModel:
    """Model tire wear and predict lap time degradation"""
    
    def __init__(self, degree: int = 2):
        """
        Initialize tire degradation model
        
        Args:
            degree: Polynomial degree (2 = quadratic, 3 = cubic)
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self.is_trained = False
        self.baseline_lap_time = None
        self.feature_names = []
        self.metrics = {}
        
    def train(self, df: pd.DataFrame, features: Optional[list] = None) -> 'TireDegradationModel':
        """
        Train model on historical lap data
        
        Args:
            df: DataFrame with columns:
                - tire_age (required)
                - lap_time_seconds (required)
                - air_temp (optional)
                - track_temp (optional)
                - is_pit_lap (optional, for filtering)
                - is_caution (optional, for filtering)
        
        Returns:
            self (for method chaining)
        """
        # Validate required columns
        if 'tire_age' not in df.columns:
            raise ValueError("DataFrame must have 'tire_age' column")
        if 'lap_time_seconds' not in df.columns:
            raise ValueError("DataFrame must have 'lap_time_seconds' column")
        
        # Filter out pit laps and caution laps
        training_data = df[df['lap_time_seconds'].notna()].copy()
        
        if 'is_pit_lap' in training_data.columns:
            training_data = training_data[~training_data['is_pit_lap']]
        
        if 'is_caution' in training_data.columns:
            training_data = training_data[~training_data['is_caution']]
        
        # Determine features to use
        if features is None:
            features = ['tire_age']
            if 'air_temp' in training_data.columns:
                features.append('air_temp')
            if 'track_temp' in training_data.columns and training_data['track_temp'].mean() > 0:
                features.append('track_temp')
        
        self.feature_names = features
        
        # Prepare data
        X = training_data[features].values
        y = training_data['lap_time_seconds'].values
        
        # Remove any remaining NaN values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            raise ValueError(f"Insufficient training data: {len(X)} samples")
        
        # Polynomial features
        X_poly = self.poly.fit_transform(X)
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Store baseline (best lap time in training data)
        self.baseline_lap_time = y.min()
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'n_samples': len(X),
            'baseline_lap_time': self.baseline_lap_time
        }
        
        logger.info(f"Tire degradation model trained:")
        logger.info(f"  R² (train/test): {self.metrics['train_r2']:.3f} / {self.metrics['test_r2']:.3f}")
        logger.info(f"  RMSE (train/test): {self.metrics['train_rmse']:.3f}s / {self.metrics['test_rmse']:.3f}s")
        logger.info(f"  Baseline lap time: {self.baseline_lap_time:.3f}s")
        logger.info(f"  Training samples: {self.metrics['n_samples']}")
        
        return self
    
    def predict(self, tire_age: int, conditions: Optional[Dict] = None) -> Dict:
        """
        Predict lap time for given tire age and conditions
        
        Args:
            tire_age: Current tire age in laps
            conditions: Optional dict with 'air_temp', 'track_temp', etc.
        
        Returns:
            Dictionary with:
                - predicted_lap_time: Predicted lap time in seconds
                - degradation_seconds: Time loss vs baseline
                - baseline_lap_time: Best possible lap time
                - confidence: Confidence level ('high', 'medium', 'low')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare feature vector
        feature_values = [tire_age]
        
        if 'air_temp' in self.feature_names:
            if conditions and 'air_temp' in conditions:
                feature_values.append(conditions['air_temp'])
            else:
                raise ValueError("Model requires 'air_temp' in conditions")
        
        if 'track_temp' in self.feature_names:
            if conditions and 'track_temp' in conditions:
                feature_values.append(conditions['track_temp'])
            else:
                raise ValueError("Model requires 'track_temp' in conditions")
        
        X = np.array([feature_values])
        X_poly = self.poly.transform(X)
        prediction = self.model.predict(X_poly)[0]
        
        degradation = prediction - self.baseline_lap_time
        
        # Confidence based on tire age (lower confidence for extreme ages)
        if tire_age <= 25:
            confidence = 'high'
        elif tire_age <= 35:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'predicted_lap_time': prediction,
            'degradation_seconds': degradation,
            'baseline_lap_time': self.baseline_lap_time,
            'confidence': confidence
        }
    
    def find_optimal_pit_lap(self, 
                            current_tire_age: int,
                            conditions: Optional[Dict] = None,
                            max_tire_age: int = 30,
                            degradation_threshold: float = 2.0) -> Dict:
        """
        Find lap when tire degradation exceeds threshold
        
        Args:
            current_tire_age: Current tire age in laps
            conditions: Current weather conditions
            max_tire_age: Maximum safe tire age
            degradation_threshold: Seconds slower than baseline to trigger pit
        
        Returns:
            Dictionary with:
                - laps_until_pit: Recommended laps until pit stop
                - pit_at_tire_age: Tire age when pit should occur
                - degradation_at_pit: Expected degradation at pit time
                - reason: Explanation for recommendation
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Find when degradation exceeds threshold
        for tire_age in range(current_tire_age, max_tire_age + 1):
            pred = self.predict(tire_age, conditions)
            
            if pred['degradation_seconds'] >= degradation_threshold:
                laps_until = tire_age - current_tire_age
                
                return {
                    'laps_until_pit': laps_until,
                    'pit_at_tire_age': tire_age,
                    'degradation_at_pit': pred['degradation_seconds'],
                    'reason': f"Tire degradation reaches {degradation_threshold}s threshold"
                }
        
        # If threshold never exceeded, recommend at max tire age
        laps_until = max_tire_age - current_tire_age
        pred = self.predict(max_tire_age, conditions)
        
        return {
            'laps_until_pit': laps_until,
            'pit_at_tire_age': max_tire_age,
            'degradation_at_pit': pred['degradation_seconds'],
            'reason': f"Maximum safe tire age ({max_tire_age} laps) reached"
        }
    
    def predict_stint(self, 
                     start_tire_age: int,
                     stint_length: int,
                     conditions: Optional[Dict] = None) -> pd.DataFrame:
        """
        Predict lap times for an entire stint
        
        Args:
            start_tire_age: Starting tire age
            stint_length: Number of laps in stint
            conditions: Weather conditions
        
        Returns:
            DataFrame with predictions for each lap in stint
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        predictions = []
        
        for lap_offset in range(stint_length):
            tire_age = start_tire_age + lap_offset
            pred = self.predict(tire_age, conditions)
            
            predictions.append({
                'lap_in_stint': lap_offset + 1,
                'tire_age': tire_age,
                'predicted_lap_time': pred['predicted_lap_time'],
                'degradation_seconds': pred['degradation_seconds'],
                'confidence': pred['confidence']
            })
        
        return pd.DataFrame(predictions)
    
    def get_degradation_rate(self, tire_age: int, conditions: Optional[Dict] = None) -> float:
        """
        Calculate degradation rate (seconds per lap) at given tire age
        
        Args:
            tire_age: Current tire age
            conditions: Weather conditions
        
        Returns:
            Degradation rate in seconds per lap
        """
        if not self.is_trained or tire_age <= 0:
            return 0.0
        
        pred_current = self.predict(tire_age, conditions)
        pred_next = self.predict(tire_age + 1, conditions)
        
        rate = pred_next['predicted_lap_time'] - pred_current['predicted_lap_time']
        
        return max(0.0, rate)  # Can't be negative
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        import pickle
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'poly': self.poly,
            'model': self.model,
            'degree': self.degree,
            'baseline_lap_time': self.baseline_lap_time,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TireDegradationModel':
        """Load trained model from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(degree=model_data['degree'])
        instance.poly = model_data['poly']
        instance.model = model_data['model']
        instance.baseline_lap_time = model_data['baseline_lap_time']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance


if __name__ == "__main__":
    # Test the tire degradation model
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from data.loader import RaceDataLoader
    from data.preprocessor import RaceDataPreprocessor
    from data.feature_engineering import FeatureEngineer
    
    loader = RaceDataLoader(data_root='../../data/raw')
    races = loader.list_available_races()
    
    if races:
        track, race_num = races[0]
        print(f"Testing tire degradation model with {track} Race {race_num}...")
        
        # Load and preprocess data
        race_data = loader.load_race(track, race_num)
        
        if 'sector_analysis' in race_data:
            preprocessor = RaceDataPreprocessor()
            
            df = preprocessor.clean_sector_analysis(race_data['sector_analysis'])
            df = preprocessor.calculate_tire_age(df)
            
            if 'weather' in race_data:
                df = preprocessor.merge_weather_data(df, race_data['weather'])
            
            # Train model
            print("\nTraining tire degradation model...")
            model = TireDegradationModel(degree=2)
            model.train(df)
            
            # Test predictions
            print("\nTesting predictions:")
            conditions = {
                'air_temp': df['air_temp'].mean() if 'air_temp' in df.columns else None,
                'track_temp': df['track_temp'].mean() if 'track_temp' in df.columns else None
            }
            
            for tire_age in [1, 5, 10, 15, 20, 25]:
                pred = model.predict(tire_age, conditions)
                print(f"  Tire age {tire_age:2d}: {pred['predicted_lap_time']:.3f}s "
                      f"(+{pred['degradation_seconds']:.3f}s, {pred['confidence']})")
            
            # Find optimal pit lap
            print("\nOptimal pit stop analysis:")
            current_age = 10
            pit_rec = model.find_optimal_pit_lap(current_age, conditions, degradation_threshold=2.0)
            print(f"  Current tire age: {current_age} laps")
            print(f"  Pit in: {pit_rec['laps_until_pit']} laps")
            print(f"  Reason: {pit_rec['reason']}")
            
            # Save model
            model.save_model('../../models/tire_degradation.pkl')
            print("\n✓ Model saved successfully")

"""
Data loading module for GR Cup race data
Handles loading CSV files with auto-detection of delimiters
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceDataLoader:
    """Load and validate race data from CSV files"""
    
    TRACKS = [
        'barber', 
        'COTA', 
        'indianapolis', 
        'Road America', 
        'Sebring', 
        'Sonoma', 
        'virginia-international-raceway'
    ]
    
    def __init__(self, data_root: str):
        """
        Initialize data loader
        
        Args:
            data_root: Path to root data directory containing track folders
        """
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root not found: {data_root}")
        
    def load_race(self, track: str, race_num: int) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a specific race
        
        Args:
            track: Track name (e.g., 'barber', 'COTA')
            race_num: Race number (1 or 2)
            
        Returns:
            Dictionary containing all race data:
            - results: Race results
            - sector_analysis: Lap-by-lap sector times
            - weather: Weather conditions
            - lap_times: Lap timing data
            - lap_start: Lap start timestamps
            - lap_end: Lap end timestamps
        """
        track_path = self.data_root / track
        
        if not track_path.exists():
            raise ValueError(f"Track data not found: {track}")
        
        race_data = {}
        
        # Load results
        try:
            race_data['results'] = self._load_results(track_path, race_num)
            logger.info(f"✓ Loaded results: {len(race_data['results'])} entries")
        except Exception as e:
            logger.warning(f"Could not load results: {e}")
        
        # Load sector analysis
        try:
            race_data['sector_analysis'] = self._load_sector_analysis(track_path, race_num)
            logger.info(f"✓ Loaded sector analysis: {len(race_data['sector_analysis'])} laps")
        except Exception as e:
            logger.warning(f"Could not load sector analysis: {e}")
        
        # Load weather
        try:
            race_data['weather'] = self._load_weather(track_path, race_num)
            logger.info(f"✓ Loaded weather: {len(race_data['weather'])} observations")
        except Exception as e:
            logger.warning(f"Could not load weather: {e}")
        
        # Load lap times
        try:
            race_data['lap_times'] = self._load_lap_times(track_path, race_num)
            logger.info(f"✓ Loaded lap times: {len(race_data['lap_times'])} laps")
        except Exception as e:
            logger.warning(f"Could not load lap times: {e}")
            # Fallback: extract from sector analysis if available
            if 'sector_analysis' in race_data and 'LAP_TIME' in race_data['sector_analysis'].columns:
                logger.info("Using lap times from sector analysis")
                race_data['lap_times'] = race_data['sector_analysis'][
                    ['NUMBER', 'LAP_NUMBER', 'LAP_TIME']
                ].copy()
        
        # Load lap start/end
        try:
            race_data['lap_start'] = self._load_lap_start(track_path, race_num)
            race_data['lap_end'] = self._load_lap_end(track_path, race_num)
            logger.info(f"✓ Loaded lap boundaries")
        except Exception as e:
            logger.warning(f"Could not load lap boundaries: {e}")
        
        return race_data
    
    def _try_load_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Try loading CSV with different delimiters
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame if successful, None otherwise
        """
        for sep in [';', ',', '\t']:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
                # Valid parse should have multiple columns
                if len(df.columns) > 3:
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    return df
            except Exception as e:
                continue
        
        # Try with default pandas behavior
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            df.columns = df.columns.str.strip()
            return df
        except:
            return None
    
    def _load_results(self, track_path: Path, race_num: int) -> pd.DataFrame:
        """Load race results file"""
        patterns = [
            f"03_*Results*Race {race_num}*.CSV",
            f"03_*Results*Race_{race_num}*.CSV",
            f"03_*Race {race_num}*.CSV",
        ]
        
        for pattern in patterns:
            files = list(track_path.glob(pattern))
            if files:
                df = self._try_load_csv(files[0])
                if df is not None:
                    return df
        
        # Check subdirectories for COTA, Road America, etc.
        for race_dir in ['Race 1', 'Race 2']:
            race_dir_path = track_path / race_dir
            if race_dir_path.exists() and int(race_dir.split()[-1]) == race_num:
                for pattern in patterns:
                    files = list(race_dir_path.glob(pattern))
                    if files:
                        df = self._try_load_csv(files[0])
                        if df is not None:
                            return df
        
        raise FileNotFoundError(f"Results file not found for {track_path.name} race {race_num}")
    
    def _load_sector_analysis(self, track_path: Path, race_num: int) -> pd.DataFrame:
        """Load sector analysis file with timing splits"""
        patterns = [
            f"23_*Race {race_num}*.CSV",
            f"23_*Race_{race_num}*.CSV",
        ]
        
        for pattern in patterns:
            files = list(track_path.glob(pattern))
            if files:
                df = self._try_load_csv(files[0])
                if df is not None:
                    return df
        
        # Check subdirectories
        for race_dir in ['Race 1', 'Race 2']:
            race_dir_path = track_path / race_dir
            if race_dir_path.exists() and int(race_dir.split()[-1]) == race_num:
                for pattern in patterns:
                    files = list(race_dir_path.glob(pattern))
                    if files:
                        df = self._try_load_csv(files[0])
                        if df is not None:
                            return df
        
        raise FileNotFoundError(f"Sector analysis not found for {track_path.name} race {race_num}")
    
    def _load_weather(self, track_path: Path, race_num: int) -> pd.DataFrame:
        """Load weather data"""
        patterns = [
            f"26_*Race {race_num}*.CSV",
            f"26_*Race_{race_num}*.CSV",
        ]
        
        for pattern in patterns:
            files = list(track_path.glob(pattern))
            if files:
                df = self._try_load_csv(files[0])
                if df is not None:
                    # Convert timestamp if present
                    if 'TIME_UTC_STR' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['TIME_UTC_STR'], errors='coerce')
                    return df
        
        # Check subdirectories
        for race_dir in ['Race 1', 'Race 2']:
            race_dir_path = track_path / race_dir
            if race_dir_path.exists() and int(race_dir.split()[-1]) == race_num:
                for pattern in patterns:
                    files = list(race_dir_path.glob(pattern))
                    if files:
                        df = self._try_load_csv(files[0])
                        if df is not None:
                            if 'TIME_UTC_STR' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['TIME_UTC_STR'], errors='coerce')
                            return df
        
        raise FileNotFoundError(f"Weather data not found for {track_path.name} race {race_num}")
    
    def _load_lap_times(self, track_path: Path, race_num: int) -> pd.DataFrame:
        """Load lap timing data"""
        track_name = track_path.name.lower().replace(' ', '_')
        patterns = [
            f"R{race_num}_{track_name}_lap_time.csv",
            f"R{race_num}_*_lap_time.csv",
            f"{track_name}_lap_time_R{race_num}.csv",
            f"*_lap_time_R{race_num}.csv",
        ]
        
        for pattern in patterns:
            files = list(track_path.glob(pattern))
            if files:
                df = pd.read_csv(files[0])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                return df
        
        raise FileNotFoundError(f"Lap times not found for {track_path.name} race {race_num}")
    
    def _load_lap_start(self, track_path: Path, race_num: int) -> pd.DataFrame:
        """Load lap start timestamps"""
        track_name = track_path.name.lower().replace(' ', '_')
        patterns = [
            f"R{race_num}_{track_name}_lap_start.csv",
            f"R{race_num}_*_lap_start.csv",
            f"{track_name}_lap_start_time_R{race_num}.csv",
            f"*_lap_start_time_R{race_num}.csv",
        ]
        
        for pattern in patterns:
            files = list(track_path.glob(pattern))
            if files:
                df = pd.read_csv(files[0])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                return df
        
        raise FileNotFoundError(f"Lap start data not found for {track_path.name} race {race_num}")
    
    def _load_lap_end(self, track_path: Path, race_num: int) -> pd.DataFrame:
        """Load lap end timestamps"""
        track_name = track_path.name.lower().replace(' ', '_')
        patterns = [
            f"R{race_num}_{track_name}_lap_end.csv",
            f"R{race_num}_*_lap_end.csv",
            f"{track_name}_lap_end_time_R{race_num}.csv",
            f"*_lap_end_time_R{race_num}.csv",
        ]
        
        for pattern in patterns:
            files = list(track_path.glob(pattern))
            if files:
                df = pd.read_csv(files[0])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                return df
        
        raise FileNotFoundError(f"Lap end data not found for {track_path.name} race {race_num}")
    
    def list_available_races(self) -> List[Tuple[str, int]]:
        """
        List all available race data
        
        Returns:
            List of (track, race_num) tuples
        """
        available = []
        
        for track in self.TRACKS:
            track_path = self.data_root / track
            if not track_path.exists():
                continue
            
            for race_num in [1, 2]:
                try:
                    # Check if we can load sector analysis (most reliable indicator)
                    self._load_sector_analysis(track_path, race_num)
                    available.append((track, race_num))
                except:
                    pass
        
        return available


if __name__ == "__main__":
    # Test the loader
    loader = RaceDataLoader(data_root='../../data/raw')
    
    print("Available races:")
    races = loader.list_available_races()
    for track, race_num in races:
        print(f"  - {track}: Race {race_num}")
    
    if races:
        # Load first available race
        track, race_num = races[0]
        print(f"\nLoading {track} Race {race_num}...")
        race_data = loader.load_race(track, race_num)
        
        print("\nData loaded:")
        for key, df in race_data.items():
            print(f"  {key}: {len(df)} rows, {len(df.columns)} columns")

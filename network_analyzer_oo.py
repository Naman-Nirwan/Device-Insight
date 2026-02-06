import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import json
import os
from enum import Enum
from typing import List, Dict, Tuple, Optional, Type
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine

"""Enum for connection states."""
class ConnectionState(Enum):
    LOGGED_IN = 1
    LOGGED_OUT = 0
    IDLE = 2


class APIClient:
    """Handles all API communication."""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            api_url: Base API URL (defaults to environment variable API_URL)
            api_key: API key for authentication (defaults to environment variable API_KEY)
        """
        self.base_url = api_url or os.getenv('API_URL')
        self.api_key = (api_key or os.getenv('API_KEY')).strip() if (api_key or os.getenv('API_KEY')) else None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        return headers
    
    def fetch_devices(self) -> List[Dict]:
        """
        Fetch device list from API.
        
        Returns:
            List of device dictionaries
            
        Raises:
            requests.RequestException: If API request fails
        """
        if not self.base_url:
            raise ValueError("API URL not configured")
        
        url = f"{self.base_url}devices"
        response = requests.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.json()['data']
    
    def fetch_metrics(self, device_ids: List[str], 
                     start_time: int, end_time: int, resolution: int) -> Dict:
        """
        Fetch metrics data from API.
        
        Args:
            device_ids: List of device IDs to fetch data for
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            API response with metrics data
            
        Raises:
            requests.RequestException: If API request fails
        """
        if not self.base_url:
            raise ValueError("API URL not configured")
        url = f"{self.base_url}devices/metrics"
        payload = {
            "interval": {
                "startTime": start_time,
                "endTime": end_time
            },
            "ids": device_ids,
            "metrics": ["ue_connection_state"],
            "resolution": resolution
        }
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        return response.json()['data'][0]  # only one metric requested ue_connection_state 


class DataProcessor:
    """Handles data processing and transformation."""

    def __init__(self, period: int = 60):
        """
        Initialize data processor.
        
        Args:
            period: Time period in minutes for aggregation
        """
        self.period = period
        self.api_client = None
    
    def process_raw_data(self,raw_data: List[Dict]) -> pd.DataFrame:
        """
        Load raw API data into pandas DataFrame and extract time features.
        
        Args:
            raw_data: List of device data with dataPoints
            
        Returns:
            DataFrame with processed data including hour and day_of_week
        """
        dataframes = []
        for device in raw_data:
            datapoints = device.get('dataPoints', [])
            if datapoints is None or len(datapoints) == 0:
                continue
            df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(point[0]),
                'user_id': device['id'],
                'connection_state': point[1]
            } for point in datapoints])
            # Extract time features
            df['hour'] = df['timestamp'].dt.hour
            df['miniute'] = df['timestamp'].dt.minute
            df['second'] = df['timestamp'].dt.second
            df['period'] = (df['hour'] * 60 + df['miniute']) // self.period
            df['day_of_week'] = df['timestamp'].dt.weekday
            df['date'] = df['timestamp'].dt.date
            dataframes.append(df)
        
        if not dataframes:
            raise ValueError("No data to process")
        
        return pd.concat(dataframes, ignore_index=True)
    
    def data_from_api(self, api_url:str=None, api_key:str=None, start_time:int=None, end_time:int=None, resolution:int=600) -> pd.DataFrame:
        """Fetch and process data from API."""
        self.api_client = APIClient(api_url, api_key)
        devices = self.api_client.fetch_devices()
        device_ids = [device['id'] for device in devices]
        metrics_response = self.api_client.fetch_metrics(device_ids, start_time, end_time,resolution=resolution)
        raw_data = metrics_response['metricData']
        return self.process_raw_data(raw_data)
    
    def save_to_csv(df: pd.DataFrame, filepath: str = 'data.csv') -> None:
        """Save processed data to CSV file."""
        with open(filepath, 'w') as f:
            df.to_csv(f)


class ProbabilityAnalyzer:
    """Analyzes connection state probabilities."""
    
    def __init__(self, df: pd.DataFrame, period: int = 60):
        """
        Initialize analyzer with data.
        
        Args:
            df: Processed DataFrame with connection state data
        """
        self.df = df
        self.period = period # in minutes
        self.probabilities, self.counts = {}, {}
    
    def _compute_probabilities(self) -> Dict[Tuple, Dict[int, float]]:
        """
        Compute conditional probabilities P(state|user,hour).
        
        Returns:
            Dictionary mapping (user_id, hour) to state probabilities
        """
        probabilities = {}
        counts = {}
        
        # Group by user and hour
        for (user, period, week_day), group in self.df.groupby(['user_id', 'period', 'day_of_week']):
            state_counts = group['connection_state'].value_counts()
            total_counts = len(group)
            # Calculate probability for each state
            state_probs = {}
            for state in [ConnectionState.LOGGED_IN.value, ConnectionState.LOGGED_OUT.value, ConnectionState.IDLE.value]:
                state_probs[state] = state_counts.get(state, 0) / total_counts
            
            probabilities[(user, period, week_day)] = state_probs
            counts[(user, period, week_day)] = state_counts.to_dict()
            
        return probabilities, counts
    
    def predict_state(self, user_id: str, timestamp: int) -> Tuple[Optional[int], float]:
        """
        Predict the most likely connection state for a user at a given hour.
        
        Args:
            user_id: User identifier
            timestamp: Timestamp in seconds
            
        Returns:
            Tuple of (most_likely_state, probability) or (None, 0.0) if not found
        """
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.weekday()
        period = (hour * 60 + minute) // self.period
        key = (user_id, period, day_of_week) 
        
        if key not in self.probabilities:
            return None, 0.0
        
        state_probs = self.probabilities[key]
        most_likely_state = max(state_probs, key=state_probs.get)
        
        return most_likely_state, state_probs[most_likely_state]
    
    def update_probabilities(self, df: pd.DataFrame) -> None:
        """Update probabilities with new data."""
        self.df = df
        probabilities, counts = self._compute_probabilities()
        for key, state_probs in probabilities.items():
            
            if key in self.probabilities:
                # Weighted average based on counts
                existing_count = sum(self.counts[key].values())
                new_count = sum(counts[key].values())
                total_count = existing_count + new_count
                for state, prob in state_probs.items():
                    updated_prob = (self.counts[key].get(state, 0) + counts[key].get(state, 0)) / total_count
                    self.probabilities[key][state] = updated_prob
                    self.counts[key][state] = self.counts[key].get(state, 0) + counts[key].get(state, 0)
            else:
                self.probabilities[key] = state_probs
                self.counts[key] = counts[key]

    def get_probabilities(self) -> Dict[Tuple, Dict[int, float]]:
        """Get all computed probabilities."""
        return self.probabilities


class Visualizer:
    """Handles visualization of connection state patterns."""
    
    def __init__(self, df: pd.DataFrame, probabilities: Dict[Tuple, Dict[int, float]]):
        """
        Initialize visualizer.
        
        Args:
            df: Processed DataFrame
            probabilities: Dictionary of conditional probabilities
        """
        self.df = df
        self.probabilities = probabilities
    
    def plot_heatmaps(self, output_file: str = 'connection_state_heatmaps.png') -> None:
        """
        Plot heatmaps showing connection state patterns.
        
        Args:
            output_file: Path to save the heatmap image
        """
        users = self.df['user_id'].unique()
        
        # Create a figure with subplots for each user
        fig, axes = plt.subplots(len(users), 1, figsize=(12, 4 * len(users)))
        
        if len(users) == 1:
            axes = [axes]
        
        for idx, user in enumerate(users):
            # Create a matrix for heatmap: hours x states
            hours = range(24)
            states = [ConnectionState.LOGGED_IN, ConnectionState.LOGGED_OUT, ConnectionState.IDLE]
            state_labels = [ConnectionState.get_label(s) for s in states]
            
            heatmap_data = []
            for state in states:
                row = []
                for hour in hours:
                    key = (user, hour)
                    if key in self.probabilities:
                        row.append(self.probabilities[key].get(state, 0))
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            # Plot heatmap
            im = axes[idx].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            axes[idx].set_title(f'Connection State Probabilities for {user}')
            axes[idx].set_xlabel('Hour of Day')
            axes[idx].set_ylabel('Connection State')
            axes[idx].set_yticks(range(len(states)))
            axes[idx].set_yticklabels(state_labels)
            axes[idx].set_xticks(range(0, 24, 2))
            axes[idx].set_xticklabels(range(0, 24, 2))
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], label='Probability')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Heatmaps saved to '{output_file}'")
        plt.show()


class NetworkAnalyzer:
    """Main orchestrator for network connection state analysis."""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, period: int = 60):
        """
        Initialize the analyzer.
        
        Args:
            api_url: API base URL
            api_key: API authentication key
        """
        self.api_client = APIClient(api_url, api_key)
        self.data = None
        self.analyzer = None
        self.visualizer = None
        self.processor = None
        self.period = period  # in minutes

    def initial_train(self, start_time:int=None, end_time:int=None, resolution:int=600) -> Dict[Tuple, Dict[int, float]]:
        """
        Initially train the probability model.

        Args:
            start_time: Start timestamp for data fetching
            end_time: End timestamp for data fetching
        """

        print("Past Data Analysis for Network Connection State")
        print("=" * 50)

        if start_time is None:
            AssertionError("start_time must be provided for initial training")
        if end_time is None:
            AssertionError("end_time must be provided for initial training")

        devices = self.api_client.fetch_devices()
        device_ids = [device['id'] for device in devices]
        print(f"   Found {len(device_ids)} devices")

        # Fetch metrics
        print("\n1. Fetching metrics...")
        metrics_response = self.api_client.fetch_metrics(device_ids, start_time, end_time,resolution=resolution)
        raw_data = metrics_response['metricData']
        print(f"   Fetched {len(raw_data)} records")
        self.processor = DataProcessor(period=self.period)
        self.data = self.processor.process_raw_data(raw_data)
        # self.processor.save_to_csv(self.data)
        print(f"   Processed data shape: {self.data.shape}")
        print(f"   Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")

        print("\n4. Computing probabilities P(state|user,period,weekday)...")
        self.analyzer = ProbabilityAnalyzer(self.data,period=self.period)
        self.analyzer.probabilities, self.analyzer.counts = self.analyzer._compute_probabilities()
        self.save_probabilities()
        return self.analyzer.probabilities
    
    def update(self, start_time: int = None, end_time: int = None, resolution: int = 600)-> None:
        """
        Update the model with new data.

        Args:
            start_time: Start timestamp for data fetching
            end_time: End timestamp for data fetching
        """
        if start_time is None:
            AssertionError("start_time must be provided for initial training")
        if end_time is None:
            AssertionError("end_time must be provided for initial training")

        devices = self.api_client.fetch_devices()
        device_ids = [device['id'] for device in devices]
        print(f"   Found {len(device_ids)} devices")

        # Fetch metrics
        print("\n1. Fetching metrics...")
        metrics_response = self.api_client.fetch_metrics(device_ids, start_time, end_time,resolution=resolution)
        raw_data = metrics_response['metricData']
        print(f"   Fetched {len(raw_data)} records")
        self.data = self.processor.process_raw_data(raw_data)

        self.analyzer.update_probabilities(self.data)
        self.save_probabilities()

    def predict_user_state(self, user_id: str, timestamp: int) -> Tuple[Optional[str], float]:
        """
        Predict connection state for a specific user and hour.
        
        Args:
            user_id: User identifier
            hour: Hour of day (0-23)
            
        Returns:
            Tuple of (state_label, probability)
        """
        if self.analyzer is None:
            raise RuntimeError("Analysis not run yet. Call run() first.")
        
        state, prob = self.analyzer.predict_state(user_id, timestamp)
        state_label = ConnectionState.get_label(state) if state is not None else None
        return state_label, prob
    
    def evaluate(self, test_data: pd.DataFrame) -> float:
        """
        Evaluate prediction accuracy on test data.
        
        Args:
            test_data: DataFrame with test data
            
        Returns:
            Accuracy as a float
        """
        if self.analyzer is None:
            raise RuntimeError("Analysis not run yet. Call run() first.")
        
        correct_predictions = 0
        total_predictions = 0
        
        for _, row in test_data.iterrows():
            predicted_state, _ = self.analyzer.predict_state(row['user_id'], int(row['timestamp'].timestamp()))
            if predicted_state and (int(predicted_state) == int(row['connection_state'])):
                correct_predictions += 1
            total_predictions += 1
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy

    def train_recursive(self, start_time:int, end_time:int, interval_days:int=5, resolution:int=600) -> None:
        """
        Train the model recursively over intervals.

        Args:
            start_time: Start timestamp for data fetching
            end_time: End timestamp for data fetching
            interval_days: Number of days per training interval
        """
        current_start = start_time
        interval_seconds = interval_days * 24 * 3600
        current_end = current_start + interval_seconds
        self.initial_train(current_start, min(current_end, end_time), resolution=resolution)
        current_start = current_end
        while current_start < end_time:
            current_end = min(current_start + interval_seconds, end_time)
            print(f"\nTraining from {datetime.fromtimestamp(current_start)} to {datetime.fromtimestamp(current_end)}")
            self.update(current_start, current_end, resolution=resolution)
            current_start = current_end

    def save_probabilities(self, prob_path: str = 'probabilities.json', count_path: str = 'counts.json') -> None:
        """Save computed probabilities to a JSON file."""
        if self.analyzer is None:
            raise RuntimeError("Analysis not run yet. Train first.")
        
        probabilities = self.analyzer.probabilities
        counts = self.analyzer.counts
        with open(prob_path,'w') as f:
            json.dump({str(k): v for k, v in probabilities.items()}, f, indent=4)
        with open(count_path,'w') as f:
            json.dump({str(k): v for k, v in counts.items()}, f, indent=4)
        print(f"Probabilities saved to '{prob_path}' and counts saved to '{count_path}'")

    def load_probabilities(self, prob_path: str = 'probabilities.json', count_path: str = 'counts.json') -> None:
        """Load probabilities from a JSON file."""
        if not os.path.exists(prob_path):
            raise FileNotFoundError(f"File '{prob_path}' not found.")
        
        with open(prob_path, 'r') as f:
            prob_data = json.load(f)
        with open(count_path, 'r') as f:
            count_data = json.load(f)

        probabilities = {tuple(eval(k)): v for k, v in prob_data.items()}
        counts = {tuple(eval(k)): v for k, v in count_data.items()}
        
        self.analyzer = ProbabilityAnalyzer(pd.DataFrame(), period=self.period)
        self.analyzer.probabilities = probabilities
        self.analyzer.counts = counts

        return
    
class Classifier:
    def __init__(self):
        self._features={} #private attribute to store features
        self.classifications = {}

    @staticmethod
    def entropy(p):
        p = p + 1e-9
        return -np.sum(p * np.log(p))
    
    @staticmethod
    def build_user_tensor(prob_dict):
        """
        Returns:
        user_profiles[user] = array shape (7,96,3)
        """
        user_profiles = defaultdict(lambda: np.zeros((7,96,3)))

        for (user, block, day), probs in prob_dict.items():
            user_profiles[user][day, block, :] = [probs.get(1,0), probs.get(0,0), probs.get(2,0)]

        return user_profiles
    
    def extract_weekly_features(self,user,profile):
        """
        profile shape: (7,96,3)
        """
        feats = {}
        login = profile[:,:,0]   # p_logged_in
        idle  = profile[:,:,2]

        total_login_activity = np.sum(login)
        feats["total_login_activity"] = total_login_activity

        # If device never logs in, stop here
        if total_login_activity < 1e-3:
            feats["inactive"] = True
            self._features[user] = feats
            return feats

        feats["inactive"] = False
        # --- Temporal entropy per day ---
        day_entropies = [self.entropy(login[d]) for d in range(7)]

        feats["login_entropy_mean"] = np.mean(day_entropies)
        feats["login_entropy_var"]  = np.var(day_entropies)

        # --- Weekday consistency ---
        sims = []
        for d1 in range(7):
            for d2 in range(d1+1, 7):
                sims.append(1 - cosine(login[d1], login[d2]))

        feats["weekday_similarity"] = np.mean(sims)

        # --- Peak strength ---
        feats["login_peak_max"] = np.max(login)

        # --- Active days count ---
        feats["active_days"] = np.sum(np.max(login, axis=1) > 0.05)

        # --- Idle dominance ---
        feats["idle_mean"] = np.mean(idle)

        self._features[user] = feats

        return feats

    def classify(self, probabilities: Dict[Tuple, Dict[int, float]]) -> None:
        """
        On the basis of the past data and probabilities classify the users into different categories:
            1. Fixed Wireless Devices
            2. Shift Connected Devices
            3. Ad-hoc Connected Devices (Random Connections)
        """

        user_profiles = self.build_user_tensor(probabilities)
        for user, profile in user_profiles.items():
            feats = self.extract_weekly_features(user, profile)
            if not feats["inactive"] and feats["weekday_similarity"] > 0.85 and feats["login_entropy_mean"] < 2.5:
                classification = "Fixed Wireless Device"
            elif not feats["inactive"] and feats["weekday_similarity"] > 0.6 and feats["active_days"] >= 4:
                classification = "Shift Connected Device"
            else:
                classification = "Ad-hoc Connected Device"
            
            self.classifications[user] = classification
        
        
        

def main():
    """Main entry point."""
    analyzer = NetworkAnalyzer(period=15)

    # 10 months of data for initial training
    # 2 months of data for validation accuracy
    train_start_time = int(datetime(2025, 11, 1).timestamp())
    train_end_time = int(datetime(2025, 11, 30).timestamp())
    val_start_time = int(datetime(2025, 11, 30).timestamp())
    val_end_time = int(datetime(2025, 12, 31).timestamp())

    ValDataProcesser = DataProcessor(period=15)
    val_dataframe = ValDataProcesser.data_from_api(
        start_time=val_start_time,
        end_time=val_end_time
    )
    probability = analyzer.initial_train(train_start_time, train_end_time)
    val_accuracy = analyzer.evaluate(val_dataframe)
    print(f"\nValidation Accuracy: {val_accuracy:.2%}")



if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Network Connection State Analyzer

Analyzes user network connection states from API data.
Connection states: 0 = logged in, 1 = logged out, 2 = idle
"""

import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import json
import os

def fetch_devices(api_url=None):
    """
    Fetch device list from API.
    
    Args:
        api_url: URL to fetch device list from
        
    Returns:
        List of device dictionaries
    """
    api_url = api_url if api_url else (os.getenv('API_URL')+"/devices") if os.getenv('API_URL') else None
    headers = {'Content-Type': 'application/json'}
    api_key = os.getenv('API_KEY').strip() if os.getenv('API_KEY') else None
    if api_key:
        headers['Authorization'] = f"Bearer {api_key}"
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    return response.json()

def fetch_data(api_url=None, device_ids=None):
    """
    Fetch data from API or use sample data if no URL provided.
    
    Args:
        api_url: Optional URL to fetch data from
        
    Returns:
        List of dictionaries with timestamp, user_id, and connection_state
    """
    api_url = api_url or (os.getenv('API_URL')+"/devices/metrics") if os.getenv('API_URL') else None
    device_ids = device_ids if device_ids else []
    if api_url:
        payload = {"interval": 
                        {"startTime": 1768931759, "endTime": 1768931760},
                    "ids": device_ids,
                    "metrics": ["ue_connection_state"]}
        print(os.getenv('API_KEY'))
        headers = {'Content-Type': 'application/json'}
        api_key = os.getenv('API_KEY').strip() if os.getenv('API_KEY') else None
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    else:
        
        return AssertionError("No API URL provided and no sample data available.")


def load_and_process_data(data):
    """
    Load data into pandas DataFrame and extract time features.
    
    Args:
        data: List of dictionaries with timestamp, user_id, connection_state
        
    Returns:
        DataFrame with processed data including hour and day_of_week
    """
    dataframes = []
    for device in data:
        datapoints = device.get('dataPoints', [])
        df = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(point[0]),
            'user_id': device['id'],
            'connection_state': point[1]
        } for point in datapoints])

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['date'] = df['timestamp'].dt.date
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

def compute_probabilities(df):
    """
    Compute conditional probabilities P(state|user,hour).
    
    Args:
        df: DataFrame with processed data
        
    Returns:
        Dictionary mapping (user_id, hour) to state probabilities
    """
    probabilities = {}
    
    # Group by user and hour
    for (user, hour, week_day), group in df.groupby(['user_id', 'hour', 'day_of_week']):
        state_counts = group['connection_state'].value_counts()
        total_counts = len(group)
        
        # Calculate probability for each state
        state_probs = {}
        for state in [0, 1, 2]:
            state_probs[state] = state_counts.get(state, 0) / total_counts
        
        probabilities[(user, hour)] = state_probs
    
    return probabilities


def predict_state(probabilities, user_id, hour):
    """
    Predict the most likely connection state for a user at a given hour.
    
    Args:
        probabilities: Dictionary of conditional probabilities
        user_id: User identifier
        hour: Hour of day (0-23)
        
    Returns:
        Most likely state and its probability
    """
    key = (user_id, hour)
    
    if key not in probabilities:
        return None, 0.0
    
    state_probs = probabilities[key]
    most_likely_state = max(state_probs, key=state_probs.get)
    
    return most_likely_state, state_probs[most_likely_state]


def plot_heatmaps(df, probabilities):
    """
    Plot heatmaps showing connection state patterns.
    
    Args:
        df: DataFrame with processed data
        probabilities: Dictionary of conditional probabilities
    """
    users = df['user_id'].unique()
    
    # Create a figure with subplots for each user
    fig, axes = plt.subplots(len(users), 1, figsize=(12, 4 * len(users)))
    
    if len(users) == 1:
        axes = [axes]
    
    for idx, user in enumerate(users):
        # Create a matrix for heatmap: hours x states
        hours = range(24)
        states = [0, 1, 2]
        state_labels = ['Logged In', 'Logged Out', 'Idle']
        
        heatmap_data = []
        for state in states:
            row = []
            for hour in hours:
                key = (user, hour)
                if key in probabilities:
                    row.append(probabilities[key].get(state, 0))
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
    plt.savefig('connection_state_heatmaps.png', dpi=150, bbox_inches='tight')
    print("Heatmaps saved to 'connection_state_heatmaps.png'")
    plt.show()


def main():
    """Main function to run the network connection state analysis."""
    print("Network Connection State Analyzer")
    print("=" * 50)
    devices = fetch_devices()['data']
    device_ids = []
    for device in devices:
        device_ids.append(device['id'])
    # Fetch data (using sample data by default)
    print("\n1. Fetching data...")
    data = fetch_data(device_ids=device_ids)['data'][0]['metricData']  #only ue_connection_state metric is fetched
    print(f"Fetched {len(data)} records")
    # Load and process data
    print("\n2. Processing data...")
    dataframes = load_and_process_data(data)
    with open('data.csv', 'w') as f:
        dataframes.to_csv(f)
        
    print(f"   Processed data shape: {dataframes.shape}")
    print(f"   Date range: {dataframes['timestamp'].min()} to {dataframes['timestamp'].max()}")
    
    # Compute probabilities
    print("\n3. Computing probabilities P(state|user,hour,day_of_week)...")
    probabilities = compute_probabilities(dataframes)
    print(f"   Computed probabilities for {len(probabilities)} (user, hour, day_of_week) combinations")
    
    # Make some predictions
    print("\n4. Sample predictions:")
    for user in dataframes['user_id'].unique()[:3]:
        for hour in [8, 12, 17]:
            state, prob = predict_state(probabilities, user, hour)
            if state is not None:
                state_name = {0: 'Logged In', 1: 'Logged Out', 2: 'Idle'}[state]
                print(f"   {user} at hour {hour}: {state_name} (probability: {prob:.2f})")
    
    # Plot heatmaps
    print("\n5. Generating heatmaps...")
    plot_heatmaps(dataframes, probabilities)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")


if __name__ == "__main__":
    main()

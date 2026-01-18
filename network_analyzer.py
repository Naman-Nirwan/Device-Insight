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


def fetch_data(api_url=None):
    """
    Fetch data from API or use sample data if no URL provided.
    
    Args:
        api_url: Optional URL to fetch data from
        
    Returns:
        List of dictionaries with timestamp, user_id, and connection_state
    """
    if api_url:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    else:
        # Sample data for demonstration
        sample_data = [
            {"timestamp": "2026-01-18 08:00:00", "user_id": "user1", "connection_state": 0},
            {"timestamp": "2026-01-18 09:00:00", "user_id": "user1", "connection_state": 0},
            {"timestamp": "2026-01-18 12:00:00", "user_id": "user1", "connection_state": 2},
            {"timestamp": "2026-01-18 17:00:00", "user_id": "user1", "connection_state": 1},
            {"timestamp": "2026-01-18 08:00:00", "user_id": "user2", "connection_state": 0},
            {"timestamp": "2026-01-18 09:00:00", "user_id": "user2", "connection_state": 0},
            {"timestamp": "2026-01-18 12:00:00", "user_id": "user2", "connection_state": 0},
            {"timestamp": "2026-01-18 17:00:00", "user_id": "user2", "connection_state": 1},
            {"timestamp": "2026-01-19 08:00:00", "user_id": "user1", "connection_state": 0},
            {"timestamp": "2026-01-19 09:00:00", "user_id": "user1", "connection_state": 0},
            {"timestamp": "2026-01-19 12:00:00", "user_id": "user1", "connection_state": 2},
            {"timestamp": "2026-01-19 17:00:00", "user_id": "user1", "connection_state": 1},
            {"timestamp": "2026-01-19 08:00:00", "user_id": "user2", "connection_state": 0},
            {"timestamp": "2026-01-19 09:00:00", "user_id": "user2", "connection_state": 0},
            {"timestamp": "2026-01-19 12:00:00", "user_id": "user2", "connection_state": 0},
            {"timestamp": "2026-01-19 17:00:00", "user_id": "user2", "connection_state": 1},
            {"timestamp": "2026-01-20 08:00:00", "user_id": "user3", "connection_state": 0},
            {"timestamp": "2026-01-20 09:00:00", "user_id": "user3", "connection_state": 0},
            {"timestamp": "2026-01-20 12:00:00", "user_id": "user3", "connection_state": 2},
            {"timestamp": "2026-01-20 17:00:00", "user_id": "user3", "connection_state": 1},
        ]
        return sample_data


def load_and_process_data(data):
    """
    Load data into pandas DataFrame and extract time features.
    
    Args:
        data: List of dictionaries with timestamp, user_id, connection_state
        
    Returns:
        DataFrame with processed data including hour and day_of_week
    """
    df = pd.DataFrame(data)
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract hour and day of week
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    
    return df


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
    for (user, hour), group in df.groupby(['user_id', 'hour']):
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
    
    # Fetch data (using sample data by default)
    print("\n1. Fetching data...")
    data = fetch_data()
    print(f"   Fetched {len(data)} records")
    
    # Load and process data
    print("\n2. Processing data...")
    df = load_and_process_data(data)
    print(f"   Processed data shape: {df.shape}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Compute probabilities
    print("\n3. Computing probabilities P(state|user,hour)...")
    probabilities = compute_probabilities(df)
    print(f"   Computed probabilities for {len(probabilities)} (user, hour) combinations")
    
    # Make some predictions
    print("\n4. Sample predictions:")
    for user in df['user_id'].unique()[:3]:
        for hour in [8, 12, 17]:
            state, prob = predict_state(probabilities, user, hour)
            if state is not None:
                state_name = {0: 'Logged In', 1: 'Logged Out', 2: 'Idle'}[state]
                print(f"   {user} at hour {hour}: {state_name} (probability: {prob:.2f})")
    
    # Plot heatmaps
    print("\n5. Generating heatmaps...")
    plot_heatmaps(df, probabilities)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")


if __name__ == "__main__":
    main()

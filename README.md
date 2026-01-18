# Device-Insight

Network Connection State Analyzer - A minimal Python program to analyze user network connection states from API data.

## Overview

This tool analyzes user network connection patterns by computing conditional probabilities and visualizing connection states over time. Each record contains:
- **timestamp**: When the connection state was recorded
- **user_id**: Unique user identifier
- **connection_state**: 0 = Logged In, 1 = Logged Out, 2 = Idle

## Features

- Fetch data from API using `requests` (or use sample data)
- Load and process data with `pandas`
- Parse timestamps and extract hour and day of week
- Compute conditional probabilities P(state|user,hour)
- Predict most likely connection state for users at specific hours
- Generate heatmap visualizations with `matplotlib`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the analyzer with sample data:

```bash
python network_analyzer.py
```

To use with a real API, modify the `main()` function to pass your API URL:

```python
data = fetch_data(api_url="https://your-api-endpoint.com/data")
```

## Output

The program will:
1. Fetch and process connection state data
2. Compute probabilities for each user-hour combination
3. Display sample predictions
4. Generate a heatmap visualization saved as `connection_state_heatmaps.png`

## Example

```
Network Connection State Analyzer
==================================================

1. Fetching data...
   Fetched 20 records

2. Processing data...
   Processed data shape: (20, 5)
   Date range: 2026-01-18 08:00:00 to 2026-01-20 17:00:00

3. Computing probabilities P(state|user,hour)...
   Computed probabilities for 12 (user, hour) combinations

4. Sample predictions:
   user1 at hour 8: Logged In (probability: 1.00)
   user1 at hour 12: Idle (probability: 1.00)
   user1 at hour 17: Logged Out (probability: 1.00)

5. Generating heatmaps...
   Heatmaps saved to 'connection_state_heatmaps.png'

==================================================
Analysis complete!
```

## Technical Details

- **No Machine Learning**: Uses simple probability calculations based on historical frequency
- **Probability Computation**: P(state|user,hour) = count(state, user, hour) / count(user, hour)
- **Prediction Method**: Selects state with maximum probability for given user and hour
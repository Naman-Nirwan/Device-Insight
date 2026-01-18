#!/usr/bin/env python3
"""
Unit tests for network_analyzer.py
"""

import unittest
import pandas as pd
from network_analyzer import (
    fetch_data,
    load_and_process_data,
    compute_probabilities,
    predict_state
)


class TestNetworkAnalyzer(unittest.TestCase):
    
    def test_fetch_data_returns_list(self):
        """Test that fetch_data returns a list of records."""
        data = fetch_data()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
    
    def test_fetch_data_structure(self):
        """Test that fetched data has required fields."""
        data = fetch_data()
        record = data[0]
        self.assertIn('timestamp', record)
        self.assertIn('user_id', record)
        self.assertIn('connection_state', record)
    
    def test_load_and_process_data(self):
        """Test data loading and processing."""
        data = fetch_data()
        df = load_and_process_data(data)
        
        # Check that DataFrame has expected columns
        self.assertIn('timestamp', df.columns)
        self.assertIn('user_id', df.columns)
        self.assertIn('connection_state', df.columns)
        self.assertIn('hour', df.columns)
        self.assertIn('day_of_week', df.columns)
        
        # Check that hour and day_of_week are integers
        self.assertTrue(df['hour'].dtype in ['int64', 'int32'])
        self.assertTrue(df['day_of_week'].dtype in ['int64', 'int32'])
        
        # Check hour range
        self.assertTrue((df['hour'] >= 0).all())
        self.assertTrue((df['hour'] <= 23).all())
        
        # Check day_of_week range
        self.assertTrue((df['day_of_week'] >= 0).all())
        self.assertTrue((df['day_of_week'] <= 6).all())
    
    def test_compute_probabilities(self):
        """Test probability computation."""
        data = fetch_data()
        df = load_and_process_data(data)
        probabilities = compute_probabilities(df)
        
        # Check that probabilities is a dictionary
        self.assertIsInstance(probabilities, dict)
        
        # Check that probabilities sum to 1 for each (user, hour)
        for key, state_probs in probabilities.items():
            total_prob = sum(state_probs.values())
            self.assertAlmostEqual(total_prob, 1.0, places=5)
            
            # Check that all states are present
            for state in [0, 1, 2]:
                self.assertIn(state, state_probs)
                self.assertGreaterEqual(state_probs[state], 0)
                self.assertLessEqual(state_probs[state], 1)
    
    def test_predict_state(self):
        """Test state prediction."""
        data = fetch_data()
        df = load_and_process_data(data)
        probabilities = compute_probabilities(df)
        
        # Get a valid user and hour from the data
        user = df['user_id'].iloc[0]
        hour = df['hour'].iloc[0]
        
        # Test prediction
        state, prob = predict_state(probabilities, user, hour)
        
        # Check that state is valid
        self.assertIn(state, [0, 1, 2])
        
        # Check that probability is valid
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_predict_state_unknown(self):
        """Test prediction for unknown user/hour combination."""
        probabilities = {('user1', 8): {0: 0.7, 1: 0.2, 2: 0.1}}
        
        state, prob = predict_state(probabilities, 'unknown_user', 8)
        
        # Should return None and 0.0 for unknown combination
        self.assertIsNone(state)
        self.assertEqual(prob, 0.0)
    
    def test_connection_state_values(self):
        """Test that connection states are valid (0, 1, or 2)."""
        data = fetch_data()
        df = load_and_process_data(data)
        
        # Check all connection states are valid
        valid_states = {0, 1, 2}
        actual_states = set(df['connection_state'].unique())
        self.assertTrue(actual_states.issubset(valid_states))


if __name__ == '__main__':
    unittest.main()

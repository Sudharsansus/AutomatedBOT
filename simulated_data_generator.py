"""
Simulated Data Generator for Gold Trading Bot Testing

This module provides simulated market data for testing the gold trading bot
when external API connectivity is not available.
"""

import os
import json
import logging
import datetime as dt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulated_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SimulatedDataGenerator")

class SimulatedDataGenerator:
    """
    Generates simulated market data for gold trading bot testing.
    """
    
    def __init__(self, config_path: str = "gold_trading_bot_config.json"):
        """
        Initialize the simulated data generator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Create output directory
        self.output_dir = "simulated_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Simulated data generator initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            return {
                "data": {
                    "timeframes": {
                        "analysis": "H1",
                        "execution": "M15"
                    }
                },
                "simulation": {
                    "start_date": "2020-01-01",
                    "end_date": "2025-06-25",
                    "initial_price": 1500.0,
                    "volatility": 0.015,
                    "trend": 0.0001,
                    "regime_change_probability": 0.01
                }
            }
    
    def generate_price_data(self, timeframe: str = "H1", 
                          start_date: Optional[dt.datetime] = None,
                          end_date: Optional[dt.datetime] = None,
                          save: bool = True) -> pd.DataFrame:
        """
        Generate simulated price data.
        
        Args:
            timeframe: Timeframe for the data
            start_date: Start date
            end_date: End date
            save: Whether to save the data to file
            
        Returns:
            DataFrame with simulated price data
        """
        # Set default dates if not provided
        if start_date is None:
            start_date_str = self.config.get("simulation", {}).get("start_date", "2020-01-01")
            start_date = dt.datetime.fromisoformat(start_date_str)
        
        if end_date is None:
            end_date_str = self.config.get("simulation", {}).get("end_date", "2025-06-25")
            end_date = dt.datetime.fromisoformat(end_date_str)
        
        # Get simulation parameters
        initial_price = self.config.get("simulation", {}).get("initial_price", 1500.0)
        volatility = self.config.get("simulation", {}).get("volatility", 0.015)
        trend = self.config.get("simulation", {}).get("trend", 0.0001)
        regime_change_probability = self.config.get("simulation", {}).get("regime_change_probability", 0.01)
        
        # Generate date range based on timeframe
        if timeframe == "H1":
            freq = "H"
        elif timeframe == "M15":
            freq = "15min"
        elif timeframe == "D1":
            freq = "D"
        else:
            freq = "H"  # Default to hourly
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Initialize price series
        np.random.seed(42)  # For reproducibility
        prices = [initial_price]
        current_price = initial_price
        
        # Initialize market regime
        regimes = ["trending", "ranging", "volatile"]
        current_regime = np.random.choice(regimes)
        regime_history = [current_regime]
        
        # Adjust parameters based on regime
        regime_params = {
            "trending": {"volatility": volatility * 0.8, "trend": trend * 2.0},
            "ranging": {"volatility": volatility * 0.6, "trend": trend * 0.2},
            "volatile": {"volatility": volatility * 1.5, "trend": trend * 0.5}
        }
        
        # Generate prices
        for i in range(1, len(dates)):
            # Check for regime change
            if np.random.random() < regime_change_probability:
                current_regime = np.random.choice(regimes)
            
            regime_history.append(current_regime)
            
            # Get regime-specific parameters
            regime_volatility = regime_params[current_regime]["volatility"]
            regime_trend = regime_params[current_regime]["trend"]
            
            # Generate random return
            returns = np.random.normal(regime_trend, regime_volatility)
            
            # Update price
            current_price = current_price * (1 + returns)
            prices.append(current_price)
        
        # Create DataFrame
        df = pd.DataFrame(index=dates)
        df["close"] = prices
        
        # Generate OHLC data
        df["high"] = df["close"] * (1 + np.random.uniform(0, 0.005, len(df)))
        df["low"] = df["close"] * (1 - np.random.uniform(0, 0.005, len(df)))
        df["open"] = df["close"].shift(1)
        df.loc[df.index[0], "open"] = df["close"].iloc[0] * (1 - np.random.uniform(0, 0.002))
        
        # Generate volume
        df["volume"] = np.random.lognormal(10, 1, len(df))
        
        # Add regime column
        df["regime"] = regime_history
        
        # Save to file if requested
        if save:
            filename = os.path.join(self.output_dir, f"gold_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
            df.to_csv(filename)
            logger.info(f"Simulated price data saved to {filename}")
        
        return df
    
    def generate_news_data(self, start_date: Optional[dt.datetime] = None,
                         end_date: Optional[dt.datetime] = None,
                         save: bool = True) -> pd.DataFrame:
        """
        Generate simulated news data.
        
        Args:
            start_date: Start date
            end_date: End date
            save: Whether to save the data to file
            
        Returns:
            DataFrame with simulated news data
        """
        # Set default dates if not provided
        if start_date is None:
            start_date_str = self.config.get("simulation", {}).get("start_date", "2020-01-01")
            start_date = dt.datetime.fromisoformat(start_date_str)
        
        if end_date is None:
            end_date_str = self.config.get("simulation", {}).get("end_date", "2025-06-25")
            end_date = dt.datetime.fromisoformat(end_date_str)
        
        # Calculate number of days
        days = (end_date - start_date).days
        
        # Generate random news events (average 3 per day)
        num_events = int(days * 3)
        
        # Generate random timestamps
        timestamps = [start_date + dt.timedelta(
            days=np.random.uniform(0, days),
            hours=np.random.uniform(0, 24),
            minutes=np.random.uniform(0, 60)
        ) for _ in range(num_events)]
        timestamps.sort()
        
        # News categories
        categories = ["Economic", "Geopolitical", "Market", "Company", "Commodity"]
        
        # News impacts
        impacts = ["high", "medium", "low"]
        impact_weights = [0.2, 0.5, 0.3]
        
        # Generate news data
        news_data = []
        for timestamp in timestamps:
            category = np.random.choice(categories)
            impact = np.random.choice(impacts, p=impact_weights)
            
            # Generate sentiment (-1 to 1)
            if impact == "high":
                sentiment = np.random.uniform(-0.8, 0.8)
            elif impact == "medium":
                sentiment = np.random.uniform(-0.5, 0.5)
            else:
                sentiment = np.random.uniform(-0.3, 0.3)
            
            # Create news item
            news_item = {
                "time": timestamp,
                "category": category,
                "impact": impact,
                "sentiment": sentiment,
                "headline": f"Simulated {category} News ({impact} impact)"
            }
            
            news_data.append(news_item)
        
        # Create DataFrame
        df = pd.DataFrame(news_data)
        
        # Save to file if requested
        if save:
            filename = os.path.join(self.output_dir, f"news_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
            df.to_csv(filename, index=False)
            logger.info(f"Simulated news data saved to {filename}")
        
        return df
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all required simulated data.
        
        Returns:
            Dictionary of DataFrames with all simulated data
        """
        # Get timeframes from config
        analysis_tf = self.config.get("data", {}).get("timeframes", {}).get("analysis", "H1")
        execution_tf = self.config.get("data", {}).get("timeframes", {}).get("execution", "M15")
        
        # Set date range
        start_date_str = self.config.get("simulation", {}).get("start_date", "2020-01-01")
        end_date_str = self.config.get("simulation", {}).get("end_date", "2025-06-25")
        start_date = dt.datetime.fromisoformat(start_date_str)
        end_date = dt.datetime.fromisoformat(end_date_str)
        
        # Generate price data for different timeframes
        logger.info(f"Generating price data for {analysis_tf} timeframe")
        analysis_data = self.generate_price_data(timeframe=analysis_tf, start_date=start_date, end_date=end_date)
        
        logger.info(f"Generating price data for {execution_tf} timeframe")
        execution_data = self.generate_price_data(timeframe=execution_tf, start_date=start_date, end_date=end_date)
        
        # Generate news data
        logger.info("Generating news data")
        news_data = self.generate_news_data(start_date=start_date, end_date=end_date)
        
        return {
            "analysis": analysis_data,
            "execution": execution_data,
            "news": news_data
        }


# Example usage
if __name__ == "__main__":
    # Create generator
    generator = SimulatedDataGenerator()
    
    # Generate all data
    data = generator.generate_all_data()
    
    # Print summary
    for key, df in data.items():
        print(f"{key} data shape: {df.shape}")

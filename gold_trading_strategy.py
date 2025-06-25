"""
Advanced Gold Trading Bot Strategy Implementation

This module implements the advanced trading strategies for the next-generation
gold trading bot, including technical analysis, news sentiment analysis,
and deep learning price prediction.
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
        logging.FileHandler("gold_trading_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GoldTradingStrategy")

class GoldTradingStrategy:
    """
    Advanced trading strategy for gold, combining technical analysis,
    news sentiment, and deep learning price prediction.
    """
    
    def __init__(self, config_path: str = "gold_trading_bot_config.json"):
        """
        Initialize the gold trading strategy.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize strategy components
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.news_analyzer = NewsAnalyzer(self.config)
        self.price_predictor = PricePredictor(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector(self.config)
        
        # Initialize performance metrics
        self.signals = []
        
        logger.info("Gold trading strategy initialized")
    
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
                "strategy": {
                    "risk_per_trade": 0.01,
                    "max_drawdown": 0.2,
                    "take_profit_atr_multiple": 3,
                    "stop_loss_atr_multiple": 1.5
                },
                "news_analysis": {
                    "sentiment_threshold": 0.3,
                    "impact_levels": ["high", "medium"],
                    "lookback_hours": 24
                },
                "price_prediction": {
                    "confidence_threshold": 0.6,
                    "prediction_horizon": 12,
                    "feature_importance_threshold": 0.05
                }
            }
    
    def generate_signal(self, df: pd.DataFrame, news_data: Optional[pd.DataFrame] = None,
                      market_data: Optional[Dict] = None) -> Dict:
        """
        Generate trading signal based on all available data.
        
        Args:
            df: Historical price data
            news_data: News data
            market_data: Additional market data
            
        Returns:
            Trading signal
        """
        try:
            # Detect market regime
            regime = self.regime_detector.detect_regime(df)
            logger.info(f"Detected market regime: {regime}")
            
            # Get technical signal
            technical_signal = self.technical_analyzer.generate_signal(df, regime)
            
            # Get news signal
            news_signal = self.news_analyzer.generate_signal(news_data, df.index[-1])
            
            # Get prediction signal
            prediction_signal = self.price_predictor.generate_signal(df)
            
            # Combine signals
            combined_signal = self._combine_signals(
                technical_signal, news_signal, prediction_signal, regime
            )
            
            # Apply risk management
            final_signal = self.risk_manager.apply_risk_management(combined_signal, df)
            
            # Add to signals history
            self.signals.append(final_signal)
            
            return final_signal
        
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                "time": df.index[-1] if not df.empty else dt.datetime.now(),
                "price": df["close"].iloc[-1] if not df.empty and "close" in df.columns else 0,
                "action": "none",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _combine_signals(self, technical_signal: Dict, news_signal: Dict,
                       prediction_signal: Dict, regime: str) -> Dict:
        """
        Combine signals from different sources.
        
        Args:
            technical_signal: Signal from technical analysis
            news_signal: Signal from news analysis
            prediction_signal: Signal from price prediction
            regime: Current market regime
            
        Returns:
            Combined signal
        """
        # Initialize combined signal with technical signal
        combined_signal = technical_signal.copy()
        
        # Adjust weights based on market regime
        if regime == "trending":
            technical_weight = 0.5
            news_weight = 0.2
            prediction_weight = 0.3
        elif regime == "ranging":
            technical_weight = 0.4
            news_weight = 0.3
            prediction_weight = 0.3
        elif regime == "volatile":
            technical_weight = 0.3
            news_weight = 0.4
            prediction_weight = 0.3
        else:  # default
            technical_weight = 0.4
            news_weight = 0.3
            prediction_weight = 0.3
        
        # Calculate confidence
        technical_confidence = technical_signal.get("confidence", 0.0) * technical_weight
        news_confidence = news_signal.get("confidence", 0.0) * news_weight
        prediction_confidence = prediction_signal.get("confidence", 0.0) * prediction_weight
        
        # Determine action based on weighted confidence
        buy_confidence = 0.0
        sell_confidence = 0.0
        
        if technical_signal.get("action") == "buy":
            buy_confidence += technical_confidence
        elif technical_signal.get("action") == "sell":
            sell_confidence += technical_confidence
        
        if news_signal.get("action") == "buy":
            buy_confidence += news_confidence
        elif news_signal.get("action") == "sell":
            sell_confidence += news_confidence
        
        if prediction_signal.get("action") == "buy":
            buy_confidence += prediction_confidence
        elif prediction_signal.get("action") == "sell":
            sell_confidence += prediction_confidence
        
        # Determine final action
        if buy_confidence > 0.4 and buy_confidence > sell_confidence:
            combined_signal["action"] = "buy"
            combined_signal["confidence"] = buy_confidence
        elif sell_confidence > 0.4 and sell_confidence > buy_confidence:
            combined_signal["action"] = "sell"
            combined_signal["confidence"] = sell_confidence
        else:
            combined_signal["action"] = "none"
            combined_signal["confidence"] = max(buy_confidence, sell_confidence)
        
        # Add component signals for analysis
        combined_signal["components"] = {
            "technical": technical_signal,
            "news": news_signal,
            "prediction": prediction_signal,
            "regime": regime
        }
        
        return combined_signal


class TechnicalAnalyzer:
    """
    Technical analysis component for the gold trading strategy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the technical analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Technical analyzer initialized")
    
    def generate_signal(self, df: pd.DataFrame, regime: str) -> Dict:
        """
        Generate trading signal based on technical analysis.
        
        Args:
            df: Historical price data
            regime: Current market regime
            
        Returns:
            Trading signal
        """
        try:
            # Ensure we have enough data
            if len(df) < 50:
                logger.warning("Not enough data for technical analysis")
                return {
                    "time": df.index[-1] if not df.empty else dt.datetime.now(),
                    "price": df["close"].iloc[-1] if not df.empty and "close" in df.columns else 0,
                    "action": "none",
                    "confidence": 0.0
                }
            
            # Calculate indicators if not already present
            df = self._calculate_indicators(df)
            
            # Get latest data
            latest = df.iloc[-1]
            
            # Initialize signal
            signal = {
                "time": latest.name,
                "price": latest["close"],
                "action": "none",
                "confidence": 0.0
            }
            
            # Apply regime-specific strategy
            if regime == "trending":
                signal = self._trending_strategy(df, signal)
            elif regime == "ranging":
                signal = self._ranging_strategy(df, signal)
            elif regime == "volatile":
                signal = self._volatile_strategy(df, signal)
            else:
                # Default strategy
                signal = self._default_strategy(df, signal)
            
            # Set stop loss and take profit
            if signal["action"] != "none" and "atr" in df.columns:
                stop_loss_multiple = self.config.get("strategy", {}).get("stop_loss_atr_multiple", 1.5)
                take_profit_multiple = self.config.get("strategy", {}).get("take_profit_atr_multiple", 3.0)
                
                if signal["action"] == "buy":
                    signal["stop_loss"] = latest["close"] - latest["atr"] * stop_loss_multiple
                    signal["take_profit"] = latest["close"] + latest["atr"] * take_profit_multiple
                else:  # sell
                    signal["stop_loss"] = latest["close"] + latest["atr"] * stop_loss_multiple
                    signal["take_profit"] = latest["close"] - latest["atr"] * take_profit_multiple
            
            return signal
        
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {
                "time": df.index[-1] if not df.empty else dt.datetime.now(),
                "price": df["close"].iloc[-1] if not df.empty and "close" in df.columns else 0,
                "action": "none",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: Historical price data
            
        Returns:
            DataFrame with indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Moving averages
        if "ema20" not in df.columns:
            df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        if "ema50" not in df.columns:
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        if "sma200" not in df.columns:
            df["sma200"] = df["close"].rolling(window=200).mean()
        
        # RSI
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df["rsi"] = 100 - (100 / (1 + rs))
        
        # ATR
        if "atr" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df["atr"] = true_range.rolling(window=14).mean()
        
        # MACD
        if "macd" not in df.columns:
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        if "bb_upper" not in df.columns:
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            df["bb_std"] = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        
        # Stochastic Oscillator
        if "stoch_k" not in df.columns:
            low_14 = df["low"].rolling(window=14).min()
            high_14 = df["high"].rolling(window=14).max()
            df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
            df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        
        return df
    
    def _trending_strategy(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """
        Strategy for trending markets.
        
        Args:
            df: Historical price data with indicators
            signal: Initial signal
            
        Returns:
            Updated signal
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Moving average trend following
        if latest["ema20"] > latest["ema50"] > latest["sma200"]:
            # Strong uptrend
            if latest["rsi"] < 70:  # Not overbought
                signal["action"] = "buy"
                signal["confidence"] = 0.8
        elif latest["ema20"] < latest["ema50"] < latest["sma200"]:
            # Strong downtrend
            if latest["rsi"] > 30:  # Not oversold
                signal["action"] = "sell"
                signal["confidence"] = 0.8
        # MACD crossover
        elif prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]:
            # Bullish crossover
            if latest["close"] > latest["ema50"]:  # Confirm with trend
                signal["action"] = "buy"
                signal["confidence"] = 0.7
        elif prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
            # Bearish crossover
            if latest["close"] < latest["ema50"]:  # Confirm with trend
                signal["action"] = "sell"
                signal["confidence"] = 0.7
        
        return signal
    
    def _ranging_strategy(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """
        Strategy for ranging markets.
        
        Args:
            df: Historical price data with indicators
            signal: Initial signal
            
        Returns:
            Updated signal
        """
        latest = df.iloc[-1]
        
        # Bollinger Band strategy
        if latest["close"] < latest["bb_lower"] and latest["rsi"] < 30:
            # Oversold at lower band
            signal["action"] = "buy"
            signal["confidence"] = 0.7
        elif latest["close"] > latest["bb_upper"] and latest["rsi"] > 70:
            # Overbought at upper band
            signal["action"] = "sell"
            signal["confidence"] = 0.7
        
        # RSI divergence
        elif latest["rsi"] < 30 and latest["close"] > df["close"].rolling(window=5).min():
            # Bullish divergence
            signal["action"] = "buy"
            signal["confidence"] = 0.6
        elif latest["rsi"] > 70 and latest["close"] < df["close"].rolling(window=5).max():
            # Bearish divergence
            signal["action"] = "sell"
            signal["confidence"] = 0.6
        
        return signal
    
    def _volatile_strategy(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """
        Strategy for volatile markets.
        
        Args:
            df: Historical price data with indicators
            signal: Initial signal
            
        Returns:
            Updated signal
        """
        latest = df.iloc[-1]
        
        # More conservative approach in volatile markets
        # Only take strong signals with confirmation
        
        # Strong oversold
        if latest["rsi"] < 20 and latest["stoch_k"] < 20 and latest["stoch_k"] > latest["stoch_d"]:
            signal["action"] = "buy"
            signal["confidence"] = 0.6
        
        # Strong overbought
        elif latest["rsi"] > 80 and latest["stoch_k"] > 80 and latest["stoch_k"] < latest["stoch_d"]:
            signal["action"] = "sell"
            signal["confidence"] = 0.6
        
        # Bollinger Band squeeze breakout
        bb_width = (latest["bb_upper"] - latest["bb_lower"]) / latest["bb_middle"]
        prev_width = (df["bb_upper"].iloc[-20:-1] - df["bb_lower"].iloc[-20:-1]) / df["bb_middle"].iloc[-20:-1]
        
        if bb_width > prev_width.mean() * 1.5:  # Breakout from squeeze
            if latest["close"] > latest["bb_upper"]:
                signal["action"] = "buy"
                signal["confidence"] = 0.7
            elif latest["close"] < latest["bb_lower"]:
                signal["action"] = "sell"
                signal["confidence"] = 0.7
        
        return signal
    
    def _default_strategy(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """
        Default strategy when regime is unclear.
        
        Args:
            df: Historical price data with indicators
            signal: Initial signal
            
        Returns:
            Updated signal
        """
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Moving average crossover
        if prev["ema20"] < prev["ema50"] and latest["ema20"] > latest["ema50"]:
            # Bullish crossover
            signal["action"] = "buy"
            signal["confidence"] = 0.6
        elif prev["ema20"] > prev["ema50"] and latest["ema20"] < latest["ema50"]:
            # Bearish crossover
            signal["action"] = "sell"
            signal["confidence"] = 0.6
        
        # RSI extremes
        elif latest["rsi"] < 30:
            signal["action"] = "buy"
            signal["confidence"] = 0.5
        elif latest["rsi"] > 70:
            signal["action"] = "sell"
            signal["confidence"] = 0.5
        
        return signal


class NewsAnalyzer:
    """
    News analysis component for the gold trading strategy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the news analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sentiment_threshold = config.get("news_analysis", {}).get("sentiment_threshold", 0.3)
        self.impact_levels = config.get("news_analysis", {}).get("impact_levels", ["high", "medium"])
        self.lookback_hours = config.get("news_analysis", {}).get("lookback_hours", 24)
        logger.info("News analyzer initialized")
    
    def generate_signal(self, news_data: Optional[pd.DataFrame], current_time: dt.datetime) -> Dict:
        """
        Generate trading signal based on news sentiment analysis.
        
        Args:
            news_data: News data
            current_time: Current time
            
        Returns:
            Trading signal
        """
        try:
            # Initialize signal
            signal = {
                "time": current_time,
                "action": "none",
                "confidence": 0.0
            }
            
            # If no news data, return neutral signal
            if news_data is None or news_data.empty:
                return signal
            
            # Filter relevant news
            lookback_time = current_time - dt.timedelta(hours=self.lookback_hours)
            relevant_news = news_data[
                (news_data["time"] >= lookback_time) & 
                (news_data["time"] <= current_time) &
                (news_data["impact"].isin(self.impact_levels))
            ]
            
            if relevant_news.empty:
                return signal
            
            # Calculate weighted sentiment
            weighted_sentiment = 0.0
            total_weight = 0.0
            
            for _, news in relevant_news.iterrows():
                # Calculate time weight (more recent news has higher weight)
                hours_ago = (current_time - news["time"]).total_seconds() / 3600
                time_weight = max(0, 1 - (hours_ago / self.lookback_hours))
                
                # Calculate impact weight
                if news["impact"] == "high":
                    impact_weight = 1.0
                elif news["impact"] == "medium":
                    impact_weight = 0.6
                else:  # low
                    impact_weight = 0.3
                
                # Combined weight
                weight = time_weight * impact_weight
                
                # Add to weighted sentiment
                weighted_sentiment += news["sentiment"] * weight
                total_weight += weight
            
            # Normalize sentiment
            if total_weight > 0:
                normalized_sentiment = weighted_sentiment / total_weight
            else:
                normalized_sentiment = 0.0
            
            # Generate signal based on sentiment
            if normalized_sentiment > self.sentiment_threshold:
                signal["action"] = "buy"
                signal["confidence"] = min(0.8, normalized_sentiment)
            elif normalized_sentiment < -self.sentiment_threshold:
                signal["action"] = "sell"
                signal["confidence"] = min(0.8, abs(normalized_sentiment))
            
            # Add sentiment info to signal
            signal["sentiment"] = normalized_sentiment
            signal["news_count"] = len(relevant_news)
            
            return signal
        
        except Exception as e:
            logger.error(f"Error in news analysis: {e}")
            return {
                "time": current_time,
                "action": "none",
                "confidence": 0.0,
                "error": str(e)
            }


class PricePredictor:
    """
    Price prediction component for the gold trading strategy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the price predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.confidence_threshold = config.get("price_prediction", {}).get("confidence_threshold", 0.6)
        self.prediction_horizon = config.get("price_prediction", {}).get("prediction_horizon", 12)
        logger.info("Price predictor initialized")
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal based on price prediction.
        
        Args:
            df: Historical price data
            
        Returns:
            Trading signal
        """
        try:
            # Initialize signal
            signal = {
                "time": df.index[-1] if not df.empty else dt.datetime.now(),
                "price": df["close"].iloc[-1] if not df.empty and "close" in df.columns else 0,
                "action": "none",
                "confidence": 0.0
            }
            
            # Ensure we have enough data
            if len(df) < 100:
                logger.warning("Not enough data for price prediction")
                return signal
            
            # In a real implementation, this would use a trained model
            # For this demonstration, we'll use a simplified approach
            
            # Calculate momentum
            returns = df["close"].pct_change(self.prediction_horizon)
            momentum = returns.rolling(window=20).mean().iloc[-1]
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Normalize momentum by volatility for a crude prediction
            if volatility > 0:
                normalized_momentum = momentum / volatility
            else:
                normalized_momentum = 0
            
            # Convert to a confidence score (sigmoid-like)
            confidence = 2 / (1 + np.exp(-5 * abs(normalized_momentum))) - 1
            
            # Generate signal based on prediction
            if normalized_momentum > 0 and confidence > self.confidence_threshold:
                signal["action"] = "buy"
                signal["confidence"] = confidence
            elif normalized_momentum < 0 and confidence > self.confidence_threshold:
                signal["action"] = "sell"
                signal["confidence"] = confidence
            
            # Add prediction info to signal
            signal["prediction"] = {
                "direction": "up" if normalized_momentum > 0 else "down",
                "magnitude": abs(normalized_momentum),
                "confidence": confidence,
                "horizon": self.prediction_horizon
            }
            
            return signal
        
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return {
                "time": df.index[-1] if not df.empty else dt.datetime.now(),
                "price": df["close"].iloc[-1] if not df.empty and "close" in df.columns else 0,
                "action": "none",
                "confidence": 0.0,
                "error": str(e)
            }


class RiskManager:
    """
    Risk management component for the gold trading strategy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_per_trade = config.get("strategy", {}).get("risk_per_trade", 0.01)
        self.max_drawdown = config.get("strategy", {}).get("max_drawdown", 0.2)
        self.stop_loss_atr_multiple = config.get("strategy", {}).get("stop_loss_atr_multiple", 1.5)
        self.take_profit_atr_multiple = config.get("strategy", {}).get("take_profit_atr_multiple", 3.0)
        logger.info("Risk manager initialized")
    
    def apply_risk_management(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """
        Apply risk management to trading signal.
        
        Args:
            signal: Trading signal
            df: Historical price data
            
        Returns:
            Risk-adjusted signal
        """
        try:
            # If no action, return as is
            if signal["action"] == "none":
                return signal
            
            # Get latest data
            latest = df.iloc[-1]
            
            # Set risk-adjusted position size
            signal["risk_per_trade"] = self.risk_per_trade
            
            # Set stop loss and take profit if not already set
            if "stop_loss" not in signal and "atr" in df.columns:
                if signal["action"] == "buy":
                    signal["stop_loss"] = latest["close"] - latest["atr"] * self.stop_loss_atr_multiple
                else:  # sell
                    signal["stop_loss"] = latest["close"] + latest["atr"] * self.stop_loss_atr_multiple
            
            if "take_profit" not in signal and "atr" in df.columns:
                if signal["action"] == "buy":
                    signal["take_profit"] = latest["close"] + latest["atr"] * self.take_profit_atr_multiple
                else:  # sell
                    signal["take_profit"] = latest["close"] - latest["atr"] * self.take_profit_atr_multiple
            
            # Calculate risk-reward ratio
            if "stop_loss" in signal and "take_profit" in signal:
                if signal["action"] == "buy":
                    risk = latest["close"] - signal["stop_loss"]
                    reward = signal["take_profit"] - latest["close"]
                else:  # sell
                    risk = signal["stop_loss"] - latest["close"]
                    reward = latest["close"] - signal["take_profit"]
                
                if risk > 0:
                    signal["risk_reward_ratio"] = reward / risk
                else:
                    signal["risk_reward_ratio"] = 0
                
                # Filter out poor risk-reward trades
                if signal["risk_reward_ratio"] < 1.5:
                    signal["action"] = "none"
                    signal["confidence"] = 0.0
                    signal["filtered_reason"] = "poor_risk_reward"
            
            # Check for overexposure (would be implemented with position tracking)
            # This is a placeholder for a real implementation
            
            return signal
        
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            # In case of error, default to no action
            signal["action"] = "none"
            signal["confidence"] = 0.0
            signal["error"] = str(e)
            return signal


class MarketRegimeDetector:
    """
    Market regime detection component for the gold trading strategy.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the market regime detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.info("Market regime detector initialized")
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime.
        
        Args:
            df: Historical price data
            
        Returns:
            Market regime ("trending", "ranging", "volatile", or "unknown")
        """
        try:
            # Ensure we have enough data
            if len(df) < 100:
                logger.warning("Not enough data for regime detection")
                return "unknown"
            
            # Calculate indicators if not already present
            if "atr" not in df.columns:
                high_low = df["high"] - df["low"]
                high_close = (df["high"] - df["close"].shift()).abs()
                low_close = (df["low"] - df["close"].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df["atr"] = true_range.rolling(window=14).mean()
            
            if "sma20" not in df.columns:
                df["sma20"] = df["close"].rolling(window=20).mean()
            
            if "sma50" not in df.columns:
                df["sma50"] = df["close"].rolling(window=50).mean()
            
            # Calculate volatility
            recent_volatility = df["atr"].iloc[-20:].mean() / df["close"].iloc[-20:].mean()
            baseline_volatility = df["atr"].iloc[-100:-20].mean() / df["close"].iloc[-100:-20].mean()
            relative_volatility = recent_volatility / baseline_volatility if baseline_volatility > 0 else 1.0
            
            # Calculate trend strength
            price_change = (df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]
            sma_diff = (df["sma20"].iloc[-1] - df["sma50"].iloc[-1]) / df["sma50"].iloc[-1]
            
            # Detect regime
            if relative_volatility > 1.5:
                return "volatile"
            elif abs(price_change) > 0.05 or abs(sma_diff) > 0.02:
                return "trending"
            else:
                return "ranging"
        
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return "unknown"


# Example usage
if __name__ == "__main__":
    # Create strategy
    strategy = GoldTradingStrategy()
    
    # Load sample data
    try:
        df = pd.read_csv("sample_gold_data.csv", index_col=0, parse_dates=True)
        
        # Generate signal
        signal = strategy.generate_signal(df)
        
        print(f"Generated signal: {signal}")
    except Exception as e:
        print(f"Error loading sample data: {e}")

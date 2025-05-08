import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple
from config import TradingConfig
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data"""

    symbol: str
    timeframe: str
    current_price: float
    timestamp: datetime
    spread: float
    volume: float
    tick_value: float
    point: float
    digits: int


class MarketAccessor:
    """Class for accessing market data and managing signals"""

    def __init__(self):
        self.is_initialized = False
        self.current_signals: Dict[str, Dict] = {}
        self.market_data_cache: Dict[str, MarketData] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.CACHE_DURATION = timedelta(minutes=1)

    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False

            # Login to MT5 (read-only access)
            if not mt5.login(
                login=TradingConfig.MT5_LOGIN,
                password=TradingConfig.MT5_PASSWORD,
                server=TradingConfig.MT5_SERVER,
            ):
                logger.error("Failed to login to MT5")
                return False

            self.is_initialized = True
            logger.info("Successfully connected to MT5")
            return True

        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}"
            current_time = datetime.now(timezone.utc)

            if (
                cache_key in self.market_data_cache
                and cache_key in self.cache_expiry
                and current_time < self.cache_expiry[cache_key]
            ):
                return self.market_data_cache[cache_key]

            # Initialize MT5 if needed
            if not self.is_initialized and not self.initialize_mt5():
                return None

            # Get symbol information
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return None

            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get current tick for {symbol}")
                return None

            # Create market data object
            market_data = MarketData(
                symbol=symbol,
                timeframe="CURRENT",
                current_price=tick.ask,  # Using ask price
                timestamp=datetime.fromtimestamp(tick.time, timezone.utc),
                spread=tick.ask - tick.bid,
                volume=tick.volume,
                tick_value=symbol_info.trade_tick_value,
                point=symbol_info.point,
                digits=symbol_info.digits,
            )

            # Update cache
            self.market_data_cache[cache_key] = market_data
            self.cache_expiry[cache_key] = current_time + self.CACHE_DURATION

            return market_data

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None

    def get_historical_data(
        self, symbol: str, timeframe: str, start_pos: int = 0, num_candles: int = 1000
    ) -> Optional[pd.DataFrame]:
        """Get historical price data"""
        try:
            if not self.is_initialized and not self.initialize_mt5():
                return None

            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
            }

            if timeframe not in timeframe_map:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None

            # Get historical data
            rates = mt5.copy_rates_from_pos(
                symbol, timeframe_map[timeframe], start_pos, num_candles
            )

            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get historical data for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None

    def update_signal(self, symbol: str, timeframe: str, signal_data: Dict) -> bool:
        """Update current signal for a symbol"""
        try:
            key = f"{symbol}_{timeframe}"
            self.current_signals[key] = {
                **signal_data,
                "updated_at": datetime.now(timezone.utc),
            }
            logger.info(f"Updated signal for {key}: {signal_data['signal']}")
            return True

        except Exception as e:
            logger.error(f"Error updating signal: {str(e)}")
            return False

    def get_current_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get current signal for a symbol"""
        try:
            key = f"{symbol}_{timeframe}"
            signal = self.current_signals.get(key)

            if signal is None:
                return None

            # Check if signal has expired
            signal_time = datetime.fromisoformat(signal["timestamp"])
            current_time = datetime.now(timezone.utc)

            if (
                current_time - signal_time
            ).total_seconds() > TradingConfig.SIGNAL_PARAMS.signal_expiry * 60:
                del self.current_signals[key]
                return None

            return signal

        except Exception as e:
            logger.error(f"Error getting current signal: {str(e)}")
            return None

    def check_market_conditions(self, symbol: str) -> Dict:
        """Check current market conditions"""
        try:
            market_data = self.get_market_data(symbol)
            if market_data is None:
                return {"is_valid": False, "message": "Failed to get market data"}

            # Get current time in UTC
            current_time = datetime.now(timezone.utc)

            # Check if market is open
            if not TradingConfig.is_market_open(symbol):
                return {"is_valid": False, "message": "Market is closed"}

            # Check spread
            if market_data.spread > market_data.point * 20:  # Max 2.0 pips spread
                return {"is_valid": False, "message": "Spread too high"}

            # Check if there's enough recent market activity
            if market_data.volume == 0:
                return {"is_valid": False, "message": "No recent market activity"}

            return {
                "is_valid": True,
                "message": "Market conditions OK",
                "data": {
                    "spread": market_data.spread,
                    "volume": market_data.volume,
                    "current_price": market_data.current_price,
                    "timestamp": market_data.timestamp,
                },
            }

        except Exception as e:
            logger.error(f"Error checking market conditions: {str(e)}")
            return {"is_valid": False, "message": f"Error: {str(e)}"}

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.is_initialized:
                mt5.shutdown()
                self.is_initialized = False
                logger.info("MT5 connection closed")
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

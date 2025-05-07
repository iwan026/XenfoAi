from datetime import datetime, timezone
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple
from config import TradingConfig
from data_processor import ForexDataProcessor
from model import DeepForexModel

logger = logging.getLogger(__name__)


class ForexTrader:
    """Class for handling forex trading operations"""

    def __init__(self):
        self.data_processor = ForexDataProcessor()
        self.model = DeepForexModel()
        self.last_prediction_time: Dict[str, datetime] = {}
        self.open_positions: Dict[str, Dict] = {}

    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False

            # Login to MT5
            if not mt5.login(
                login=TradingConfig.MT5_LOGIN,
                password=TradingConfig.MT5_PASSWORD,
                server=TradingConfig.MT5_SERVER,
            ):
                logger.error("Failed to login to MT5")
                return False

            logger.info("Successfully connected to MT5")
            return True

        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False

    def get_trading_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get trading signal for a symbol"""
        try:
            # Check if we already predicted recently
            current_time = datetime.now(timezone.utc)
            last_time = self.last_prediction_time.get(f"{symbol}_{timeframe}")

            if last_time and (current_time - last_time).total_seconds() < 60:
                logger.info(
                    f"Skipping prediction for {symbol}, too soon since last prediction"
                )
                return None

            # Get latest data
            df = self.data_processor.get_mt5_historical_data(
                symbol, timeframe, TradingConfig.LOOKBACK_PERIOD + 100
            )

            if df is None:
                return None

            # Prepare sequence for prediction
            sequence = self.data_processor.prepare_latest_sequence(df)
            if sequence is None:
                return None

            # Load model if not already loaded
            if self.model.model is None:
                if not self.model.load_model(symbol, timeframe):
                    return None

            # Make prediction
            prediction = self.model.model.predict(sequence)[0][0]
            confidence = float(abs(prediction - 0.5) * 2)  # Scale to 0-1

            # Determine signal
            if confidence < TradingConfig.CONFIDENCE_THRESHOLD:
                signal = "NEUTRAL"
            elif prediction > 0.5:
                signal = "BUY"
            else:
                signal = "SELL"

            # Get current market info
            current_tick = mt5.symbol_info_tick(symbol)
            if current_tick is None:
                logger.error(f"Failed to get current tick for {symbol}")
                return None

            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": signal,
                "confidence": confidence,
                "prediction": float(prediction),
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "price": current_tick.ask,
                "spread": current_tick.ask - current_tick.bid,
                "atr": df["atr"].iloc[-1] if "atr" in df else None,
                "volatility": df["volatility"].iloc[-1] if "volatility" in df else None,
            }

            # Update last prediction time
            self.last_prediction_time[f"{symbol}_{timeframe}"] = current_time

            return result

        except Exception as e:
            logger.error(f"Error getting trading signal: {str(e)}")
            return None

    def calculate_position_size(
        self, symbol: str, risk_percent: float = 1.0, stop_loss_pips: float = 50.0
    ) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                return 0.0

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0

            # Calculate position size
            account_balance = account_info.balance
            risk_amount = account_balance * (risk_percent / 100)

            # Convert stop loss to price
            point_value = symbol_info.point
            stop_loss_price = stop_loss_pips * point_value

            # Calculate lots
            lot_step = symbol_info.volume_step
            min_lot = symbol_info.volume_min
            max_lot = symbol_info.volume_max

            contract_size = symbol_info.trade_contract_size
            tick_value = symbol_info.trade_tick_value

            # Position size calculation
            position_size = risk_amount / (stop_loss_price * tick_value * contract_size)

            # Round to nearest lot step
            position_size = round(position_size / lot_step) * lot_step

            # Ensure within limits
            position_size = max(min_lot, min(position_size, max_lot))

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def place_market_order(
        self,
        symbol: str,
        order_type: str,
        lots: float,
        stop_loss_pips: float = 50.0,
        take_profit_pips: float = 100.0,
    ) -> bool:
        """Place a market order with specified parameters"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return False

            point = symbol_info.point
            digits = symbol_info.digits

            if order_type == "BUY":
                order = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask
                sl = price - stop_loss_pips * point
                tp = price + take_profit_pips * point
            else:  # SELL
                order = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid
                sl = price + stop_loss_pips * point
                tp = price - take_profit_pips * point

            # Prepare the request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lots),
                "type": order,
                "price": price,
                "sl": round(sl, digits),
                "tp": round(tp, digits),
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Signal",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return False

            # Track open position
            self.open_positions[symbol] = {
                "ticket": result.order,
                "type": order_type,
                "lots": lots,
                "open_price": price,
                "sl": sl,
                "tp": tp,
                "open_time": datetime.now(timezone.utc),
            }

            logger.info(f"Order placed successfully: {result.order}")
            return True

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False

    def manage_trades(self) -> None:
        """Manage existing trades"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                logger.info("No open positions")
                return

            for position in positions:
                symbol = position.symbol

                # Check if we're tracking this position
                if symbol not in self.open_positions:
                    continue

                current_price = (
                    mt5.symbol_info_tick(symbol).bid
                    if position.type == mt5.ORDER_TYPE_BUY
                    else mt5.symbol_info_tick(symbol).ask
                )

                # Get new trading signal
                signal = self.get_trading_signal(
                    symbol, "H1"
                )  # Using H1 timeframe for management
                if signal is None:
                    continue

                # Check for exit conditions
                should_exit = False

                # 1. Signal reversal with high confidence
                if signal["confidence"] >= TradingConfig.CONFIDENCE_THRESHOLD and (
                    (position.type == mt5.ORDER_TYPE_BUY and signal["signal"] == "SELL")
                    or (
                        position.type == mt5.ORDER_TYPE_SELL
                        and signal["signal"] == "BUY"
                    )
                ):
                    should_exit = True

                if should_exit:
                    self.close_position(position)

        except Exception as e:
            logger.error(f"Error managing trades: {str(e)}")

    def close_position(self, position) -> bool:
        """Close a specific position"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Signal Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position: {result.comment}")
                return False

            # Remove from tracking
            if position.symbol in self.open_positions:
                del self.open_positions[position.symbol]

            logger.info(f"Position closed successfully: {position.ticket}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False

    def run_trading_cycle(self, symbols: List[str], timeframe: str) -> None:
        """Run a complete trading cycle"""
        try:
            if not self.initialize_mt5():
                return

            for symbol in symbols:
                # Get trading signal
                signal = self.get_trading_signal(symbol, timeframe)
                if signal is None or signal["signal"] == "NEUTRAL":
                    continue

                # Check if symbol already has an open position
                if symbol in self.open_positions:
                    logger.info(f"Position already open for {symbol}")
                    continue

                # Calculate position size
                lots = self.calculate_position_size(symbol)
                if lots == 0:
                    continue

                # Place order
                if signal["signal"] in ["BUY", "SELL"]:
                    self.place_market_order(symbol, signal["signal"], lots)

            # Manage existing trades
            self.manage_trades()

        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
        finally:
            mt5.shutdown()

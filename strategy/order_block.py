import pandas as pd

def add_order_block_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add Order Block detection and analysis features"""
    try:
        # Detect Bullish Order Blocks
        df["bull_ob"] = (
            (df["close"] < df["open"])  # Bearish candle
            & (df["close"].shift(-1) > df["high"])  # Next candle breaks high
            & (df["volume"] > df["volume"].rolling(20).mean())  # High volume
            & (abs(df["close"] - df["open"]) > df["atr"])  # Significant size
        )

        # Detect Bearish Order Blocks
        df["bear_ob"] = (
            (df["close"] > df["open"])  # Bullish candle
            & (df["close"].shift(-1) < df["low"])  # Next candle breaks low
            & (df["volume"] > df["volume"].rolling(20).mean())  # High volume
            & (abs(df["close"] - df["open"]) > df["atr"])  # Significant size
        )

        # Order Block Strength
        df["ob_strength"] = np.where(
            df["bull_ob"],
            (df["high"].shift(-1) - df["low"]) / df["atr"],
            np.where(df["bear_ob"], (df["high"] - df["low"].shift(-1)) / df["atr"], 0),
        )

        # Order Block Volume Analysis
        df["ob_volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # Order Block Momentum
        df["ob_momentum"] = np.where(
            df["bull_ob"] | df["bear_ob"],
            abs(df["close"] - df["open"]) * df["ob_volume_ratio"],
            0,
        )

        # Order Block Quality Score (0-1)
        df["ob_quality"] = np.where(
            df["bull_ob"] | df["bear_ob"],
            (
                df["ob_strength"] * 0.4
                + df["ob_volume_ratio"] * 0.3
                + df["ob_momentum"] * 0.3
            )
            / 2,  # Normalize to 0-1
            0,
        )

        # Reversal Zone Detection
        df["reversal_zone"] = (df["bull_ob"] | df["bear_ob"]) & (
            df["rsi"].shift(1) < 30
        ) | (df["rsi"].shift(1) > 70)

        return df

    except Exception as e:
        logger.error(f"Error adding order block features: {str(e)}")
        return df

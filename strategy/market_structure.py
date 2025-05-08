import pandas as pd

def add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add Market Structure analysis features"""
    try:
        # Identify Swing Points
        df["swing_high"] = (
            (df["high"] > df["high"].shift(1))
            & (df["high"] > df["high"].shift(-1))
            & (df["high"] > df["high"].shift(2))
            & (df["high"] > df["high"].shift(-2))
        )

        df["swing_low"] = (
            (df["low"] < df["low"].shift(1))
            & (df["low"] < df["low"].shift(-1))
            & (df["low"] < df["low"].shift(2))
            & (df["low"] < df["low"].shift(-2))
        )

        # Higher Highs & Lower Lows
        df["higher_high"] = df["swing_high"] & (
            df["high"] > df["high"].rolling(20).max().shift(1)
        )

        df["lower_low"] = df["swing_low"] & (
            df["low"] < df["low"].rolling(20).min().shift(1)
        )

        # Market Structure States
        df["structure_state"] = np.where(
            df["higher_high"],
            "uptrend",
            np.where(df["lower_low"], "downtrend", "ranging"),
        )

        # Breakout Detection
        df["breakout_level"] = np.where(
            df["close"] > df["high"].rolling(20).max(),
            1,  # Bullish breakout
            np.where(
                df["close"] < df["low"].rolling(20).min(),
                -1,  # Bearish breakout
                0,
            ),
        )

        # Breakout Strength
        df["breakout_strength"] = np.where(
            df["breakout_level"] != 0,
            abs(df["close"] - df["open"])
            * df["volume"]
            / df["volume"].rolling(20).mean(),
            0,
        )

        # Trend Validation
        df["trend_validated"] = (
            (df["breakout_level"] != 0)
            & (df["volume"] > df["volume"].rolling(20).mean())
            & (abs(df["close"] - df["open"]) > df["atr"])
        )

        # Market Structure Change Points
        df["structure_change"] = (
            df["structure_state"] != df["structure_state"].shift(1)
        ) & (df["volume"] > df["volume"].rolling(20).mean())

        # Structure Quality Score (0-1)
        df["structure_quality"] = (
            (df["trend_validated"].astype(int) * 0.4)
            + (
                df["breakout_strength"]
                / df["breakout_strength"].rolling(20).max()
                * 0.3
            )
            + ((df["volume"] / df["volume"].rolling(20).mean()) * 0.3)
        ).fillna(0)

        return df

    except Exception as e:
        logger.error(f"Error adding market structure features: {str(e)}")
        return df

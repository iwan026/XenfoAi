import pandas as pd

def add_liquidity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add Liquidity Analysis features"""
    try:
        # Liquidity Levels Detection
        df["buy_liquidity"] = df["volume"].where(
            (df["close"] > df["open"])
            & (df["volume"] > df["volume"].rolling(20).mean()),
            0,
        )

        df["sell_liquidity"] = df["volume"].where(
            (df["close"] < df["open"])
            & (df["volume"] > df["volume"].rolling(20).mean()),
            0,
        )

        # Liquidity Sweep Detection
        df["liquidity_sweep_up"] = (
            (df["high"] > df["high"].rolling(20).max().shift(1))
            & (df["close"] < df["open"])
            & (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
        )

        df["liquidity_sweep_down"] = (
            (df["low"] < df["low"].rolling(20).min().shift(1))
            & (df["close"] > df["open"])
            & (df["volume"] > df["volume"].rolling(20).mean() * 1.5)
        )

        # Manipulation Zone Detection
        df["manipulation_zone"] = (
            df["liquidity_sweep_up"] | df["liquidity_sweep_down"]
        ) & (df["volume"] > df["volume"].rolling(20).mean() * 2)

        # Stop Hunt Levels
        df["stop_hunt_level"] = np.where(
            df["liquidity_sweep_up"],
            df["high"],
            np.where(df["liquidity_sweep_down"], df["low"], np.nan),
        )

        return df

    except Exception as e:
        logger.error(f"Error adding liquidity analysis features: {str(e)}")
        return df

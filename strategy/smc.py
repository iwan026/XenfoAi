import pandas as pd

def add_smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add Smart Money Concepts analysis features"""
    try:
        # Fair Value Gaps Detection
        df["fvg_up"] = (df["low"].shift(2) > df["high"].shift(1)) & (
            df["low"] > df["high"].shift(1)
        )
        df["fvg_down"] = (df["high"].shift(2) < df["low"].shift(1)) & (
            df["high"] < df["low"].shift(1)
        )

        # Gap size measurement
        df["fvg_up_size"] = np.where(
            df["fvg_up"], df["low"].shift(2) - df["high"].shift(1), 0
        )
        df["fvg_down_size"] = np.where(
            df["fvg_down"], df["low"].shift(1) - df["high"].shift(2), 0
        )

        # Inefficient Price Movement Detection
        df["price_inefficiency"] = abs(df["close"] - df["open"]) / (
            df["high"] - df["low"] + 1e-7
        )

        # Institutional Money Flow
        df["money_flow"] = (
            ((df["close"] - df["low"]) - (df["high"] - df["close"]))
            / (df["high"] - df["low"] + 1e-7)
            * df["volume"]
        )
        df["inst_money_flow"] = df["money_flow"].rolling(window=20).sum()

        # Smart Money Divergence
        df["price_delta"] = df["close"] - df["close"].shift(1)
        df["volume_delta"] = df["volume"] - df["volume"].shift(1)
        df["smart_money_div"] = np.where(
            (df["price_delta"] > 0) & (df["volume_delta"] < 0),
            -1,  # Potential distribution
            np.where(
                (df["price_delta"] < 0) & (df["volume_delta"] < 0),
                1,  # Potential accumulation
                0,
            ),
        )

        return df

    except Exception as e:
        logger.error(f"Error adding smart money features: {str(e)}")
        return df

import pandas as pd

def add_volume_profile_features(
    self, df: pd.DataFrame, n_levels: int = 10
) -> pd.DataFrame:
    """Add Volume Profile analysis features"""
    try:
        # Calculate price levels
        df["price_range"] = (
            df["high"].rolling(window=20).max() - df["low"].rolling(window=20).min()
        )
        level_height = df["price_range"] / n_levels

        # Volume distribution per level
        for i in range(n_levels):
            level_low = df["low"] + (i * level_height)
            level_high = level_low + level_height

            # Calculate volume at each price level
            df[f"vol_level_{i}"] = np.where(
                (df["low"] <= level_high) & (df["high"] >= level_low), df["volume"], 0
            )

        # Point of Control (POC)
        volume_cols = [f"vol_level_{i}" for i in range(n_levels)]
        df["poc_level"] = df[volume_cols].idxmax(axis=1)
        df["poc_strength"] = (
            df[volume_cols].max(axis=1) / df["volume"].rolling(20).mean()
        )

        # Value Area calculation (70% of volume)
        df["value_area_high"] = df["high"].rolling(20).quantile(0.85)
        df["value_area_low"] = df["low"].rolling(20).quantile(0.15)
        df["in_value_area"] = (df["close"] >= df["value_area_low"]) & (
            df["close"] <= df["value_area_high"]
        )

        # Volume Interest Zones
        df["high_volume_zone"] = df["volume"] > df["volume"].rolling(20).mean() * 1.5
        df["low_volume_zone"] = df["volume"] < df["volume"].rolling(20).mean() * 0.5

        # Volume Delta
        df["volume_delta"] = np.where(
            df["close"] > df["open"],
            df["volume"],  # Buying volume
            -df["volume"],  # Selling volume
        )
        df["cumulative_delta"] = df["volume_delta"].rolling(20).sum()

        # Volume Based Support/Resistance
        df["vol_support"] = (
            np.where(
                (df["volume"] > df["volume"].rolling(20).mean())
                & (df["close"] > df["open"]),
                df["low"],
                np.nan,
            )
            .rolling(20, min_periods=1)
            .mean()
        )

        df["vol_resistance"] = (
            np.where(
                (df["volume"] > df["volume"].rolling(20).mean())
                & (df["close"] < df["open"]),
                df["high"],
                np.nan,
            )
            .rolling(20, min_periods=1)
            .mean()
        )

        return df

    except Exception as e:
        logger.error(f"Error adding volume profile features: {str(e)}")
        return df

import pandas as pd
import numpy as np
import pandas_ta as ta
import talib


def add_support_resistance(data):
    """Detect support and resistance levels automatically based on given data."""
    # Step 1: Combine all 'high' and 'low' prices into a single list
    prices = data["high"].tolist() + data["low"].tolist()

    # Step 2: Create bins with a size of 0.01 from the minimum to maximum price
    min_price = min(prices)
    max_price = max(prices)
    bins = np.arange(min_price, max_price, 0.01)

    # Step 3: Count the frequency of prices falling into each bin
    frequency, _ = np.histogram(prices, bins=bins)

    # Step 4: Select bins with frequency >= 2 (or more)
    support_resistance_levels = []
    for i in range(len(frequency)):
        if frequency[i] >= 2:
            support_resistance_levels.append(bins[i])

    # Step 5: Sort the levels from low to high
    support_resistance_levels.sort()

    # Print and return the levels
    print("Support and Resistance Levels:", support_resistance_levels)
    return df

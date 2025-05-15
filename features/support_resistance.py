import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN


def add_support_resistance(
    data,
    lookback=100,
    zone_sensitivity=0.005,
    strength_threshold=3,
    num_touches=2,
    recency_weight=True,
):
    """
    Deteksi support dan resistance dengan pendekatan zone

    Args:
        data: DataFrame dengan kolom OHLC
        lookback: Jumlah periode untuk melihat ke belakang
        zone_sensitivity: Toleransi untuk menentukan area zone (dalam persentase)
        strength_threshold: Minimum kekuatan level untuk dipertimbangkan
        num_touches: Minimum jumlah sentuhan yang dibutuhkan untuk konfirmasi level
        recency_weight: Jika True, data terbaru akan diberi bobot lebih

    Returns:
        Tuple (List support, List resistance, DataFrame data dengan kolom baru)
    """
    df = data.copy()

    # Pastikan data urut berdasarkan waktu
    if "date" in df.columns or "time" in df.columns:
        date_col = "date" if "date" in df.columns else "time"
        df = df.sort_values(by=date_col)

    # 1. Temukan swing high dan swing lows menggunakan local extrema
    high_prices = df["high"].values
    low_prices = df["low"].values
    order = int(len(df) * 0.02)  # 2% dari data sebagai order untuk extrema
    order = max(5, min(order, 20))  # Between 5 and 20

    # Identifikasi local maxima dan minima
    high_idx = argrelextrema(high_prices, np.greater, order=order)[0]
    low_idx = argrelextrema(low_prices, np.less, order=order)[0]

    # 2. Buat array gabungan swing high dan low dengan bobot
    swings = []
    lookback = min(lookback, len(df))
    recent_df = df.iloc[-lookback:]

    # Tambahkan swing highs
    for idx in high_idx:
        if idx >= len(df) - lookback:  # Filter hanya swing dalam lookback period
            price = high_prices[idx]
            # Hitung bobot berdasarkan recency dan volume jika tersedia
            recency = 1
            if recency_weight:
                # Makin dekat dengan data terbaru, bobot makin tinggi (1.0 to 2.0)
                recency = 1 + (idx - (len(df) - lookback)) / lookback

            volume_weight = 1
            if "volume" in df.columns:
                volume = df.iloc[idx]["volume"]
                avg_volume = df["volume"].mean()
                volume_weight = max(0.5, min(2.0, volume / avg_volume))

            strength = recency * volume_weight
            swings.append(
                {"price": price, "type": "resistance", "strength": strength, "idx": idx}
            )

    # Tambahkan swing lows
    for idx in low_idx:
        if idx >= len(df) - lookback:  # Filter hanya swing dalam lookback period
            price = low_prices[idx]
            # Hitung bobot
            recency = 1
            if recency_weight:
                recency = 1 + (idx - (len(df) - lookback)) / lookback

            volume_weight = 1
            if "volume" in df.columns:
                volume = df.iloc[idx]["volume"]
                avg_volume = df["volume"].mean()
                volume_weight = max(0.5, min(2.0, volume / avg_volume))

            strength = recency * volume_weight
            swings.append(
                {"price": price, "type": "support", "strength": strength, "idx": idx}
            )

    if not swings:
        return [], [], df

    # 3. Clustering untuk menemukan zone
    swing_prices = np.array([s["price"] for s in swings]).reshape(-1, 1)

    # Tentukan eps berdasarkan range harga
    price_range = df["high"].max() - df["low"].min()
    eps = price_range * zone_sensitivity

    # Gunakan DBSCAN untuk clustering
    clustering = DBSCAN(eps=eps, min_samples=1).fit(swing_prices)

    # 4. Proses hasil clustering untuk mendapatkan zones
    cluster_labels = clustering.labels_
    unique_clusters = np.unique(cluster_labels)

    support_zones = []
    resistance_zones = []

    for cluster in unique_clusters:
        cluster_points = [
            swings[i] for i in range(len(swings)) if cluster_labels[i] == cluster
        ]

        if not cluster_points:
            continue

        # Hitung informasi zone
        prices = [p["price"] for p in cluster_points]
        zone_min = min(prices)
        zone_max = max(prices)

        # Hitung kekuatan zone (jumlah points dan average strength)
        points_count = len(cluster_points)
        avg_strength = sum(p["strength"] for p in cluster_points) / points_count
        total_strength = points_count * avg_strength

        # Tentukan tipe zone (support atau resistance)
        support_count = sum(1 for p in cluster_points if p["type"] == "support")
        resistance_count = points_count - support_count

        zone_info = {
            "min": zone_min,
            "max": zone_max,
            "mid": (zone_min + zone_max) / 2,
            "width": zone_max - zone_min,
            "strength": total_strength,
            "touches": points_count,
            "indices": [p["idx"] for p in cluster_points],
        }

        # Filter berdasarkan kekuatan dan jumlah sentuhan
        if total_strength >= strength_threshold and points_count >= num_touches:
            if support_count > resistance_count:
                support_zones.append(zone_info)
            else:
                resistance_zones.append(zone_info)

    # 5. Urutkan zones berdasarkan kekuatan
    support_zones.sort(key=lambda x: x["strength"], reverse=True)
    resistance_zones.sort(key=lambda x: x["strength"], reverse=True)

    # 6. Tambahkan kolom-kolom baru ke dataframe
    df["support_zones"] = None
    df["resistance_zones"] = None

    for i, row in df.iterrows():
        current_price = row["close"]

        # Cek apakah harga berada dalam zone support
        in_support = False
        for zone in support_zones:
            if zone["min"] * 0.998 <= current_price <= zone["max"] * 1.002:
                in_support = True
                df.at[i, "support_zones"] = zone["mid"]
                break

        # Cek apakah harga berada dalam zone resistance
        in_resistance = False
        for zone in resistance_zones:
            if zone["min"] * 0.998 <= current_price <= zone["max"] * 1.002:
                in_resistance = True
                df.at[i, "resistance_zones"] = zone["mid"]
                break

    return support_zones, resistance_zones, df


def plot_support_resistance_zones(df, support_zones, resistance_zones, figsize=(14, 7)):
    """
    Plot data dengan support dan resistance zones

    Args:
        df: DataFrame dengan data OHLC
        support_zones: List zone support
        resistance_zones: List zone resistance
        figsize: Tuple ukuran plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.figure(figsize=figsize)

    # Plot candlestick
    width = 0.4
    width2 = width / 2

    # Pastikan data urut berdasarkan waktu
    if "date" in df.columns or "time" in df.columns:
        date_col = "date" if "date" in df.columns else "time"
        df = df.sort_values(by=date_col)

    # Plot candlestick
    up = df[df["close"] >= df["open"]]
    down = df[df["close"] < df["open"]]

    # Buat x-axis sebagai index
    x = np.arange(len(df))

    # Plot candles
    plt.bar(
        x[up.index], up["close"] - up["open"], width, bottom=up["open"], color="green"
    )
    plt.bar(
        x[up.index], up["high"] - up["close"], width2, bottom=up["close"], color="green"
    )
    plt.bar(
        x[up.index], up["open"] - up["low"], width2, bottom=up["low"], color="green"
    )

    plt.bar(
        x[down.index],
        down["open"] - down["close"],
        width,
        bottom=down["close"],
        color="red",
    )
    plt.bar(
        x[down.index],
        down["high"] - down["open"],
        width2,
        bottom=down["open"],
        color="red",
    )
    plt.bar(
        x[down.index],
        down["close"] - down["low"],
        width2,
        bottom=down["low"],
        color="red",
    )

    # Plot support zones
    for zone in support_zones:
        plt.axhspan(zone["min"], zone["max"], alpha=0.2, color="green")
        plt.axhline(
            y=zone["mid"],
            alpha=0.8,
            color="green",
            linestyle="--",
            label=f"Support: {zone['mid']:.4f} (S:{zone['strength']:.1f})",
        )

    # Plot resistance zones
    for zone in resistance_zones:
        plt.axhspan(zone["min"], zone["max"], alpha=0.2, color="red")
        plt.axhline(
            y=zone["mid"],
            alpha=0.8,
            color="red",
            linestyle="--",
            label=f"Resistance: {zone['mid']:.4f} (S:{zone['strength']:.1f})",
        )

    plt.title("Chart dengan Support/Resistance Zones")
    plt.ylabel("Harga")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt


# Contoh penggunaan:
"""
# Load data
df = pd.read_csv('data/eurusd_h1.csv')

# Deteksi support dan resistance
support_zones, resistance_zones, df_with_zones = add_support_resistance(
    df, lookback=200, zone_sensitivity=0.0005, strength_threshold=2, num_touches=2
)

# Plot hasil
plot = plot_support_resistance_zones(df, support_zones, resistance_zones)
plot.show()
"""

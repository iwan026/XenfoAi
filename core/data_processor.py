import logging
import os
import pandas as pd

from features.support_resistance import add_support_resistance
from config import KonfigurasiPath

logger = logging.getLogger(__name__)


def preprocessing_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Proses ulang data mentah untuk training model"""
    try:
        symbol_upper = symbol.upper()
        raw_symbol_dir = os.path.join(
            KonfigurasiPath.DATASET_FOREX_RAW_DIR, symbol_upper
        )

        filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
        raw_data_path = os.path.join(raw_symbol_dir, filename)

        # Baca data CSV
        data = pd.read_csv(raw_data_path)
        
        # Cek dan konversi kolom tanggal jika ada
        if 'date' in data.columns or 'time' in data.columns:
            date_col = 'date' if 'date' in data.columns else 'time'
            data[date_col] = pd.to_datetime(data[date_col])
            
        # Pastikan format kolom OHLC telah sesuai
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Kolom {col} tidak ditemukan dalam data")
                return None
        
        # Deteksi support dan resistance zones
        support_zones, resistance_zones, data_with_zones = add_support_resistance(
            data, 
            lookback=200,             # Periode lookback
            zone_sensitivity=0.0005,  # Sensitivitas zona (0.05%)
            strength_threshold=2,     # Minimum kekuatan level
            num_touches=2,            # Minimum sentuhan
            recency_weight=True       # Bobot data terbaru
        )
        
        # Tambahkan informasi support resistance sebagai fitur
        data = data_with_zones
        
        # Tambahkan kolom untuk level dukungan dan resistance terdekat
        data['nearest_support'] = None
        data['nearest_resistance'] = None
        data['distance_to_support'] = None  
        data['distance_to_resistance'] = None
        
        # Hitung level terdekat untuk setiap baris data
        for i, row in data.iterrows():
            price = row['close']
            
            # Cari support terdekat di bawah harga saat ini
            supports_below = [zone for zone in support_zones if zone['mid'] < price]
            if supports_below:
                nearest_support = max(supports_below, key=lambda x: x['mid'])
                data.at[i, 'nearest_support'] = nearest_support['mid']
                data.at[i, 'distance_to_support'] = (price - nearest_support['mid']) / price
            
            # Cari resistance terdekat di atas harga saat ini
            resistances_above = [zone for zone in resistance_zones if zone['mid'] > price]
            if resistances_above:
                nearest_resistance = min(resistances_above, key=lambda x: x['mid'])
                data.at[i, 'nearest_resistance'] = nearest_resistance['mid']
                data.at[i, 'distance_to_resistance'] = (nearest_resistance['mid'] - price) / price
        
        # Hitung rasio risk/reward berdasarkan S/R terdekat
        data['risk_reward_ratio'] = data['distance_to_support'] / data['distance_to_resistance']
        
        # Bersihkan NaN values
        data = data.fillna(0)
        
        logger.info(f"Preprocessing data berhasil untuk {symbol}_{timeframe}")
        return data

    except Exception as e:
        logger.error(f"Gagal preprocessing data: {str(e)}")
        return None


def save_processed_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Simpan processed data ke direktori"""
    try:
        symbol_upper = symbol.upper()
        dir_path = os.path.join(
            KonfigurasiPath.DATASET_FOREX_PROCESSED_DIR, symbol_upper
        )
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
        file_path = os.path.join(dir_path, filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Data processed berhasil disimpan ke: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Gagal menyimpan preprocessing data: {str(e)}")
        return False


def visualize_support_resistance(symbol: str, timeframe: str, output_dir=None):
    """Visualisasikan support dan resistance pada chart"""
    try:
        # Load data
        processed_data = preprocessing_data(symbol, timeframe)
        if processed_data is None:
            logger.error("Tidak dapat memvisualisasikan - data tidak berhasil diproses")
            return False
            
        # Import fungsi plotting dari modul support_resistance
        from features.support_resistance import plot_support_resistance_zones
        
        # Deteksi support dan resistance lagi untuk mendapatkan zones
        support_zones, resistance_zones, _ = add_support_resistance(
            processed_data, 
            lookback=200, 
            zone_sensitivity=0.0005,
            strength_threshold=2,
            num_touches=2
        )
        
        # Plot chart
        plt = plot_support_resistance_zones(processed_data, support_zones, resistance_zones)
        
        # Simpan visualisasi jika output_dir ditentukan
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{symbol.lower()}_{timeframe.lower()}_sr_zones.png")
            plt.savefig(output_path)
            logger.info(f"Visualisasi disimpan ke: {output_path}")
        else:
            plt.show()
            
        return True
        
    except Exception as e:
        logger.error(f"Gagal memvisualisasikan support/resistance: {str(e)}")
        return False
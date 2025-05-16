import time
from typing import Dict, Any, Optional
from datetime import datetime
from config import CACHE_EXPIRY, MAX_CACHE_SIZE


class CacheManager:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Membuat cache key"""
        return f"{symbol}_{timeframe}"

    def get(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Mengambil data dari cache"""
        key = self.get_cache_key(symbol, timeframe)
        cache_data = self.cache.get(key)

        if cache_data is None:
            return None

        # Cek expiry
        if time.time() - cache_data["timestamp"] > CACHE_EXPIRY:
            del self.cache[key]
            return None

        return cache_data["data"]

    def set(self, symbol: str, timeframe: str, data: Dict) -> None:
        """Menyimpan data ke cache"""
        key = self.get_cache_key(symbol, timeframe)

        # Cek ukuran cache
        if len(self.cache) >= MAX_CACHE_SIZE:
            # Hapus cache terlama
            oldest_key = min(
                self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]

        self.cache[key] = {"data": data, "timestamp": time.time()}

    def clear(self) -> None:
        """Membersihkan cache"""
        self.cache.clear()

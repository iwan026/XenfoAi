import os
import psutil
import logging
from typing import Dict
from datetime import datetime
from config import MAX_MEMORY_USAGE, MAX_CPU_USAGE

logger = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> float:
        """Mendapatkan penggunaan memori dalam GB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
        except Exception as e:
            logger.error(f"Error mengukur penggunaan memori: {e}")
            return 0.0

    def get_cpu_usage(self) -> float:
        """Mendapatkan penggunaan CPU dalam persen"""
        try:
            return self.process.cpu_percent(interval=1)
        except Exception as e:
            logger.error(f"Error mengukur penggunaan CPU: {e}")
            return 0.0

    def check_resources(self) -> Dict:
        """Memeriksa penggunaan resource"""
        memory_usage = self.get_memory_usage()
        cpu_usage = self.get_cpu_usage()

        status = {
            "memory_ok": memory_usage < MAX_MEMORY_USAGE,
            "cpu_ok": cpu_usage < MAX_CPU_USAGE,
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not status["memory_ok"]:
            logger.warning(f"Penggunaan memori tinggi: {memory_usage:.2f}GB")
        if not status["cpu_ok"]:
            logger.warning(f"Penggunaan CPU tinggi: {cpu_usage:.2f}%")

        return status

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any


class DataProcessor:
    """Computations and caching - handles all data processing and caching operations"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def process_spectrum_data(self, spectrum_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process the raw spectrum data and return processed data."""
        '''processed_data = spectrum_data / np.max(spectrum_data)
        return processed_data, spectrum_data'''
        return spectrum_data #, spectrum_data

    def cache_data(self, key: str, data: Any):
        """Cache data with a specific key."""
        self.cache[key] = data

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data by key."""
        return self.cache.get(key)
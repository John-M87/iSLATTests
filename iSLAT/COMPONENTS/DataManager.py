#import iSLAT.COMPONENTS.FileHandling.FileHandler as FileHandler
from iSLAT.COMPONENTS.FileHandling.FileHandler import FileHandler
#import iSLAT.COMPONENTS.DataProcessing as DataProcessing
from iSLAT.COMPONENTS.DataProcessing.DataProcessor import DataProcessor
import pandas as pd

class DataManager:
    """This class uses file handling and data processing components to load and manage data."""

    def __init__(self, file_handler: FileHandler = None, data_processor: DataProcessor = None):
        self.file_handler = file_handler if file_handler is not None else FileHandler()
        self.data_processor = data_processor if data_processor is not None else DataProcessor()

    def read_user_settings(self, file_path: str):
        """Read user settings from a JSON file."""
        self.user_settings = self.file_handler.read_user_settings(file_path)
        return self.user_settings

    def check_Hitran(self):
        """Check if HITRAN data is available."""
        return self.file_handler.check_Hitran()
    
    def load_spectrum_file(self, file_path: str):
        """Load a spectrum file and return the data."""
        self.raw_spectral_data = self.file_handler.read_spectrum_file(file_path)
        return self.raw_spectral_data

    def process_loaded_files(self):
        """Process the loaded files using the data processor."""
        if hasattr(self, 'raw_spectral_data'):
            self.processed_data = self.data_processor.process_spectrum_data(self.raw_spectral_data)
            return self.processed_data
        else:
            raise ValueError("No spectral data loaded. Please load a spectrum file first.")
    
    def get_data(self):
        """Get the processed data."""
        if hasattr(self, 'processed_data'):
            #return self.processed_data
            return pd.DataFrame(self.processed_data, columns=['wave_data', 'flux_data'])
        else:
            raise ValueError("No processed data available. Please process the loaded files first.")
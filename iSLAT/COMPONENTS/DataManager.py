from iSLAT.COMPONENTS.FileHandling.FileHandler import FileHandler
from iSLAT.COMPONENTS.DataProcessing.DataProcessor import DataProcessor
from iSLAT.COMPONENTS.DataTypes import Molecule, Spectrum, MoleculeDict
import pandas as pd

class DataManager:
    """This class uses file handling and data processing components to load and manage data."""

    def __init__(self, file_handler: FileHandler = None, data_processor: DataProcessor = None):
        self.file_handler = file_handler if file_handler is not None else FileHandler()
        self.data_processor = data_processor if data_processor is not None else DataProcessor()
        self.molecules_dict = MoleculeDict()

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
    
    def add_molecule(self, molecule: Molecule):
        """Add a molecule to the molecules dictionary."""
        if isinstance(molecule, Molecule):
            self.molecules_dict.add_molecule(molecule)
        else:
            raise TypeError("Expected a Molecule instance.")
    
    def get_molecule(self, name: str) -> Molecule:
        """Get a molecule by name from the molecules dictionary."""
        if name in self.molecules_dict:
            return self.molecules_dict[name]
        else:
            raise KeyError(f"Molecule '{name}' not found in the dictionary.")
    
    def get_molecules(self) -> MoleculeDict:
        """Get all molecules in the molecules dictionary."""
        return self.molecules_dict
    
    def save_molecule_parameters(self, molecule: Molecule, file_path: str):
        """Save the parameters of a molecule to a CSV file."""
        if isinstance(molecule, Molecule):
            df = molecule.get_table_lines
            df.to_csv(file_path, index=False)
            return True
        else:
            raise TypeError("Expected a Molecule instance.")
    
    def load_molecule_parameters(self, file_path: str) -> Molecule:
        """Load the parameters of a molecule from a CSV file."""
        df = pd.read_csv(file_path)
        molecule = Molecule.from_table_lines(df)
        self.add_molecule(molecule)
        return molecule
    
    def get_main_directory(self):
        """Get the main directory for saving and loading files."""
        return self.file_handler.get_main_directory()
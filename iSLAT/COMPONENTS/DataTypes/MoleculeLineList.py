import numpy as np
import pandas as pd
from collections import namedtuple
from iSLAT.COMPONENTS.DataTypes.MoleculeLine import MoleculeLine
from iSLAT.COMPONENTS.FileHandling.molecular_data_reader import read_molecular_data

class MoleculeLineList:
    def __init__(self, molecule_id=None, filename=None, lines_data=None):
        """
        Initialize a MoleculeLineList object.

        Parameters
        ----------
        molecule_id : str
            Identifier for the molecule.
        lines_data : list of dict, optional
            List of dictionaries containing line data with keys like 'frequency', 'wavelength', 'intensity', etc.
        filename : str, optional
            Path to a .par file to read molecular data from.
        """
        self.molecule_id = molecule_id
        self.lines = []
        self.partition_function = None
        
        # Define namedtuple types for data structure
        self._partition_type = namedtuple('partition', ['t', 'q'])
        self._lines_type = namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                                               'e_up', 'e_low', 'g_up', 'g_low'])
        
        # Cache for namedtuple conversion to avoid repeated array creation
        self._lines_cache = None
        self._lines_cache_valid = False
        
        if filename:
            self._load_from_file(filename)
        elif lines_data:
            self.lines = [MoleculeLine(molecule_id, line_data) for line_data in lines_data]

    def _load_from_file(self, filename):
        """
        Load molecular data from a .par file using the FileHandling module.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
        """
        partition_function, lines_data = read_molecular_data(self.molecule_id, filename)
        #print("Partition function:", partition_function)
        #print("Lines data:", lines_data)
        self.partition_function = partition_function
        self.lines = [MoleculeLine(self.molecule_id, line_data) for line_data in lines_data]
        self._lines_cache_valid = False  # Invalidate cache when lines change
        #print("We loaded the lines!")

        '''# Convert lines to MoleculeLine objects and then to numpy array
        molecule_lines = [MoleculeLine(self.molecule_id, line_data) for line_data in lines_data]
        self.lines = np.array([line.get_ndarray() for line in molecule_lines])'''

    def get_ndarray(self):
        """
        Convert the line data to a numpy ndarray.

        Returns
        -------
        np.ndarray
            Numpy array containing the line data.
        """
        return np.array([line.get_ndarray() for line in self.lines])
    
    def get_pandas_table(self):
        """
        Get all lines as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all line data
        """
        if not self.lines:
            return pd.DataFrame()
        
        # Combine all individual line DataFrames
        line_dfs = [line.get_pandas_table() for line in self.lines]
        return pd.concat(line_dfs, ignore_index=True)
    
    def get_partition_table(self):
        """
        Get partition function as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing partition function data
        """
        if self.partition_function is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'Temperature': self.partition_function.t,
            'Partition_Function': self.partition_function.q
        })
    
    @property
    def partition(self):
        """Access to partition function"""
        return self.partition_function
    
    @property
    def num_lines(self):
        """Number of lines in the list"""
        return len(self.lines)
    
    # Compatibility properties for legacy code expecting MolData format
    @property 
    def lines_as_namedtuple(self):
        """Get all line data as a single namedtuple structure for compatibility"""
        if not self.lines:
            return self._lines_type([], [], [], [], [], [], [], [], [], [])
        
        # Use cache if valid
        if self._lines_cache_valid and self._lines_cache is not None:
            return self._lines_cache
        
        # Extract all line data into arrays
        nr = [line.nr for line in self.lines]
        lev_up = [line.lev_up for line in self.lines]
        lev_low = [line.lev_low for line in self.lines]
        lam = [line.lam for line in self.lines]
        freq = [line.freq for line in self.lines]
        a_stein = [line.a_stein for line in self.lines]
        e_up = [line.e_up for line in self.lines]
        e_low = [line.e_low for line in self.lines]
        g_up = [line.g_up for line in self.lines]
        g_low = [line.g_low for line in self.lines]
        
        self._lines_cache = self._lines_type(
            np.array(nr), np.array(lev_up), np.array(lev_low),
            np.array(lam), np.array(freq), np.array(a_stein),
            np.array(e_up), np.array(e_low), np.array(g_up), np.array(g_low)
        )
        self._lines_cache_valid = True
        
        return self._lines_cache
    
    @property
    def name(self):
        """Molecule name for compatibility"""
        return self.molecule_id
    
    def get_wavelengths(self):
        """
        Get wavelengths of all lines as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of wavelengths in microns
        """
        if not self.lines:
            return np.array([])
        return np.array([line.lam for line in self.lines])
    
    def get_frequencies(self):
        """
        Get frequencies of all lines as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of frequencies in Hz
        """
        if not self.lines:
            return np.array([])
        return np.array([line.freq for line in self.lines])
    
    def get_einstein_coefficients(self):
        """
        Get all Einstein A coefficients from the lines.
        
        Returns
        -------
        np.ndarray
            Array of Einstein A coefficients
        """
        return np.array([line.a_stein for line in self.lines])
    
    def get_upper_energies(self):
        """
        Get all upper level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level energies in K
        """
        return np.array([line.e_up for line in self.lines])
    
    def get_lower_energies(self):
        """
        Get all lower level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level energies in K
        """
        return np.array([line.e_low for line in self.lines])
    
    def get_upper_weights(self):
        """
        Get all upper level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level statistical weights
        """
        return np.array([line.g_up for line in self.lines])
    
    def get_lower_weights(self):
        """
        Get all lower level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level statistical weights
        """
        return np.array([line.g_low for line in self.lines])
    
    @property
    def fname(self):
        """File name for compatibility with old MolData interface"""
        return getattr(self, '_filename', None)
    
    @fname.setter
    def fname(self, value):
        """Set file name"""
        self._filename = value
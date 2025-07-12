import numpy as np
import pandas as pd

class MoleculeLine:
    """
    Efficient representation of a single molecular line.
    
    Uses __slots__ for memory efficiency and direct attribute access for speed.
    """
    __slots__ = ('molecule_id', 'nr', 'lev_up', 'lev_low', 'lam', 'freq', 
                 'a_stein', 'e_up', 'e_low', 'g_up', 'g_low')
    
    def __init__(self, molecule_id, line_data, **kwargs):
        """
        Initialize a MoleculeLine object.
        
        Parameters
        ----------
        molecule_id : str
            Identifier for the molecule.
        line_data : dict
            Dictionary containing line data with keys like 'frequency', 'wavelength', 'intensity', etc.
        """
        self.molecule_id = molecule_id
        
        # Direct attribute assignment for better performance
        self.nr = line_data.get('nr', None)
        self.lev_up = line_data.get('lev_up', None)
        self.lev_low = line_data.get('lev_low', None)
        self.lam = line_data.get('lam', None)
        self.freq = line_data.get('freq', None)
        self.a_stein = line_data.get('a_stein', None)
        self.e_up = line_data.get('e_up', None)
        self.e_low = line_data.get('e_low', None)
        self.g_up = line_data.get('g_up', None)
        self.g_low = line_data.get('g_low', None)

    def get_ndarray(self):
        """
        Convert the line data to a numpy ndarray.

        Returns
        -------
        np.ndarray
            Numpy array containing the line data.
        """
        return np.array([self.nr, self.lev_up, self.lev_low, self.lam, self.freq,
                         self.a_stein, self.e_up, self.e_low, self.g_up, self.g_low])

    def get_pandas_table(self):
        """
        Convert the line data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the line data.
        """
        return pd.DataFrame({
            'nr': [self.nr],
            'lev_up': [self.lev_up],
            'lev_low': [self.lev_low],
            'lam': [self.lam],
            'freq': [self.freq],
            'a_stein': [self.a_stein],
            'e_up': [self.e_up],
            'e_low': [self.e_low],
            'g_up': [self.g_up],
            'g_low': [self.g_low]
        })
    
    @property
    def line_data(self):
        """
        Compatibility property that returns a namedtuple-like object for legacy code.
        
        Returns
        -------
        object
            Object with namedtuple-like attribute access
        """
        return LineDataView(self)
    
    def __str__(self):
        return f"MoleculeLine(molecule={self.molecule_id}, lam={self.lam}, freq={self.freq})"

    def __repr__(self):
        return self.__str__()


class LineDataView:
    """
    A lightweight view object that provides namedtuple-like access to MoleculeLine data.
    Used for backward compatibility without the overhead of creating actual namedtuples.
    """
    __slots__ = ('_line',)
    
    def __init__(self, line):
        self._line = line
    
    @property
    def nr(self):
        return self._line.nr
    
    @property
    def lev_up(self):
        return self._line.lev_up
    
    @property
    def lev_low(self):
        return self._line.lev_low
    
    @property
    def lam(self):
        return self._line.lam
    
    @property
    def freq(self):
        return self._line.freq
    
    @property
    def a_stein(self):
        return self._line.a_stein
    
    @property
    def e_up(self):
        return self._line.e_up
    
    @property
    def e_low(self):
        return self._line.e_low
    
    @property
    def g_up(self):
        return self._line.g_up
    
    @property
    def g_low(self):
        return self._line.g_low
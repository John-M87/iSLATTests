import numpy as np
import pandas as pd
from collections import namedtuple

class MoleculeLine:
    def __init__(self, molecule_id, line_data, **kwargs):
        """
        Initialize a MoleculeLine object.
        This class represents a single line of molecular data, including its properties such as frequency,
        wavelength, intensity, etc.

        Parameters
        ----------
        molecule_id : str
            Identifier for the molecule.
        line_data : dict
            Dictionary containing line data with keys like 'frequency', 'wavelength', 'intensity', etc.
        """
        self.molecule_id = molecule_id
        #self.line_data = pd.Series(line_data)
        self.partition_type = namedtuple('partition', ['t', 'q'])
        self.lines_type = namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                                                'e_up', 'e_low', 'g_up', 'g_low'])

        self.line_data = self.lines_type(
            nr=line_data.get('nr', None),
            lev_up=line_data.get('lev_up', None),
            lev_low=line_data.get('lev_low', None),
            lam=line_data.get('lam', None),
            freq=line_data.get('freq', None),
            a_stein=line_data.get('a_stein', None),
            e_up=line_data.get('e_up', None),
            e_low=line_data.get('e_low', None),
            g_up=line_data.get('g_up', None),
            g_low=line_data.get('g_low', None)
        )

    def get_ndarray(self):
        """
        Convert the line data to a numpy ndarray.

        Returns
        -------
        np.ndarray
            Numpy array containing the line data.
        """
        return np.array([self.line_data.nr,
                         self.line_data.lev_up,
                         self.line_data.lev_low,
                         self.line_data.lam,
                         self.line_data.freq,
                         self.line_data.a_stein,
                         self.line_data.e_up,
                         self.line_data.e_low,
                         self.line_data.g_up,
                         self.line_data.g_low])

    def get_pandas_table(self):
        """
        Convert the line data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the line data.
        """
        return pd.DataFrame({
            'nr': [self.line_data.nr],
            'lev_up': [self.line_data.lev_up],
            'lev_low': [self.line_data.lev_low],
            'lam': [self.line_data.lam],
            'freq': [self.line_data.freq],
            'a_stein': [self.line_data.a_stein],
            'e_up': [self.line_data.e_up],
            'e_low': [self.line_data.e_low],
            'g_up': [self.line_data.g_up],
            'g_low': [self.line_data.g_low]
        })
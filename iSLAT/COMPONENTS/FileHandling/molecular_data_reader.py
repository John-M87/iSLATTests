# -*- coding: utf-8 -*-

"""
Molecular Data Reader Module

This module provides functionality to read molecular data from .par files,
replacing the functionality previously in the MolData class.

The .par file format:
- First line: number of partition function values
- Next N lines: partition function (temperature vs Q(temperature))
- Next line: number of lines
- Following lines: molecular line data in fixed format

- 01/06/2020: SB, initial version (as MolData)
- 12/07/2025: Refactored into separate module for better organization
"""

import numpy as np
from collections import namedtuple

try:
    import pandas as pd
except ImportError:
    pd = None

__all__ = ["read_molecular_data", "MolecularDataReader"]


def read_molecular_data(molecule_name, filename):
    """
    Read molecular data from a .par file and return partition function and lines data.
    
    Parameters
    ----------
    molecule_name : str
        Name of the molecule (e.g., "CO")
    filename : str
        Path to the .par file
        
    Returns
    -------
    tuple
        (partition_function, lines_data) where:
        - partition_function: namedtuple with temperature and Q values
        - lines_data: list of dictionaries containing line data
    """
    reader = MolecularDataReader()
    return reader.read_par_file(filename)


class MolecularDataReader:
    """
    A utility class for reading molecular data files.
    
    This class provides methods for reading various molecular data formats.
    Currently supports .par files in the format used by the original Fortran 90 code.
    """
    
    def __init__(self):
        # Define namedtuple types for data structure
        self._partition_type = namedtuple('partition', ['t', 'q'])
        self._lines_type = namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                                               'e_up', 'e_low', 'g_up', 'g_low'])

    def read_par_file(self, filename):
        """
        Read molecular data from a .par file.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
            
        Returns
        -------
        tuple
            (partition_function, lines_data) where:
            - partition_function: namedtuple with temperature and Q values
            - lines_data: list of dictionaries containing line data
            
        Notes
        -----
        The file format of .par file is as follows:

        # Some comments, followed by the number of partition function values
        2933
        # Some further comments, followed by the partition function (temperature vs Q(temperature))
        70.0    25.544319
        71.0    25.904587
        ...
        # Some further comments, followed by the number of lines
        754
        # Some further comments followed by the lines
        1      1_R_13         1_R_13  196.29472  1527.25685208   1.3500e-04     3565.78714     3492.49079   58.0   54.0
        2      0_R_13         0_R_13  194.54560  1540.98815610   2.3660e-04      554.97254      481.01720   58.0   54.0

        Important: The lines section on the bottom is *fixed format*:
        - Nr: Index of the line - integer, 6 characters
        - Lev_up: Upper level label - string, 15 characters
        - Lev_low: Lower level label - string, 15 characters
        - Lam: Wavelength in micron - float, 11 characters
        - Freq: Frequency in GHz - float, 15 characters
        - A_stein: Einstein-A in s**-1 - float, 13 characters
        - E_up: Upper level energy in K - float, 15 characters
        - E_low: Lower level energy in K - float, 15 characters
        - g_up: Upper level statistical weight - float, 7 characters
        - g_low: Lower level statistical weight - float, 7 characters
        """
        with open(filename, "r") as f:
            # Read all lines at once and process them in a single pass
            data_clean = []
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    data_clean.append(stripped)

        # Split file into partition function and line section
        n_partition = int(data_clean[0])

        data_q = data_clean[1:n_partition + 1]
        data_lines = data_clean[n_partition + 1:]

        # Read partition function
        partition_function = self._read_partition_function(data_q)
        
        # Read lines
        lines_data = self._read_lines_data(data_lines)
        
        return partition_function, lines_data

    def _read_partition_function(self, data_q):
        """
        Read partition function data from raw file lines.
        
        Parameters
        ----------
        data_q : list
            List of strings containing partition function data
            
        Returns
        -------
        namedtuple
            Partition function with temperature and Q values
        """
        N = np.genfromtxt(data_q, dtype="f8,f8")
        q_temperature, q = [N[field] for field in N.dtype.names]
        return self._partition_type(q_temperature, q)

    def _read_lines_data(self, data_lines):
        """
        Read molecular lines data from raw file lines.
        
        Parameters
        ----------
        data_lines : list
            List of strings containing molecular lines data
            
        Returns
        -------
        list
            List of dictionaries containing line data
        """
        # Skip_header is to skip the number of lines printed in the file
        M = np.genfromtxt(data_lines, dtype="i4,S30,S30,f8,f8,f8,f8,f8,f8,f8", skip_header=2,
                          delimiter=(6, 30, 30, 11, 15, 13, 15, 15, 7, 7))

        nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low = [M[field] for field in M.dtype.names]

        # Decode and strip string fields more efficiently
        lev_up = [x.decode().strip() for x in lev_up]
        lev_low = [x.decode().strip() for x in lev_low]

        # Convert frequency from GHz to Hz (vectorized operation)
        freq = freq * 1e9

        # Create list of line data dictionaries using list comprehension
        lines_data = [
            {
                'nr': nr[i],
                'lev_up': lev_up[i],
                'lev_low': lev_low[i],
                'lam': lam[i],
                'freq': freq[i],
                'a_stein': a_stein[i],
                'e_up': e_up[i],
                'e_low': e_low[i],
                'g_up': g_up[i],
                'g_low': g_low[i]
            }
            for i in range(len(nr))
        ]
            
        return lines_data

    @staticmethod
    def _decode_strip(x):
        """Decode and strip bytes to string"""
        return x.decode().strip()
    
    @staticmethod
    def validate_par_file(filename):
        """
        Validate that a .par file has the correct format.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
            
        Returns
        -------
        bool
            True if file format is valid, False otherwise
        """
        try:
            with open(filename, "r") as f:
                data_raw = f.readlines()
            
            # Remove comments and empty lines
            data_clean = list(filter(lambda x: len(x.strip()) > 0 and x.strip()[0] != "#", data_raw))
            
            if len(data_clean) < 3:  # Need at least partition count, one partition entry, and line count
                return False
                
            # Check if first line is a number (partition function count)
            try:
                n_partition = int(data_clean[0])
            except ValueError:
                return False
                
            # Check if we have enough lines for partition function
            if len(data_clean) < n_partition + 2:  # +2 for partition count and line count
                return False
                
            return True
            
        except (IOError, OSError):
            return False
    
    @staticmethod
    def get_file_info(filename):
        """
        Get basic information about a .par file without fully loading it.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
            
        Returns
        -------
        dict
            Dictionary containing file information: 
            {'num_partition_points': int, 'num_lines': int, 'valid': bool}
        """
        info = {'num_partition_points': 0, 'num_lines': 0, 'valid': False}
        
        try:
            with open(filename, "r") as f:
                data_raw = f.readlines()
            
            # Remove comments and empty lines
            data_clean = list(filter(lambda x: len(x.strip()) > 0 and x.strip()[0] != "#", data_raw))
            
            if len(data_clean) < 3:
                return info
                
            # Get partition function count
            try:
                n_partition = int(data_clean[0])
                info['num_partition_points'] = n_partition
            except ValueError:
                return info
                
            # Get line count
            if len(data_clean) > n_partition + 1:
                try:
                    n_lines = int(data_clean[n_partition + 1])
                    info['num_lines'] = n_lines
                    info['valid'] = True
                except ValueError:
                    pass
                    
        except (IOError, OSError):
            pass
            
        return info

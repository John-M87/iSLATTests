# -*- coding: utf-8 -*-

"""
The class Intensity calculates the intensities

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

"""

import numpy as np
from scipy.integrate import fixed_quad
from typing import Optional, Union, Literal, TYPE_CHECKING
import pandas as pd

from ..MoleculeLineList import MoleculeLineList
import iSLAT.Constants as c

__all__ = ["Intensity"]

class Intensity:
    def __init__(self, molecule_line_list: MoleculeLineList) -> None:
        """Initialize an intensity class which calculates the intensities for a given molecule and provided
        physical parameters.

        Parameters
        ----------
        molecule_line_list: MoleculeLineList
            Molecular line list data to calculate the intensity
        """

        self._molecule: MoleculeLineList = molecule_line_list
        self._intensity: Optional[np.ndarray] = None
        self._tau: Optional[np.ndarray] = None
        self._t_kin: Optional[float] = None
        self._n_mol: Optional[float] = None
        self._dv: Optional[float] = None

    @staticmethod
    def _bb(nu: np.ndarray, T: float) -> np.ndarray:
        """Blackbody function for one temperature and an array of frequencies. Uses the short and long wavelength
        approximations for accuracy.

        Parameters
        ----------
        nu: np.ndarray
            Frequency in Hz to calculate the blackbody values
        T: float
            Temperature in K

        Returns
        -------
        np.ndarray:
            Blackbody intensity in erg/s/cm**2/sr/Hz
        """

        x = c.PLANCK_CONSTANT * nu / (c.BOLTZMANN_CONSTANT * T)
        bb_RJ = np.where(x < 1.0e-5, 2.0 * nu ** 2 * c.BOLTZMANN_CONSTANT * T / (c.SPEED_OF_LIGHT_CGS ** 2), 0.0)
        bb_Wien = np.where(x > 20.0, 2.0 * c.PLANCK_CONSTANT * nu ** 3 / c.SPEED_OF_LIGHT_CGS ** 2 * np.exp(-x), 0.0)
        bb_Planck = np.where((x >= 1.e-5) * (x <= 20.0),
                             2. * c.PLANCK_CONSTANT * nu ** 3 / c.SPEED_OF_LIGHT_CGS ** 2 * 1. / (np.exp(np.where(x <= 20.0, x, 20.0)) - 1.), 0.0)

        return bb_RJ + bb_Wien + bb_Planck

    @staticmethod
    def _fint(tau: np.ndarray) -> np.ndarray:
        """Evaluates the integral in Eq. A1 of Banzatti et al. 2012 for an array of tau values.

        To calculate the integral

        int 1.0 - exp(-tau*exp(-x**2)) dx

        for all values of tau simultaneously, we cannot use an adaptive algorithm such as quad in scipy.integrate.
        Thus, the same approach as in the Fortran 90 code is used and the integral is evaluated using a 20-point
        Gaussian quadrature.

        Parameters
        ----------
        tau: np.ndarray
            Array with tau values

        Returns
        -------
        np.ndarray:
            Array with the integral values
        """

        tau_param = np.array(tau).repeat(20).reshape(-1, 20)

        i, _ = fixed_quad(lambda x: 1.0 - np.exp(-tau_param * np.exp(-x ** 2)), -6, 6, n=20)

        return i

    def calc_intensity(self, t_kin: Optional[float] = None, n_mol: Optional[float] = None, 
                      dv: Optional[float] = None, method: Literal["curve_growth", "radex"] = "curve_growth") -> None:
        """Calculate the intensity for a given set of physical parameters. This implements Eq. A1 and A2 in
        Banzatti et al. 2012.

        The calculation method to obtain the intensity from the opacity can be switched between the curve-of-growth
        method used in Banzatti et al. 2012, which considers broadening for high values of tau and the simple expression
        used e.g. in RADEX (van der Tak et al. 2007) which saturates at tau ~ few. For low values (tau < 1), they
        yield the same values.

        Parameters
        ----------
        t_kin: float, optional
            Kinetic temperature in K
        n_mol: float, optional
            Column density in cm**-2
        dv: float, optional
            Intrinsic (turbulent) line width in km/s
        method: Literal["curve_growth", "radex"], default "curve_growth"
            Calculation method, either "curve_growth" for Eq. A1 or "radex" for less accurate approximation
        """

        self._t_kin = t_kin
        self._n_mol = n_mol
        self._dv = dv

        m = self._molecule

        # 1. calculate the partition function
        if t_kin < np.min(m.partition.t) or t_kin > np.max(m.partition.t):
            raise ValueError('t_kin outside range of partition function')

        q_sum = np.interp(t_kin, m.partition.t, m.partition.q)

        # 2. line opacity
        lines = m.lines_as_namedtuple
        
        '''# Debug: Check input data quality
        print(f"Lines data quality check:")
        print(f"  g_low - finite: {np.sum(np.isfinite(lines.g_low))}, NaN: {np.sum(np.isnan(lines.g_low))}, total: {len(lines.g_low)}")
        print(f"  g_up - finite: {np.sum(np.isfinite(lines.g_up))}, NaN: {np.sum(np.isnan(lines.g_up))}, total: {len(lines.g_up)}")
        print(f"  e_low - finite: {np.sum(np.isfinite(lines.e_low))}, NaN: {np.sum(np.isnan(lines.e_low))}, total: {len(lines.e_low)}")
        print(f"  e_up - finite: {np.sum(np.isfinite(lines.e_up))}, NaN: {np.sum(np.isnan(lines.e_up))}, total: {len(lines.e_up)}")
        print(f"  a_stein - finite: {np.sum(np.isfinite(lines.a_stein))}, NaN: {np.sum(np.isnan(lines.a_stein))}, total: {len(lines.a_stein)}")
        print(f"  freq - finite: {np.sum(np.isfinite(lines.freq))}, NaN: {np.sum(np.isnan(lines.freq))}, total: {len(lines.freq)}")
        
        #print specfic lines that have nan intries
        print("  Lines with NaN values:")
        for i, line in enumerate(m.lines):
            if not np.isfinite(line.g_low) or not np.isfinite(line.g_up) or not np.isfinite(line.e_low) or \
               not np.isfinite(line.e_up) or not np.isfinite(line.a_stein) or not np.isfinite(line.freq):
                print(f"    Line {i}: g_low={line.g_low}, g_up={line.g_up}, e_low={line.e_low}, e_up={line.e_up}, "
                      f"a_stein={line.a_stein}, freq={line.freq}")'''

        x_low = lines.g_low * np.exp(-lines.e_low / t_kin) / q_sum
        x_up = lines.g_up * np.exp(-lines.e_up / t_kin) / q_sum

        # Debug: Check intermediate calculations
        #print(f"  x_low - finite: {np.sum(np.isfinite(x_low))}, NaN: {np.sum(np.isnan(x_low))}")
        #print(f"  x_up - finite: {np.sum(np.isfinite(x_up))}, NaN: {np.sum(np.isnan(x_up))}")

        # Eq. A2 of Banzatti et al. 2012
        tau = lines.a_stein * c.SPEED_OF_LIGHT_CGS ** 3 / (8.0 * np.pi * lines.freq ** 3 * 1e5 * dv * c.FGAUSS_PREFACTOR) * n_mol \
            * (x_low * lines.g_up / lines.g_low - x_up)
        
        #print(f"  tau - finite: {np.sum(np.isfinite(tau))}, NaN: {np.sum(np.isnan(tau))}")
        
        # Check for division by zero in g_low
        zero_g_low = np.sum(lines.g_low == 0)
        if zero_g_low > 0:
            print(f"  Warning: {zero_g_low} lines have g_low = 0, which will cause division by zero")

        # 3. line intensity
        if method == "radex":
            intensity = c.FGAUSS_PREFACTOR * (1e5 * dv) * lines.freq / c.SPEED_OF_LIGHT_CGS * self._bb(lines.freq, t_kin) * \
                        (1.0 - np.exp(-tau))
        elif method == "curve_growth":
            # Eq. A1 of Banzatti et al. 2012
            intensity = 1.0 / (2.0 * np.sqrt(np.log(2.0))) * (1e5 * dv) * lines.freq / c.SPEED_OF_LIGHT_CGS * \
                        self._bb(lines.freq, t_kin) * self._fint(tau)
        else:
            raise ValueError("Intensity calculation method not known")

        #print(f"  Final intensity - finite: {np.sum(np.isfinite(intensity))}, NaN: {np.sum(np.isnan(intensity))}")

        self._tau = tau
        self._intensity = intensity

    def get_table_in_range(self, lam_min: float, lam_max: float) -> "pd.DataFrame":
        """Get a table with the lines in the specified wavelength range.

        Parameters
        ----------
        lam_min: float
            Minimum wavelength in microns
        lam_max: float
            Maximum wavelength in microns

        Returns
        -------
        pd.DataFrame:
            Dataframe with the lines in the specified range
        """

        if pd is None:
            raise ImportError("Pandas required to create table")

        mask = (self.molecule.lines_as_namedtuple.lam >= lam_min) & (self.molecule.lines_as_namedtuple.lam <= lam_max)
        return self.get_table[mask]

    @property
    def tau(self) -> Optional[np.ndarray]:
        """np.ndarray: Opacities per line"""
        return self._tau

    @property
    def intensity(self) -> Optional[np.ndarray]:
        """np.ndarray: Calculated intensity per line in erg/s/cm**2/sr/Hz"""
        return self._intensity

    @property
    def molecule(self) -> MoleculeLineList:
        """MoleculeLineList: Molecular line list data used for calculation"""
        return self._molecule

    @property
    def t_kin(self) -> Optional[float]:
        """float: Kinetic temperature in K used for calculation"""
        return self._t_kin

    @property
    def n_mol(self) -> Optional[float]:
        """float: Molecular column density in cm**-2 used for calculation"""
        return self._n_mol

    @property
    def dv(self) -> Optional[float]:
        """float: Line width in km/s used for calculation"""
        return self._dv

    def __repr__(self) -> str:
        return f"Intensity(Mol-Name={self.molecule.name}, t_kin={self.t_kin} n_mol={self.n_mol} dv={self.dv}, " \
               f"tau={self.tau}, intensity={self.intensity})"

    @property
    def get_table(self) -> "pd.DataFrame":
        """pd.DataFrame: Pandas dataframe with line data"""

        if pd is None:
            raise ImportError("Pandas required to create table")

        lines = self.molecule.lines_as_namedtuple
        return pd.DataFrame({'lev_up': lines.lev_up,
                             'lev_low': lines.lev_low,
                             'lam': lines.lam,
                             'tau': self.tau,
                             'intens': self.intensity,
                             'a_stein': lines.a_stein,
                             'e_up': lines.e_up,
                             'g_up': lines.g_up})

    def _repr_html_(self) -> Optional[str]:
        # noinspection PyProtectedMember
        return self.get_table._repr_html_() if pd is not None else None
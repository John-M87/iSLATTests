import os
import pandas as pd
import numpy as np
from scipy.optimize import fmin
from astropy.io import ascii
from iSLAT.ir_model import Chi2Spectrum, MolData, Intensity, Spectrum


class Config:
    """Configuration class for slab fitting parameters."""
    
    def __init__(self, target, save_folder, mol, molpath, dist, fwhm, min_lamb, max_lamb, pix_per_fwhm, intrinsic_line_width, cc):
        self.target = target
        self.input_file = os.path.join(save_folder, target)
        self.dist = dist
        self.fwhm = fwhm
        self.npix = pix_per_fwhm
        self.model_lam_min = min_lamb
        self.model_lam_max = max_lamb
        self.intrins_line_broad = intrinsic_line_width
        self.rings = 1
        self.cc = cc  # speed of light in km/s
        self.molecule_name = mol  # Add molecule_name attribute
        self.molecule_path = molpath  # Add molecule_name attribute

    @property
    def model_line_width(self):
        return self.cc / self.fwhm

    @property
    def model_pixel_res(self):
        return (self.model_lam_min + self.model_lam_max) / 2 / self.cc * self.fwhm / self.npix


class DataLoader:
    """Data loader class for loading spectroscopic data and molecular data."""
    
    def __init__(self, config):
        self.config = config
        self.chi2_h2o = Chi2Spectrum()
        self.mol_h2o = None
        self.intensity_h2o = None
        
    def load_data(self):
        """Load the spectroscopic data and initialize molecular data."""
        self.chi2_h2o.load_file(self.config.input_file)
        self.mol_h2o = MolData(f"{self.config.molecule_name}", f"{self.config.molecule_path}")
        self.intensity_h2o = Intensity(self.mol_h2o)


class ModelFitting:
    """Model fitting class for performing slab model optimization."""
    
    def __init__(self, data_loader, config, data_field=None):
        self.data_loader = data_loader
        self.config = config
        self.data_field = data_field

    def eval_function(self, t_kin, n_mol, radius):
        """
        Evaluate the chi-squared for given physical parameters.
        
        Parameters:
        -----------
        t_kin : float
            Kinetic temperature in K
        n_mol : float
            Molecular column density
        radius : float
            Emitting radius
            
        Returns:
        --------
        float
            Chi-squared value
        """
        intensity_h2o = self.data_loader.intensity_h2o
        chi2_h2o = self.data_loader.chi2_h2o
        intensity_h2o.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=self.config.intrins_line_broad)
        
        test_spectrum = Spectrum(
            lam_min=self.config.model_lam_min, lam_max=self.config.model_lam_max, 
            dlambda=self.config.model_pixel_res, R=self.config.model_line_width, 
            distance=self.config.dist)
        test_spectrum.add_intensity(intensity_h2o, radius**2 * np.pi)

        chi2_h2o.evaluate_spectrum(test_spectrum)
    
        print(f"For t_kin = {t_kin:.2f}, n_mol = {n_mol:.2e}, radius = {radius:.3f} chi2 = {chi2_h2o.chi2_total:.3e}")

        return chi2_h2o.chi2_total

    def fit_model(self, start_t, start_n_mol, start_r):
        """
        Fit the slab model using optimization.
        
        Parameters:
        -----------
        start_t : float
            Starting temperature guess
        start_n_mol : float
            Starting molecular column density guess
        start_r : float
            Starting radius guess
            
        Returns:
        --------
        array
            Optimized parameters [temperature, log10(n_mol), radius]
        """
        func = lambda p: self.eval_function(p[0], 10**p[1], p[2])
        x_0 = [start_t, np.log10(start_n_mol), start_r]
        result = fmin(func, x_0)
        return result
    
    def message(self):
        """Update the main GUI data field with fitting status."""
        if self.data_field is not None:
            self.data_field.clear()
            self.data_field.insert_text("Fitting slab...")


class SlabFit:
    """
    Main slab fitting class that combines all functionality.
    
    This class provides a unified interface for slab model fitting,
    combining configuration, data loading, and model fitting capabilities.
    """
    
    def __init__(self, target, save_folder, mol, molpath, dist, fwhm, min_lamb, max_lamb, 
                 pix_per_fwhm, intrinsic_line_width, cc, data_field=None):
        """
        Initialize the slab fitting system.
        
        Parameters:
        -----------
        target : str
            Target filename
        save_folder : str
            Directory containing the target file
        mol : str
            Molecule name
        molpath : str
            Path to molecular data
        dist : float
            Distance to object
        fwhm : float
            Full width at half maximum
        min_lamb : float
            Minimum wavelength
        max_lamb : float
            Maximum wavelength
        pix_per_fwhm : int
            Pixels per FWHM
        intrinsic_line_width : float
            Intrinsic line broadening
        cc : float
            Speed of light in km/s
        data_field : object, optional
            GUI data field for status updates
        """
        self.config = Config(target, save_folder, mol, molpath, dist, fwhm, min_lamb, max_lamb, 
                           pix_per_fwhm, intrinsic_line_width, cc)
        self.data_loader = DataLoader(self.config)
        self.model_fitting = ModelFitting(self.data_loader, self.config, data_field)
        
    def initialize(self):
        """Load all necessary data for fitting."""
        self.data_loader.load_data()
        
    def fit(self, start_t=500, start_n_mol=1e17, start_r=1.0):
        """
        Perform the slab model fitting.
        
        Parameters:
        -----------
        start_t : float, optional
            Starting temperature guess (default: 500 K)
        start_n_mol : float, optional
            Starting molecular column density guess (default: 1e17)
        start_r : float, optional
            Starting radius guess (default: 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing fitted parameters and results
        """
        # Update status if GUI field is available
        if self.model_fitting.data_field is not None:
            self.model_fitting.message()
            
        # Perform the fit
        result = self.model_fitting.fit_model(start_t, start_n_mol, start_r)
        
        # Return results in a structured format
        fitted_params = {
            'temperature': result[0],
            'log_n_mol': result[1],
            'n_mol': 10**result[1],
            'radius': result[2],
            'chi2_final': self.model_fitting.eval_function(result[0], 10**result[1], result[2])
        }
        
        return fitted_params
        
    def evaluate(self, t_kin, n_mol, radius):
        """
        Evaluate the model for given parameters without fitting.
        
        Parameters:
        -----------
        t_kin : float
            Kinetic temperature in K
        n_mol : float
            Molecular column density
        radius : float
            Emitting radius
            
        Returns:
        --------
        float
            Chi-squared value
        """
        return self.model_fitting.eval_function(t_kin, n_mol, radius)

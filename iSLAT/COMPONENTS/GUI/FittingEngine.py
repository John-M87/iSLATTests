"""
FittingEngine - LMFIT operations and model fitting functionality

This class handles all fitting operations including:
- Gaussian line fitting
- Multi-component fitting (deblending)
- Slab model fitting
- Chi-squared calculations
- Parameter estimation and uncertainty analysis
"""

import numpy as np
from datetime import datetime
from lmfit.models import GaussianModel, PseudoVoigtModel
from lmfit import Parameters, minimize, fit_report
from scipy.optimize import fmin
from scipy.signal import find_peaks
from iSLAT.ir_model import Chi2Spectrum, FluxMeasurement
from iSLAT.COMPONENTS.slabfit import SlabFit


class FittingEngine:
    """
    Centralized fitting engine for all LMFIT operations and model fitting.
    
    This class provides a unified interface for various fitting operations
    including line fitting, deblending, and slab model fitting.
    """
    
    def __init__(self, islat_instance):
        """
        Initialize the fitting engine.
        
        Parameters
        ----------
        islat_instance : iSLAT
            Reference to the main iSLAT instance for accessing data and configuration
        """
        self.islat = islat_instance
        self.last_fit_result = None
        self.last_fit_params = None
        self.fit_uncertainty = 1.0  # Default uncertainty factor
        
    def set_fit_uncertainty(self, uncertainty):
        """Set the uncertainty factor for fitting operations."""
        self.fit_uncertainty = uncertainty
    
    def fit_gaussian_line(self, wave_data, flux_data, xmin=None, xmax=None, 
                         initial_guess=None, deblend=False):
        """
        Fit a Gaussian model to spectral line data.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        xmin, xmax : float, optional
            Wavelength range for fitting. If None, uses full range
        initial_guess : dict, optional
            Initial parameter guesses {'center': float, 'amplitude': float, 'sigma': float}
        deblend : bool, optional
            If True, attempt to fit multiple components
            
        Returns
        -------
        fit_result : lmfit.ModelResult
            Fitting result object
        fitted_wave : array_like
            Wavelength array for fitted model
        fitted_flux : array_like
            Fitted flux values
        """
        # Apply wavelength range constraints
        if xmin is not None and xmax is not None:
            mask = (wave_data >= xmin) & (wave_data <= xmax)
            fit_wave = wave_data[mask]
            fit_flux = flux_data[mask]
        else:
            fit_wave = wave_data
            fit_flux = flux_data
            
        if len(fit_wave) < 3:
            raise ValueError("Insufficient data points for fitting")
        
        if deblend:
            return self._fit_multi_gaussian(fit_wave, fit_flux, initial_guess)
        else:
            return self._fit_single_gaussian(fit_wave, fit_flux, initial_guess)
    
    def _fit_single_gaussian(self, wave_data, flux_data, initial_guess=None):
        """Fit a single Gaussian component."""
        model = GaussianModel()
        
        # Generate initial parameter estimates
        if initial_guess is None:
            initial_guess = self._estimate_gaussian_params(wave_data, flux_data)
        
        params = model.make_params(
            center=initial_guess.get('center', wave_data[np.argmax(flux_data)]),
            amplitude=initial_guess.get('amplitude', np.max(flux_data)),
            sigma=initial_guess.get('sigma', (wave_data[-1] - wave_data[0]) / 10)
        )
        
        # Set reasonable bounds
        params['center'].set(min=wave_data.min(), max=wave_data.max())
        params['amplitude'].set(min=0)
        params['sigma'].set(min=1e-6, max=(wave_data[-1] - wave_data[0]))
        
        # Perform fit
        result = model.fit(flux_data, params, x=wave_data)
        
        # Generate fitted curve
        fitted_wave = np.linspace(wave_data.min(), wave_data.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def _fit_multi_gaussian(self, wave_data, flux_data, initial_guess=None):
        """Fit multiple Gaussian components for deblending."""
        # Estimate number of components based on peaks
        n_components = self._estimate_n_components(wave_data, flux_data)
        
        # Create composite model
        model = None
        params = Parameters()
        
        for i in range(n_components):
            prefix = f'g{i}_'
            if model is None:
                model = GaussianModel(prefix=prefix)
            else:
                model += GaussianModel(prefix=prefix)
            
            # Estimate parameters for each component
            component_guess = self._estimate_component_params(
                wave_data, flux_data, i, n_components
            )
            
            params.add(f'{prefix}center', value=component_guess['center'],
                      min=wave_data.min(), max=wave_data.max())
            params.add(f'{prefix}amplitude', value=component_guess['amplitude'], min=0)
            params.add(f'{prefix}sigma', value=component_guess['sigma'],
                      min=1e-6, max=(wave_data[-1] - wave_data[0]))
        
        # Perform fit
        result = model.fit(flux_data, params, x=wave_data)
        
        # Generate fitted curve
        fitted_wave = np.linspace(wave_data.min(), wave_data.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def _estimate_gaussian_params(self, wave_data, flux_data):
        """Estimate initial Gaussian parameters from data."""
        max_idx = np.argmax(flux_data)
        center = wave_data[max_idx]
        amplitude = flux_data[max_idx]
        
        # Estimate sigma from FWHM
        half_max = amplitude / 2
        indices = np.where(flux_data >= half_max)[0]
        if len(indices) > 1:
            fwhm = wave_data[indices[-1]] - wave_data[indices[0]]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = (wave_data[-1] - wave_data[0]) / 10
        
        return {'center': center, 'amplitude': amplitude, 'sigma': sigma}
    
    def _estimate_n_components(self, wave_data, flux_data):
        """Estimate number of Gaussian components needed."""
        # Simple peak finding - count local maxima
        
        peaks, _ = find_peaks(flux_data, height=np.max(flux_data) * 0.1)
        n_components = max(1, min(len(peaks), 3))  # Limit to 3 components
        
        return n_components
    
    def _estimate_component_params(self, wave_data, flux_data, component_idx, n_components):
        """Estimate parameters for a specific component in multi-component fit."""
        # Divide wavelength range into regions
        wave_range = wave_data[-1] - wave_data[0]
        region_size = wave_range / n_components
        region_start = wave_data[0] + component_idx * region_size
        region_end = region_start + region_size
        
        # Find peak in this region
        mask = (wave_data >= region_start) & (wave_data <= region_end)
        if np.any(mask):
            region_flux = flux_data[mask]
            region_wave = wave_data[mask]
            max_idx = np.argmax(region_flux)
            center = region_wave[max_idx]
            amplitude = region_flux[max_idx]
        else:
            center = region_start + region_size / 2
            amplitude = np.max(flux_data) / n_components
        
        sigma = region_size / 4  # Conservative estimate
        
        return {'center': center, 'amplitude': amplitude, 'sigma': sigma}
    
    def fit_voigt_profile(self, wave_data, flux_data, xmin=None, xmax=None):
        """
        Fit a Voigt profile to spectral line data.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        xmin, xmax : float, optional
            Wavelength range for fitting
            
        Returns
        -------
        fit_result : lmfit.ModelResult
            Fitting result object
        fitted_wave : array_like
            Wavelength array for fitted model
        fitted_flux : array_like
            Fitted flux values
        """
        # Apply wavelength range constraints
        if xmin is not None and xmax is not None:
            mask = (wave_data >= xmin) & (wave_data <= xmax)
            fit_wave = wave_data[mask]
            fit_flux = flux_data[mask]
        else:
            fit_wave = wave_data
            fit_flux = flux_data
        
        model = PseudoVoigtModel()
        
        # Initial parameter estimates
        initial_guess = self._estimate_gaussian_params(fit_wave, fit_flux)
        
        params = model.make_params(
            center=initial_guess['center'],
            amplitude=initial_guess['amplitude'],
            sigma=initial_guess['sigma'],
            fraction=0.5  # Start with equal Gaussian and Lorentzian components
        )
        
        # Set bounds
        params['center'].set(min=fit_wave.min(), max=fit_wave.max())
        params['amplitude'].set(min=0)
        params['sigma'].set(min=1e-6, max=(fit_wave[-1] - fit_wave[0]))
        params['fraction'].set(min=0, max=1)
        
        # Perform fit
        result = model.fit(fit_flux, params, x=fit_wave)
        
        # Generate fitted curve
        fitted_wave = np.linspace(fit_wave.min(), fit_wave.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def perform_slab_fit(self, target_file, molecule_name, 
                        start_temp=500, start_n_mol=1e17, start_radius=1.0):
        """
        Perform slab model fitting for a given target spectrum.
        
        Parameters
        ----------
        target_file : str
            Path to target spectrum file
        molecule_name : str
            Name of molecule to fit
        start_temp : float, optional
            Initial temperature guess (K)
        start_n_mol : float, optional
            Initial column density guess (cm^-2)
        start_radius : float, optional
            Initial radius guess (AU)
            
        Returns
        -------
        fit_result : dict
            Dictionary containing fitted parameters and statistics
        """
        try:
            # Get molecule data path
            mol_data = None
            for mol_info in self.islat.default_molecule_csv_data():
                if mol_info['name'] == molecule_name:
                    mol_data = mol_info
                    break
            
            if mol_data is None:
                raise ValueError(f"Molecule {molecule_name} not found in available molecules")
            
            # Create slab fit instance
            slab_fitter = SlabFit(
                target=target_file,
                save_folder="EXAMPLE-data",
                mol=molecule_name,
                molpath=mol_data['file'],
                dist=self.islat.global_dist,
                fwhm=self.islat.global_fwhm,
                min_lamb=self.islat.wavelength_range[0],
                max_lamb=self.islat.wavelength_range[1],
                pix_per_fwhm=10,
                intrinsic_line_width=self.islat.global_intrinsic_line_width,
                cc=3e8,
                data_field=getattr(self.islat.gui, 'data_field', None)
            )
            
            # Initialize and perform fit
            slab_fitter.initialize()
            result = slab_fitter.fit(start_temp, start_n_mol, start_radius)
            
            return result
            
        except Exception as e:
            print(f"Error in slab fitting: {str(e)}")
            return None
    
    def calculate_chi_squared(self, wave_obs, flux_obs, flux_error, 
                            wave_model, flux_model):
        """
        Calculate chi-squared statistic between observed and model data.
        
        Parameters
        ----------
        wave_obs : array_like
            Observed wavelength data
        flux_obs : array_like
            Observed flux data
        flux_error : array_like
            Flux uncertainties
        wave_model : array_like
            Model wavelength data
        flux_model : array_like
            Model flux data
            
        Returns
        -------
        chi2 : float
            Chi-squared statistic
        reduced_chi2 : float
            Reduced chi-squared statistic
        """
        # Interpolate model to observed wavelength grid
        flux_model_interp = np.interp(wave_obs, wave_model, flux_model)
        
        # Calculate chi-squared
        chi2 = np.sum(((flux_obs - flux_model_interp) / flux_error) ** 2)
        
        # Calculate reduced chi-squared (adjust for number of fitted parameters)
        n_params = len(self.last_fit_params) if self.last_fit_params else 0
        dof = len(wave_obs) - n_params
        reduced_chi2 = chi2 / dof if dof > 0 else chi2
        
        return chi2, reduced_chi2
    
    def get_fit_report(self):
        """
        Get a detailed report of the last fitting operation.
        
        Returns
        -------
        report : str
            Formatted fit report
        """
        if self.last_fit_result is None:
            return "No fitting results available."
        
        return fit_report(self.last_fit_result)
    
    def get_fit_statistics(self):
        """
        Get statistical information about the last fit.
        
        Returns
        -------
        stats : dict
            Dictionary containing fit statistics
        """
        if self.last_fit_result is None:
            return {}
        
        result = self.last_fit_result
        
        stats = {
            'chi_squared': result.chisqr,
            'reduced_chi_squared': result.redchi,
            'aic': result.aic,
            'bic': result.bic,
            'n_data': result.ndata,
            'n_variables': result.nvarys,
            'n_function_evals': result.nfev,
            'success': result.success,
            'method': result.method
        }
        
        return stats
    
    def extract_line_parameters(self):
        """
        Extract line parameters from the last fitting result.
        
        Returns
        -------
        line_params : dict
            Dictionary containing line parameters (center, amplitude, width, etc.)
        """
        if self.last_fit_result is None or self.last_fit_params is None:
            return {}
        
        params = self.last_fit_params
        line_params = {}
        
        # Extract parameters for single Gaussian fit
        if 'center' in params:
            line_params['center'] = params['center'].value
            line_params['center_stderr'] = params['center'].stderr
            line_params['amplitude'] = params['amplitude'].value
            line_params['amplitude_stderr'] = params['amplitude'].stderr
            line_params['sigma'] = params['sigma'].value
            line_params['sigma_stderr'] = params['sigma'].stderr
            
            # Calculate derived parameters
            line_params['fwhm'] = 2.355 * params['sigma'].value  # 2*sqrt(2*ln(2))
            line_params['area'] = np.sqrt(2 * np.pi) * params['amplitude'].value * params['sigma'].value
            
        # Extract parameters for multi-component fits
        component_idx = 0
        while f'g{component_idx}_center' in params:
            prefix = f'g{component_idx}_'
            component_params = {}
            
            component_params['center'] = params[f'{prefix}center'].value
            component_params['center_stderr'] = params[f'{prefix}center'].stderr
            component_params['amplitude'] = params[f'{prefix}amplitude'].value
            component_params['amplitude_stderr'] = params[f'{prefix}amplitude'].stderr
            component_params['sigma'] = params[f'{prefix}sigma'].value
            component_params['sigma_stderr'] = params[f'{prefix}sigma'].stderr
            
            # Calculate derived parameters for component
            component_params['fwhm'] = 2.355 * params[f'{prefix}sigma'].value
            component_params['area'] = (np.sqrt(2 * np.pi) * 
                                      params[f'{prefix}amplitude'].value * 
                                      params[f'{prefix}sigma'].value)
            
            line_params[f'component_{component_idx}'] = component_params
            component_idx += 1
        
        return line_params
    
    def save_fit_results(self, filename=None):
        """
        Save the last fitting results to a file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, generates automatic filename.
        """
        if self.last_fit_result is None:
            print("No fitting results to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fit_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("iSLAT Fitting Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.get_fit_report())
            f.write("\n\nLine Parameters:\n")
            f.write("-" * 20 + "\n")
            
            line_params = self.extract_line_parameters()
            for key, value in line_params.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Fit results saved to {filename}")

from typing import Optional, Dict, Any, Tuple, Union, Callable
import numpy as np
from collections import defaultdict
import threading

# Lazy imports with thread safety
_spectrum_module = None
_intensity_module = None
_import_lock = threading.Lock()

def _get_spectrum_module():
    """Thread-safe lazy import of spectrum module"""
    global _spectrum_module
    if _spectrum_module is None:
        with _import_lock:
            if _spectrum_module is None:  # Double-check pattern
                from .ir_model.spectrum import Spectrum
                _spectrum_module = Spectrum
    return _spectrum_module

def _get_intensity_module():
    """Thread-safe lazy import of intensity module"""
    global _intensity_module
    if _intensity_module is None:
        with _import_lock:
            if _intensity_module is None:  # Double-check pattern
                from .ir_model.intensity import Intensity
                _intensity_module = Intensity
    return _intensity_module

import iSLAT.Constants as c
from .MoleculeLineList import MoleculeLineList

class Molecule:
    """
    Optimized Molecule class with enhanced caching and performance improvements.
    """
    # Class-level optimizations
    __slots__ = (
        # Core attributes
        'name', 'filepath', 'displaylabel', 'color', 'is_visible', 'stellar_rv',
        'user_save_data', 'hitran_data', 'initial_molecule_parameters',
        'lines', 'intensity', 'spectrum',
        
        # Physical parameters (private)
        '_temp', '_radius', '_n_mol', '_distance', '_fwhm', '_broad',
        
        # Temporary attributes for initialization
        '_temp_val', '_radius_val', '_n_mol_val', '_distance_val', '_fwhm_val', '_broad_val',
        '_lines_filepath',
        
        # Derived parameters
        't_kin', 'scale_exponent', 'scale_number', 'radius_init', 'n_mol_init',
        'wavelength_range', 'model_pixel_res', 'model_line_width',
        
        # Plot data
        'plot_lam', 'plot_flux',
        
        # Caching and validation
        '_flux_cache', '_spectrum_valid', '_intensity_valid', '_interpolated_flux_cache',
        '_parameter_hash', '_last_wave_data_hash'
    )
    
    # Callbacks to notify when individual molecule parameters change
    _molecule_parameter_change_callbacks = []
    
    # Cache for expensive calculations shared across molecules
    _shared_calculation_cache = {}
    _cache_lock = threading.Lock()
    
    @classmethod
    def add_molecule_parameter_change_callback(cls, callback):
        """Add a callback function to be called when individual molecule parameters change"""
        cls._molecule_parameter_change_callbacks.append(callback)
    
    @classmethod
    def remove_molecule_parameter_change_callback(cls, callback):
        """Remove a callback function for molecule parameter changes"""
        if callback in cls._molecule_parameter_change_callbacks:
            cls._molecule_parameter_change_callbacks.remove(callback)
    
    @classmethod
    def _notify_molecule_parameter_change(cls, molecule_name, parameter_name, old_value, new_value):
        """Notify all callbacks that a molecule parameter has changed"""
        for callback in cls._molecule_parameter_change_callbacks:
            try:
                callback(molecule_name, parameter_name, old_value, new_value)
            except Exception as e:
                print(f"Error in molecule parameter change callback: {e}")
    
    def _notify_my_parameter_change(self, parameter_name, old_value, new_value):
        """Notify that this specific molecule's parameter has changed"""
        self._invalidate_parameter_hash()  # Invalidate parameter hash
        self.__class__._notify_molecule_parameter_change(self.name, parameter_name, old_value, new_value)
    
    def __init__(self, **kwargs):
        """
        Initialize a molecule with its parameters.
        Optimized with better caching and lazy initialization.
        """
        # Initialize caching system with enhanced features
        self._flux_cache = {}
        self._spectrum_valid = False
        self._intensity_valid = False
        self._interpolated_flux_cache = {}
        self._parameter_hash = None
        self._last_wave_data_hash = None
        
        # Initialize plot data as None (lazy loading)
        self.plot_lam = None
        self.plot_flux = None
        
        # Load user saved data if provided
        if 'hitran_data' in kwargs:
            print("Generating new molecule from default parameters.")
            self.user_save_data = None
            self.hitran_data = kwargs['hitran_data']
        elif 'user_save_data' in kwargs:
            print("Generating new molecule from user saved data.")
            self.user_save_data = kwargs['user_save_data']
            self.hitran_data = None
        else:
            self.user_save_data = None
            self.hitran_data = None

        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', {})

        # Initialize default values first to ensure all attributes exist
        self._temp_val = None
        self._radius_val = None  
        self._n_mol_val = None
        self._distance_val = None
        self._fwhm_val = None
        self._broad_val = None

        # Process user save data or use kwargs
        if self.user_save_data is not None:
            self._load_from_user_save_data(kwargs)
        else:
            self._load_from_kwargs(kwargs)

        # Set all instance-level parameters
        # Load molecular line data from file (lazy loading)
        self.lines = None  # Will be loaded when needed
        self._lines_filepath = self.filepath
        
        # Calculate derived parameters
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        
        # Initialize private attributes for properties with safe defaults
        self._temp = float(self._temp_val if self._temp_val is not None else self.t_kin)
        self._radius = float(self._radius_val if self._radius_val is not None else self.radius_init)
        self._n_mol = float(self._n_mol_val if self._n_mol_val is not None else self.n_mol_init)
        self._distance = float(self._distance_val if self._distance_val is not None else c.DEFAULT_DISTANCE)
        self._fwhm = float(self._fwhm_val if self._fwhm_val is not None else c.DEFAULT_FWHM)
        self._broad = float(self._broad_val if self._broad_val is not None else c.INTRINSIC_LINE_WIDTH)

        self.wavelength_range = kwargs.get('wavelength_range', c.WAVELENGTH_RANGE)
        self.model_pixel_res = kwargs.get('model_pixel_res', c.MODEL_PIXEL_RESOLUTION)
        self.model_line_width = kwargs.get('model_line_width', c.MODEL_LINE_WIDTH)

        # Defer intensity and spectrum creation until needed (lazy loading)
        self.intensity = None
        self.spectrum = None
        
        # Calculate initial parameter hash
        self._calculate_parameter_hash()

    def _load_from_user_save_data(self, kwargs):
        """Load parameters from user save data"""
        usd = self.user_save_data
        self.name = usd.get('Molecule Name', kwargs.get('name', 'Unknown Molecule'))
        self.filepath = usd.get('File Path', kwargs.get('filepath', None))
        self.displaylabel = usd.get('Molecule Label', self.name)
        self._temp_val = usd.get('Temp', kwargs.get('temp', None))
        self._radius_val = usd.get('Rad', kwargs.get('radius', None))
        self._n_mol_val = usd.get('N_Mol', kwargs.get('n_mol', None))
        self.color = usd.get('Color', kwargs.get('color', None))
        self.is_visible = usd.get('Vis', kwargs.get('is_visible', True))
        
        # Get instance values from user save data or kwargs
        self._distance_val = usd.get('Dist', kwargs.get('distance', c.DEFAULT_DISTANCE))
        self._fwhm_val = usd.get('FWHM', kwargs.get('fwhm', c.DEFAULT_FWHM))
        self._broad_val = usd.get('Broad', kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH))
        self.stellar_rv = kwargs.get('stellar_rv', c.DEFAULT_STELLAR_RV)
        
        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)

    def _load_from_kwargs(self, kwargs):
        """Load parameters from kwargs"""
        self.name = kwargs.get('name', kwargs.get('displaylabel', kwargs.get('filepath', 'Unknown Molecule')))
        self.filepath = kwargs.get('filepath', (self.hitran_data if hasattr(self, 'hitran_data') else None))
        self.displaylabel = kwargs.get('displaylabel', kwargs.get('name', 'Unknown Molecule'))
        self._temp_val = kwargs.get('temp', self.initial_molecule_parameters.get('t_kin', 300.0))
        self._radius_val = kwargs.get('radius', self.initial_molecule_parameters.get('radius_init', 1.0))
        self._n_mol_val = kwargs.get('n_mol', self.initial_molecule_parameters.get('n_mol', None))
        self.color = kwargs.get('color', None)
        self.is_visible = kwargs.get('is_visible', True)
        
        # Get instance values from kwargs or defaults
        self._distance_val = kwargs.get('distance', c.DEFAULT_DISTANCE)
        self._fwhm_val = kwargs.get('fwhm', c.DEFAULT_FWHM)
        self._broad_val = kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH)
        self.stellar_rv = kwargs.get('stellar_rv', c.DEFAULT_STELLAR_RV)
        
        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)
        
    def _ensure_lines_loaded(self):
        """Ensure molecular line data is loaded (lazy loading)"""
        if self.lines is None:
            if self._lines_filepath:
                print("Loading lines from filepath:", self._lines_filepath)
                self.lines = MoleculeLineList(molecule_id=self.name, filename=self._lines_filepath)
            else:
                print("Creating empty line list")
                self.lines = MoleculeLineList(molecule_id=self.name)
    
    def _ensure_intensity_initialized(self):
        """Ensure intensity object is initialized (lazy loading)"""
        if self.intensity is None:
            self._ensure_lines_loaded()
            Intensity = _get_intensity_module()
            self.intensity = Intensity(self.lines)
            self._intensity_valid = False  # Mark as needing calculation
    
    def _ensure_spectrum_initialized(self):
        """Ensure spectrum object is initialized (lazy loading)"""
        if self.spectrum is None:
            self._ensure_intensity_initialized()
            Spectrum = _get_spectrum_module()
            # Calculate spectral resolution from FWHM: R = λ/FWHM where FWHM is in km/s
            # Convert FWHM from km/s to wavelength units and calculate R
            mean_wavelength = (self.wavelength_range[0] + self.wavelength_range[1]) / 2.0
            # FWHM in wavelength units: Δλ = λ * (FWHM_velocity / c)
            delta_lambda = mean_wavelength * (self.fwhm / 299792.458)  # c in km/s
            spectral_resolution = mean_wavelength / delta_lambda if delta_lambda > 0 else self.model_line_width
            
            self.spectrum = Spectrum(
                lam_min=self.wavelength_range[0],
                lam_max=self.wavelength_range[1],
                dlambda=self.model_pixel_res,
                R=spectral_resolution,  # Use calculated resolution from molecule's FWHM
                distance=self.distance  # Use property to get correct value
            )
            self._spectrum_valid = False  # Mark as needing update
    
    def _calculate_parameter_hash(self):
        """Calculate a hash of current parameters for cache validation"""
        param_tuple = (
            self._temp, self._radius, self._n_mol, self._distance, 
            self._fwhm, self._broad, self.wavelength_range, 
            self.model_pixel_res, self.model_line_width,
            getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV)  # Include stellar RV in hash
        )
        self._parameter_hash = hash(param_tuple)
    
    def _invalidate_parameter_hash(self):
        """Invalidate the parameter hash"""
        self._parameter_hash = None

    def calculate_intensity(self):
        """Calculate intensity only if not cached or if parameters changed"""
        if self._intensity_valid and self._parameter_hash is not None:
            return
            
        self._ensure_intensity_initialized()
        
        # Calculate current parameter hash
        current_hash = self._get_current_parameter_hash()
        
        # Check if we can use cached calculation
        cache_key = (self.name, current_hash)
        with self._cache_lock:
            if cache_key in self._shared_calculation_cache:
                cached_result = self._shared_calculation_cache[cache_key]
                # Copy cached intensity data
                self.intensity._intensity = cached_result['intensity'].copy() if cached_result['intensity'] is not None else None
                self.intensity._tau = cached_result['tau'].copy() if cached_result['tau'] is not None else None
                self._intensity_valid = True
                self._spectrum_valid = False
                self._clear_flux_caches()
                return
            
        # Use the molecule's current parameter values (properties handle instance/class resolution)
        t_kin = self.temp  # Uses the property which returns _temp
        n_mol = self.n_mol  # Uses the property which returns _n_mol  
        dv = self.broad     # Uses the intrinsic line width property which returns _broad
        
        self.intensity.calc_intensity(
            t_kin=t_kin,
            n_mol=n_mol,
            dv=dv
        )
        
        # Cache the result for other molecules with same parameters
        with self._cache_lock:
            self._shared_calculation_cache[cache_key] = {
                'intensity': self.intensity._intensity.copy() if self.intensity._intensity is not None else None,
                'tau': self.intensity._tau.copy() if self.intensity._tau is not None else None
            }
            # Limit cache size to prevent memory issues
            if len(self._shared_calculation_cache) > 100:
                # Remove oldest entries
                keys_to_remove = list(self._shared_calculation_cache.keys())[:10]
                for key in keys_to_remove:
                    del self._shared_calculation_cache[key]
        
        # Mark intensity as valid and invalidate spectrum
        self._intensity_valid = True
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        # Update parameter hash
        self._parameter_hash = current_hash
        
        # Notify that intensity has been recalculated
        self._notify_my_parameter_change('intensity_recalculated', None, None)

    def _get_current_parameter_hash(self):
        """Get hash of current parameters for intensity calculation"""
        param_tuple = (
            getattr(self, '_temp', self.t_kin),
            getattr(self, '_n_mol', self.n_mol_init),
            getattr(self, '_broad', c.INTRINSIC_LINE_WIDTH),  # Use broad for intensity dv parameter
            getattr(self, '_fwhm', c.DEFAULT_FWHM),  # Include FWHM for spectrum resolution
            getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV),  # Include stellar RV
            # Include line data hash if available
            hash(str(self.lines.molecule_id)) if self.lines else 0
        )
        return hash(param_tuple)

    def _clear_flux_caches(self):
        """Clear all cached flux data"""
        self._flux_cache.clear()
        self._interpolated_flux_cache.clear()
        # Reset plot data
        self.plot_lam = None
        self.plot_flux = None

    def get_flux(self, wavelength_array):
        """Get flux with enhanced caching for performance"""
        # Create a cache key from the wavelength array
        try:
            cache_key = hash(wavelength_array.tobytes()) if hasattr(wavelength_array, 'tobytes') else str(wavelength_array)
        except (TypeError, ValueError) as e:
            # If hashing fails, use string representation as fallback
            cache_key = f"fallback_{id(wavelength_array)}"
        
        # Check parameter hash for cache validity
        current_param_hash = self._get_current_parameter_hash()
        cache_entry = self._interpolated_flux_cache.get(cache_key)
        
        if (cache_entry is not None and 
            cache_entry.get('param_hash') == current_param_hash and 
            self._spectrum_valid):
            return cache_entry['flux']
        
        # Ensure spectrum is valid
        self._ensure_spectrum_valid()
        
        lam_grid = self.spectrum._lamgrid
        flux_grid = self.spectrum.flux
        interpolated_flux = np.interp(wavelength_array, lam_grid, flux_grid)
        
        # Cache the result with parameter hash
        self._interpolated_flux_cache[cache_key] = {
            'flux': interpolated_flux,
            'param_hash': current_param_hash
        }
        
        # Limit cache size
        if len(self._interpolated_flux_cache) > 50:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._interpolated_flux_cache.keys())[:10]
            for key in keys_to_remove:
                del self._interpolated_flux_cache[key]
        
        return interpolated_flux
    
    def _ensure_spectrum_valid(self):
        """Ensure spectrum is calculated and up to date"""
        if not self._spectrum_valid:
            if not self._intensity_valid:
                self.calculate_intensity()
            self._update_spectrum()
            self._spectrum_valid = True
    
    def prepare_plot_data(self, wave_data):
        """
        Prepares wavelength and flux data aligned to the global observational wavelength grid.
        Uses enhanced caching for performance.

        Args:
            wave_data (np.ndarray): The wavelength grid (microns) used for observational data and plots.

        Returns:
            tuple: (wavelength array, flux array) matching wave_data
        """
        self._ensure_spectrum_initialized()
        
        # Create cache key for this wave_data
        wave_data_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else str(wave_data)
        current_param_hash = self._get_current_parameter_hash()
        
        # Check if we have cached data that's still valid
        cache_entry = self._flux_cache.get(wave_data_hash)
        if (cache_entry is not None and 
            cache_entry.get('param_hash') == current_param_hash and 
            self._spectrum_valid):
            self.plot_lam, self.plot_flux = cache_entry['data']
            return (self.plot_lam, self.plot_flux)
        
        # Ensure spectrum is valid before interpolation
        self._ensure_spectrum_valid()
        
        # Interpolate molecule flux onto the global wavelength grid
        interpolated_flux = np.interp(wave_data, self.spectrum.lamgrid, self.spectrum.flux_jy, left=0, right=0)
        
        # Apply stellar RV Doppler shift to the flux if stellar_rv is non-zero
        if hasattr(self, 'stellar_rv') and abs(self.stellar_rv) > 1e-6:
            # Doppler shift: λ_observed = λ_rest * (1 + v/c)
            # where v is radial velocity (positive = receding)
            doppler_factor = 1.0 + (self.stellar_rv / 299792.458)  # c in km/s
            shifted_wavelengths = wave_data / doppler_factor
            
            # Re-interpolate with shifted wavelengths
            interpolated_flux = np.interp(shifted_wavelengths, self.spectrum.lamgrid, self.spectrum.flux_jy, left=0, right=0)
        
        # Store results
        self.plot_lam = wave_data
        self.plot_flux = interpolated_flux
        
        # Cache the results with parameter hash
        self._flux_cache[wave_data_hash] = {
            'data': (self.plot_lam, self.plot_flux),
            'param_hash': current_param_hash
        }
        
        # Limit cache size
        if len(self._flux_cache) > 20:
            # Remove oldest entries
            keys_to_remove = list(self._flux_cache.keys())[:5]
            for key in keys_to_remove:
                del self._flux_cache[key]
        
        return (self.plot_lam, self.plot_flux)

    @property
    def temp(self):
        """Temperature getter"""
        return self._temp
    
    @temp.setter
    def temp(self, value):
        """Temperature setter - recalculates intensity when changed"""
        old_value = self._temp
        self._temp = float(value)
        self.t_kin = self._temp
        
        # Invalidate caches
        self._intensity_valid = False
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('temp', old_value, self._temp)
    
    @property
    def radius(self):
        """Radius getter"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Radius setter - updates spectrum area when changed"""
        old_value = self._radius
        self._radius = float(value)
        
        # Invalidate spectrum cache (but not intensity)
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('radius', old_value, self._radius)
    
    @property
    def n_mol(self):
        """Column density getter"""
        return self._n_mol
    
    @n_mol.setter
    def n_mol(self, value):
        """Column density setter - recalculates intensity when changed"""
        if value is None:
            # If None is passed, use the computed n_mol_init
            value = getattr(self, 'n_mol_init', 1e17)
        old_value = self._n_mol
        self._n_mol = float(value)
        
        # Invalidate caches
        self._intensity_valid = False
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('n_mol', old_value, self._n_mol)
    
    @property
    def distance(self):
        """Distance getter"""
        return self._distance
    
    @distance.setter
    def distance(self, value):
        """Distance setter - updates spectrum when changed"""
        old_value = self._distance
        self._distance = float(value)
        
        # Distance changes require spectrum recreation
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('distance', old_value, self._distance)
    
    @property
    def fwhm(self):
        """FWHM getter"""
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value):
        """FWHM setter - recalculates intensity and recreates spectrum when changed"""
        old_value = self._fwhm
        self._fwhm = float(value)
        
        # FWHM changes require both intensity recalculation and spectrum recreation
        # because FWHM affects the spectral resolution R parameter
        self._intensity_valid = False
        self._spectrum_valid = False
        self.spectrum = None  # Force spectrum recreation with new FWHM-based resolution
        self._clear_flux_caches()
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('fwhm', old_value, self._fwhm)

    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], skip_notification: bool = False):
        """
        Bulk update multiple parameters efficiently.
        
        Args:
            parameter_dict: Dictionary of parameter names and values
            skip_notification: If True, skip individual parameter change notifications
        """
        # Store old values for notification
        old_values = {}
        
        # Update all parameters without triggering individual cache invalidations
        for param_name, value in parameter_dict.items():
            if hasattr(self, f'_{param_name}'):
                old_values[param_name] = getattr(self, f'_{param_name}')
                setattr(self, f'_{param_name}', float(value))
            elif param_name == 'intrinsic_line_width':
                old_values[param_name] = self._broad
                self._broad = float(value)
            elif hasattr(self, param_name):
                old_values[param_name] = getattr(self, param_name)
                setattr(self, param_name, value)
        
        # Invalidate all caches once
        if any(param in parameter_dict for param in ['temp', 'n_mol', 'fwhm', 'intrinsic_line_width']):
            self._intensity_valid = False
        
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        # Send notifications if not skipped
        if not skip_notification:
            for param_name, value in parameter_dict.items():
                old_value = old_values.get(param_name)
                self._notify_my_parameter_change(param_name, old_value, value)

    @property
    def star_rv(self):
        """Stellar RV getter"""
        return getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV)
    
    @star_rv.setter
    def star_rv(self, value):
        """Stellar RV setter"""
        old_value = getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV)
        self.stellar_rv = float(value)
        
        # Stellar RV changes affect the final spectrum due to Doppler shift
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('stellar_rv', old_value, self.stellar_rv)
    
    @property
    def broad(self):
        """Intrinsic line width getter"""
        return self._broad
    
    @broad.setter
    def broad(self, value):
        """Intrinsic line width setter"""
        old_value = self._broad
        self._broad = float(value)

        # Invalidate caches
        self._intensity_valid = False
        self._spectrum_valid = False
        self._clear_flux_caches()

        self._notify_my_parameter_change('broad', old_value, self._broad)

    @property
    def intrinsic_line_width(self):
        """Alias for broad (intrinsic line width)"""
        return self.broad
    
    @intrinsic_line_width.setter
    def intrinsic_line_width(self, value):
        """Alias setter for broad (intrinsic line width)"""
        self.broad = value
    
    def _update_spectrum(self):
        """Update the spectrum with current intensity and area"""
        if hasattr(self, 'spectrum') and hasattr(self, 'intensity'):
            # Ensure intensity is valid first
            if not self._intensity_valid:
                self.calculate_intensity()
                
            # Clear previous intensity data
            self.spectrum._I_list = []
            self.spectrum._lam_list = []
            self.spectrum._dA_list = []
            
            # Add updated intensity with current radius
            self.spectrum.add_intensity(
                intensity=self.intensity,
                dA=self.radius ** 2 * np.pi  # Use property to get correct value
            )
            
            # Mark spectrum as valid and clear flux caches
            self._spectrum_valid = True
            self._clear_flux_caches()
    
    def _recreate_spectrum(self):
        """Recreate the spectrum when distance or other fundamental parameters change"""
        if hasattr(self, 'intensity'):
            # Ensure intensity is valid first
            if not self._intensity_valid:
                self.calculate_intensity()
                
            Spectrum = _get_spectrum_module()
            # Calculate spectral resolution from FWHM
            mean_wavelength = (self.wavelength_range[0] + self.wavelength_range[1]) / 2.0
            delta_lambda = mean_wavelength * (self.fwhm / 299792.458)  # c in km/s
            spectral_resolution = mean_wavelength / delta_lambda if delta_lambda > 0 else self.model_line_width
            
            self.spectrum = Spectrum(
                lam_min=self.wavelength_range[0],
                lam_max=self.wavelength_range[1],
                dlambda=self.model_pixel_res,
                R=spectral_resolution,  # Use calculated resolution from molecule's FWHM
                distance=self.distance  # Use the property which handles instance vs class values
            )
            
            self.spectrum.add_intensity(
                intensity=self.intensity,
                dA=self.radius ** 2 * np.pi  # Use property to get correct value
            )
            
            # Mark spectrum as valid and clear flux caches
            self._spectrum_valid = True
            self._clear_flux_caches()

    def __str__(self):
        attrs = vars(self)
        def truncate(val):
            s = str(val)
            return s if len(s) <= 3000 else s[:3000] + '...<truncated>'
        attr_str = '\n'.join(f"{key}={truncate(value)}" for key, value in attrs.items())
        return f"Molecule(\n{attr_str}\n)"
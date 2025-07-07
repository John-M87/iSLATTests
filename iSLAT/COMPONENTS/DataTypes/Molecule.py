'''from iSLAT.ir_model.spectrum import Spectrum
from iSLAT.ir_model.moldata import MolData
from iSLAT.ir_model.intensity import Intensity
from iSLAT.ir_model.constants import constants as c'''
from .moldata import MolData
from .spectrum import Spectrum
from .intensity import Intensity
from iSLAT.IRconstants import constants as c
import iSLAT.Constants as default_parms
import numpy as np

class Molecule:
    # Callbacks to notify when individual molecule parameters change
    _molecule_parameter_change_callbacks = []
    
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
        self.__class__._notify_molecule_parameter_change(self.name, parameter_name, old_value, new_value)
    
    def __init__(self, **kwargs):
        """
        Initialize a molecule with its parameters.
        All parameters are now instance-level.
        """
        # Initialize caching system
        self._flux_cache = {}
        self._spectrum_valid = False
        self._intensity_valid = False
        self._interpolated_flux_cache = {}
        
        # Load user saved data if provided
        if 'hitran_data' in kwargs:
            print("Generating new molecule from default parameters.")
            self.user_save_data = None
            self.hitran_data = kwargs['hitran_data']
        elif 'user_save_data' in kwargs:
            print("Generating new molecule from user saved data.")
            self.user_save_data = kwargs['user_save_data']

        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', {})

        if self.user_save_data is not None:
            usd = self.user_save_data
            # Map user saved data fields to class attributes
            self.name = usd.get('Molecule Name', kwargs.get('name', 'Unknown Molecule'))
            self.filepath = usd.get('File Path', kwargs.get('filepath', None))
            self.displaylabel = usd.get('Molecule Label', self.name)
            temp_val = usd.get('Temp', kwargs.get('temp', None))
            radius_val = usd.get('Rad', kwargs.get('radius', None))
            n_mol_val = usd.get('N_Mol', kwargs.get('n_mol', None))
            self.color = usd.get('Color', kwargs.get('color', None))
            self.is_visible = usd.get('Vis', kwargs.get('is_visible', True))
            
            # Get instance values from user save data or kwargs
            distance_val = usd.get('Dist', kwargs.get('distance', default_parms.dist))
            fwhm_val = usd.get('FWHM', kwargs.get('fwhm', default_parms.fwhm))
            broad_val = usd.get('Broad', kwargs.get('broad', default_parms.intrinsic_line_width))
            self.stellar_rv = kwargs.get('stellar_rv', default_parms.star_rv)
        else:
            self.name = kwargs.get('name', kwargs.get('displaylabel', kwargs.get('filepath', 'Unknown Molecule')))
            self.filepath = kwargs.get('filepath', (self.hitran_data if hasattr(self, 'hitran_data') else None))
            self.displaylabel = kwargs.get('displaylabel', kwargs.get('name', 'Unknown Molecule'))
            temp_val = kwargs.get('temp', self.initial_molecule_parameters.get('t_kin', 300.0))
            radius_val = kwargs.get('radius', self.initial_molecule_parameters.get('radius_init', 1.0))
            n_mol_val = kwargs.get('n_mol', self.initial_molecule_parameters.get('n_mol', None))
            self.color = kwargs.get('color', None)
            self.is_visible = kwargs.get('is_visible', True)
            
            # Get instance values from kwargs or defaults
            distance_val = kwargs.get('distance', default_parms.dist)
            fwhm_val = kwargs.get('fwhm', default_parms.fwhm)
            broad_val = kwargs.get('broad', default_parms.intrinsic_line_width)
            self.stellar_rv = kwargs.get('stellar_rv', default_parms.star_rv)

        # Set all instance-level parameters
        self.mol_data = MolData(self.name, self.filepath)

        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', temp_val if temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', radius_val if radius_val is not None else 1.0)

        # Calculate n_mol_init from scale_number and scale_exponent
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        
        # Initialize private attributes for properties
        self._temp = float(temp_val if temp_val is not None else self.t_kin)
        self._radius = float(radius_val if radius_val is not None else self.radius_init)
        self._n_mol = float(n_mol_val if n_mol_val is not None else self.n_mol_init)
        self._distance = float(distance_val)
        self._fwhm = float(fwhm_val)
        self.broad = float(broad_val)

        # Set wavelength range and model parameters
        self.wavelength_range = kwargs.get('wavelength_range', default_parms.wavelength_range)
        self.model_pixel_res = kwargs.get('model_pixel_res', default_parms.model_pixel_res)
        self.model_line_width = kwargs.get('model_line_width', default_parms.model_line_width)

        self.intensity = Intensity(self.mol_data)
        self.calculate_intensity()

        self.spectrum = Spectrum(
            lam_min=self.wavelength_range[0],
            lam_max=self.wavelength_range[1],
            dlambda=self.model_pixel_res,
            R=self.model_line_width,
            distance=self._distance
        )

        self.spectrum.add_intensity(
            intensity=self.intensity,
            dA=self._radius ** 2 * np.pi
        )
        
        # Mark spectrum and intensity as valid after initial calculation
        self._spectrum_valid = True
        self._intensity_valid = True

    def calculate_intensity(self):
        """Calculate intensity only if not cached or if parameters changed"""
        if self._intensity_valid:
            return
            
        t_kin = getattr(self, '_temp', self.t_kin)
        n_mol = getattr(self, '_n_mol', self.n_mol_init)
        dv = getattr(self, '_fwhm', default_parms.fwhm)
        self.intensity.calc_intensity(
            t_kin=t_kin,
            n_mol=n_mol,
            dv=dv
        )
        
        # Mark intensity as valid and invalidate spectrum
        self._intensity_valid = True
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        # Notify that intensity has been recalculated
        self._notify_my_parameter_change('intensity_recalculated', None, None)

    def _clear_flux_caches(self):
        """Clear all cached flux data"""
        self._flux_cache.clear()
        self._interpolated_flux_cache.clear()

    def get_flux(self, wavelength_array):
        """Get flux with caching for performance"""
        # Create a cache key from the wavelength array
        cache_key = hash(wavelength_array.tobytes()) if hasattr(wavelength_array, 'tobytes') else str(wavelength_array)
        
        if cache_key in self._interpolated_flux_cache and self._spectrum_valid:
            return self._interpolated_flux_cache[cache_key]
        
        # Ensure spectrum is valid
        self._ensure_spectrum_valid()
        
        lam_grid = self.spectrum._lamgrid
        flux_grid = self.spectrum.flux
        interpolated_flux = np.interp(wavelength_array, lam_grid, flux_grid)
        
        # Cache the result
        self._interpolated_flux_cache[cache_key] = interpolated_flux
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
        Uses caching for performance.

        Args:
            wave_data (np.ndarray): The wavelength grid (microns) used for observational data and plots.

        Sets:
            self.plot_lam : wavelength array matching wave_data
            self.plot_flux: flux density array interpolated to wave_data
        """
        if self.spectrum is None:
            raise ValueError(f"Spectrum for molecule '{self.name}' is not initialized.")
        
        # Create cache key for this wave_data
        cache_key = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else str(wave_data)
        
        # Check if we have cached data and spectrum is still valid
        if cache_key in self._flux_cache and self._spectrum_valid:
            self.plot_lam, self.plot_flux = self._flux_cache[cache_key]
            return (self.plot_lam, self.plot_flux)
        
        # Ensure spectrum is valid before interpolation
        self._ensure_spectrum_valid()
        
        # Interpolate molecule flux onto the global wavelength grid
        interpolated_flux = np.interp(wave_data, self.spectrum.lamgrid, self.spectrum.flux_jy, left=0, right=0)
        
        # Store results
        self.plot_lam = wave_data
        self.plot_flux = interpolated_flux
        
        # Cache the results
        self._flux_cache[cache_key] = (self.plot_lam, self.plot_flux)
        
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
        """FWHM setter - recalculates intensity when changed"""
        old_value = self._fwhm
        self._fwhm = float(value)
        
        # Invalidate caches
        self._intensity_valid = False
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('fwhm', old_value, self._fwhm)

    @property
    def star_rv(self):
        """Stellar RV getter"""
        return getattr(self, 'stellar_rv', default_parms.star_rv)
    
    @star_rv.setter
    def star_rv(self, value):
        """Stellar RV setter"""
        old_value = getattr(self, 'stellar_rv', default_parms.star_rv)
        self.stellar_rv = float(value)
        self._notify_my_parameter_change('stellar_rv', old_value, self.stellar_rv)
    
    @property
    def intrinsic_line_width(self):
        """Intrinsic line width getter"""
        return self.broad
    
    @intrinsic_line_width.setter
    def intrinsic_line_width(self, value):
        """Intrinsic line width setter"""
        old_value = self.broad
        self.broad = float(value)
        
        # Invalidate caches
        self._intensity_valid = False
        self._spectrum_valid = False
        self._clear_flux_caches()
        
        self._notify_my_parameter_change('intrinsic_line_width', old_value, self.broad)
    
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
                dA=self._radius ** 2 * np.pi
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
                
            self.spectrum = Spectrum(
                lam_min=self.wavelength_range[0],
                lam_max=self.wavelength_range[1],
                dlambda=self.model_pixel_res,
                R=self.model_line_width,
                distance=self.distance  # Use the property which handles instance vs class values
            )
            
            self.spectrum.add_intensity(
                intensity=self.intensity,
                dA=self._radius ** 2 * np.pi
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
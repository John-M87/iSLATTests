from iSLAT.ir_model.spectrum import Spectrum
from iSLAT.ir_model.moldata import MolData
from iSLAT.ir_model.intensity import Intensity
from iSLAT.ir_model.constants import constants as c
#from iSLAT.iSLATDefaultInputParms import model_line_width, model_pixel_res #, wavelength_range
#from iSLAT.iSLATDefaultInputParms import *
import iSLAT.iSLATDefaultInputParms as default_parms
import numpy as np

class Molecule:
    def __init__(self, **kwargs):
        """
        Initialize a molecule with its parameters.

        Parameters
        ----------
        name: str
            Name of the molecule
        intrinsic_line_width: float
            Intrinsic line width
        model_pixel_res: float
            Pixel resolution for the model spectrum
        model_line_width: float
            Line width for the model spectrum
        distance: float
            Distance
        wavelength_range: tuple
            Wavelength range
        hitran_data: str
            Path to HITRAN data
        kwargs: dict
            Additional parameters, possibly from user saved data
        """
        #self.name = name

        # Load user saved data if provided
        #self.user_save_data = user_save_data

        if 'hitran_data' in kwargs:
            print("Generating new molecule from default parameters.")
            # If hitran_data is provided, assume there is no user saved data for this molecule and use default parameters
            self.user_save_data = None
            self.hitran_data = kwargs['hitran_data']
        elif 'user_save_data' in kwargs:
            print("Generating new molecule from user saved data.")
            # If user_save_data is provided, use it to initialize the molecule
            self.user_save_data = kwargs['user_save_data']

        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', None)

        if self.user_save_data is not None:
            usd = self.user_save_data
            # Map user saved data fields to class attributes
            self.name = usd.get('Molecule Name', kwargs.get('name', 'Unknown Molecule'))
            self.filepath = usd.get('File Path', kwargs.get('filepath', None))
            self.displaylabel = usd.get('Molecule Label', self.name)
            self.temp = float(usd.get('Temp', kwargs.get('temp', None)))
            self.radius = float(usd.get('Rad', kwargs.get('radius', None)))
            self.n_mol = float(usd.get('N_Mol', kwargs.get('n_mol', None)))
            self.color = usd.get('Color', kwargs.get('color', None))
            self.is_visible = usd.get('Vis', kwargs.get('is_visible', True))
            self.distance = float(usd.get('Dist', kwargs.get('distance', default_parms.dist)))
            # Optional: handle StellarRV, FWHM, Broad if needed
            self.stellar_rv = kwargs.get('stellar_rv', default_parms.star_rv)
            self.fwhm = float(usd.get('FWHM', default_parms.fwhm))
            self.broad = float(usd.get('Broad', default_parms.broadening))
        else:
            #if hasattr(self, 'hitran_data'):
            self.name = kwargs.get('name', kwargs.get('displaylabel', kwargs.get('filepath', 'Unknown Molecule')))
            self.filepath = kwargs.get('filepath', (self.hitran_data if 'hitran_data' in kwargs else None))
            self.displaylabel = kwargs.get('displaylabel', kwargs.get('name', 'Unknown Molecule'))
            self.temp = kwargs.get('temp', self.initial_molecule_parameters.get('t_kin', 300.0))
            self.radius = kwargs.get('radius', self.initial_molecule_parameters.get('radius_init', 1.0))
            self.n_mol = kwargs.get('n_mol', None)
            self.color = kwargs.get('color', None)
            self.is_visible = kwargs.get('is_visible', True)
            self.distance = kwargs.get('distance', default_parms.dist)
            self.stellar_rv = kwargs.get('stellar_rv', default_parms.star_rv)
            self.fwhm = kwargs.get('fwhm', default_parms.fwhm)
            self.broad = kwargs.get('broad', default_parms.broadening)

        #self.hitran_data = hitran_data

        # Initial molecule parameters
        if self.initial_molecule_parameters is None:
            # If no initial molecule parameters, use user_save_data as the source
            if self.user_save_data is not None:
                # Convert user_save_data keys to expected keys if needed
                self.initial_molecule_parameters = {
                    't_kin': self.user_save_data.get('Temp', self.temp),
                    'scale_exponent': 1.0,
                    'scale_number': 1.0,
                    'radius_init': self.user_save_data.get('Rad', self.radius),
                    #'intrinsic_line_width': self.user_save_data.get('FWHM', None)
            }
            else:
                self.initial_molecule_parameters = {}
        else:
            if self.user_save_data is not None:
                # Update initial parameters with user saved data
                self.initial_molecule_parameters.update(self.user_save_data)

        self.mol_data = MolData(self.name, self.filepath)

        # Set kinetic temperature
        self.t_kin = self.initial_molecule_parameters.get('t_kin', (self.temp if self.temp is not None else self.initial_molecule_parameters.get('t_kin', 300.0)))
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', self.initial_molecule_parameters.get('scale_exponent', 1.0))
        self.scale_number = self.initial_molecule_parameters.get('scale_number', self.initial_molecule_parameters.get('scale_number', 1.0))
        self.radius_init = self.initial_molecule_parameters.get('radius_init', (self.radius if self.radius is not None else self.initial_molecule_parameters.get('radius_init', 1.0)))

        # Store current values temporarily before setting up properties
        temp_val = self.temp if self.temp is not None else self.t_kin
        radius_val = self.radius if self.radius is not None else self.radius_init
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        n_mol_val = self.n_mol if self.n_mol is not None else self.n_mol_init
        distance_val = self.distance if hasattr(self, 'distance') else default_parms.dist

        # Initialize private attributes for properties
        self._temp = temp_val
        self._radius = radius_val
        self._n_mol = n_mol_val
        self._distance = distance_val
        self._fwhm = self.fwhm

        #self.intrinsic_line_width = intrinsic_line_width
        self.model_pixel_res = kwargs.get('model_pixel_res', default_parms.model_pixel_res)  # Pixel resolution for the model spectrum
        self.model_line_width = kwargs.get('model_line_width', default_parms.model_line_width)  # Line width for the model spectrum
        #self.distance = distance
        self.wavelength_range = kwargs.get('wavelength_range', (0.3, 1000))

        self.intensity = Intensity(self.mol_data)
        self.calculate_intensity()

        self.spectrum = Spectrum(
            lam_min = self.wavelength_range[0],
            lam_max = self.wavelength_range[1],
            dlambda = self.model_pixel_res,
            R = self.model_line_width,
            distance = self._distance
        )

        '''print("Here are the parameters right now, we finna boutta add intensity:")
        print(self.__str__())'''

        self.spectrum.add_intensity(
            intensity=self.intensity,
            dA=self._radius ** 2 * np.pi
        )

        '''print("Here are the spectrums lists after adding intensity:")
        print(f'Spectrum _I_list: {self.spectrum._I_list}')
        print(f'Spectrum _lam_list: {self.spectrum._lam_list}')'''

    def calculate_intensity(self):
        t_kin = getattr(self, '_temp', self.t_kin)
        n_mol = getattr(self, '_n_mol', self.n_mol_init)
        dv = getattr(self, '_fwhm', self.broad)  # Use _fwhm if available, fallback to broad
        print(f"Calculating intensity for {self.name} with T={t_kin}, n_mol={n_mol}, dv={dv}")
        self.intensity.calc_intensity(
            t_kin=t_kin,
            n_mol=n_mol,
            dv=dv
        )

    def get_flux(self, wavelength_array):
        lam_grid = self.spectrum._lamgrid
        flux_grid = self.spectrum.flux
        return np.interp(wavelength_array, lam_grid, flux_grid)
    
    def prepare_plot_data(self, wave_data):
        """
        Prepares wavelength and flux data aligned to the global observational wavelength grid.

        Args:
            wave_data (np.ndarray): The wavelength grid (microns) used for observational data and plots.

        Sets:
            self.plot_lam : wavelength array matching wave_data
            self.plot_flux: flux density array interpolated to wave_data
        """
        if self.spectrum is None:
            raise ValueError(f"Spectrum for molecule '{self.name}' is not initialized.")
        
        # Interpolate molecule flux onto the global wavelength grid
        interpolated_flux = np.interp(wave_data, self.spectrum.lamgrid, self.spectrum.flux_jy, left=0, right=0)
        
        # Store results
        self.plot_lam = wave_data
        self.plot_flux = interpolated_flux
        return (self.plot_lam, self.plot_flux)

    @property
    def temp(self):
        """Temperature getter"""
        return self._temp
    
    @temp.setter
    def temp(self, value):
        """Temperature setter - recalculates intensity when changed"""
        self._temp = float(value)
        self.t_kin = self._temp
        if hasattr(self, 'intensity') and hasattr(self, 'spectrum'):
            self.calculate_intensity()
            self._update_spectrum()
    
    @property
    def radius(self):
        """Radius getter"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Radius setter - updates spectrum area when changed"""
        self._radius = float(value)
        if hasattr(self, 'intensity') and hasattr(self, 'spectrum'):
            self._update_spectrum()
    
    @property
    def n_mol(self):
        """Column density getter"""
        return self._n_mol
    
    @n_mol.setter
    def n_mol(self, value):
        """Column density setter - recalculates intensity when changed"""
        self._n_mol = float(value)
        if hasattr(self, 'intensity') and hasattr(self, 'spectrum'):
            self.calculate_intensity()
            self._update_spectrum()
    
    @property
    def distance(self):
        """Distance getter"""
        return self._distance
    
    @distance.setter
    def distance(self, value):
        """Distance setter - updates spectrum when changed"""
        self._distance = float(value)
        if hasattr(self, 'spectrum'):
            self._recreate_spectrum()
    
    @property
    def fwhm(self):
        """FWHM getter"""
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value):
        """FWHM setter - recalculates intensity when changed"""
        self._fwhm = float(value)
        if hasattr(self, 'intensity') and hasattr(self, 'spectrum'):
            self.calculate_intensity()
            self._update_spectrum()
    
    def _update_spectrum(self):
        """Update the spectrum with current intensity and area"""
        if hasattr(self, 'spectrum') and hasattr(self, 'intensity'):
            # Clear previous intensity data
            self.spectrum._I_list = []
            self.spectrum._lam_list = []
            self.spectrum._dA_list = []
            
            # Add updated intensity with current radius
            self.spectrum.add_intensity(
                intensity=self.intensity,
                dA=self._radius ** 2 * np.pi
            )
    
    def _recreate_spectrum(self):
        """Recreate the spectrum when distance or other fundamental parameters change"""
        if hasattr(self, 'intensity'):
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

    def __str__(self):
        attrs = vars(self)
        def truncate(val):
            s = str(val)
            return s if len(s) <= 3000 else s[:3000] + '...<truncated>'
        attr_str = '\n'.join(f"{key}={truncate(value)}" for key, value in attrs.items())
        return f"Molecule(\n{attr_str}\n)"
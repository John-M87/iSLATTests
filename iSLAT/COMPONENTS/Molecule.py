from iSLAT.ir_model.spectrum import Spectrum
from iSLAT.ir_model.moldata import MolData
from iSLAT.ir_model.intensity import Intensity
from iSLAT.ir_model.constants import constants as c
from iSLAT.iSLATDefaultInputParms import model_line_width, model_pixel_res #, wavelength_range
import numpy as np

class Molecule:
    #def __init__(self, name, intrinsic_line_width, model_pixel_res, model_line_width, distance, wavelength_range, hitran_data, is_visible = True, color= None, displaylabel= None, filepath= None, initial_molecule_parameters = None, temp = None, n_mol= None, radius= None):
    #def __init__(self, *, name, intrinsic_line_width, model_pixel_res, model_line_width, distance, wavelength_range, hitran_data, is_visible=True, color=None, displaylabel=None, filepath=None, initial_molecule_parameters=None, temp=None, n_mol=None, radius=None):
    def __init__(self, user_save_data, **kwargs):
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
        self.user_save_data = user_save_data
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
            self.distance = float(usd.get('Dist', kwargs.get('distance', None)))
            # Optional: handle StellarRV, FWHM, Broad if needed
            self.stellar_rv = float(usd.get('StellarRV', None))
            self.fwhm = float(usd.get('FWHM', None))
            self.broad = float(usd.get('Broad', None))
        else:
            self.filepath = kwargs.get('filepath', None)
            self.displaylabel = kwargs.get('displaylabel', kwargs.get('name', 'Unknown Molecule'))
            self.temp = kwargs.get('temp', None)
            self.radius = kwargs.get('radius', None)
            self.n_mol = kwargs.get('n_mol', None)
            self.color = kwargs.get('color', None)
            self.is_visible = kwargs.get('is_visible', True)
            self.distance = kwargs.get('distance', None)
            self.stellar_rv = kwargs.get('stellar_rv', None)
            self.fwhm = kwargs.get('fwhm', None)
            self.broad = kwargs.get('broad', None)

        #self.hitran_data = hitran_data

        # Initial molecule parameters
        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', None)
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
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self.temp)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self.radius)

        self.temp = self.temp if self.temp is not None else self.t_kin
        self.radius = self.radius if self.radius is not None else self.radius_init
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        self.n_mol = self.n_mol if self.n_mol is not None else self.n_mol_init

        #self.intrinsic_line_width = intrinsic_line_width
        self.model_pixel_res = kwargs.get('model_pixel_res', model_pixel_res)  # Pixel resolution for the model spectrum
        self.model_line_width = kwargs.get('model_line_width', model_line_width)  # Line width for the model spectrum
        #self.distance = distance
        self.wavelength_range = kwargs.get('wavelength_range', (0.3, 1000))

        self.intensity = Intensity(self.mol_data)
        self.calculate_intensity()

        self.spectrum = Spectrum(
            lam_min = self.wavelength_range[0],
            lam_max = self.wavelength_range[1],
            dlambda = self.model_pixel_res,
            R = self.model_line_width,
            distance = self.distance
        )

        self.spectrum.add_intensity(
            intensity=self.intensity,
            dA=self.radius * 2 ** np.pi
        )

    def calculate_intensity(self):
        # Use values from user_save_data or fallback to defaults
        #t_kin = self.temp if self.temp is not None else self.t_kin
        t_kin = self.t_kin
        n_mol = self.n_mol
        dv = self.fwhm
        print(f"Using t_kin={t_kin}, n_mol={n_mol}, dv={dv} for {self.name}")
        # Try to get dv (intrinsic line width) from user_save_data or initial_molecule_parameters
        #if self.user_save_data is not None:
        #    dv = self.user_save_data.get('FWHM', self.initial_molecule_parameters.get('intrinsic_line_width', None))
        #else:
        #    dv = self.initial_molecule_parameters.get('intrinsic_line_width', None)
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
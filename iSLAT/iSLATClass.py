iSLAT_version = 'v5.00.00'
print(' ')
print('Loading iSLAT ' + iSLAT_version + ': Please Wait ...')

# Import necessary modules
import numpy as np
import pandas as pd
import os
import json

#from lmfit.models import GaussianModel
import tkinter as tk
from tkinter import filedialog
import ssl
import certifi
import datetime

context = ssl.create_default_context(cafile=certifi.where ())

from .iSLATFileHandling import load_user_settings, read_default_molecule_parameters, read_initial_molecule_parameters, read_save_data, read_HITRAN_data, read_from_user_csv, read_default_csv, read_spectral_data

from .ir_model import *
from .COMPONENTS.chart_window import MoleculeSelector
from .COMPONENTS.Hitran_data import get_Hitran_data
from .COMPONENTS.partition_function_writer import write_partition_function
from .COMPONENTS.line_data_writer import write_line_data
from .COMPONENTS.slabfit_config import *
from .COMPONENTS.slabfit_loader import *
from .COMPONENTS.slabfit_runner import *
from .iSLATDefaultInputParms import *
#from .iSLATFileHandling import *
from .COMPONENTS.GUI import *
from .COMPONENTS.Molecule import Molecule
from .COMPONENTS.MoleculeDict import MoleculeDict

class iSLAT:
    """
    iSLAT class to handle the iSLAT functionalities.
    This class is used to initialize the iSLAT application, load user settings, and manage the main functionalities.
    """

    def __init__(self):
        """
        Initialize the iSLAT application.
        """
        #self.directorypath = os.path.dirname(os.path.abspath(__file__))
        #print(f"iSLAT directory path: {self.directorypath}")
        self._hitran_data = {}
        self.create_folders()

        # Load settings
        #self.user_settings = self.load_user_settings()
        self.user_settings = load_user_settings()
        #self.update_default_molecule_parameters()
        #self.update_initial_molecule_parameters()
        self.initial_molecule_parameters = read_initial_molecule_parameters()
        self.molecules_parameters_default = read_default_molecule_parameters()
        
        self.mols = ["H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N"]
        self.basem = ["H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N"]
        self.isot = [1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1]
        self.default_molecule_csv_data = read_default_csv()
        self.user_saved_molecules = read_from_user_csv()

        self.wavelength_range = wavelength_range
        self._display_range = (23.52, 25.41)

        self.min_vu = 1 / (self.wavelength_range[0] / 1E6) / 100.
        self.max_vu = 1 / (self.wavelength_range[1] / 1E6) / 100.

        # Check for HITRAN files and download if necessary
        self.check_HITRAN()
        #self.load_default_HITRAN_data()

        self.molecules_data_default = molecules_data.copy()
        self.deleted_molecules = []

        self.xp1 = self.xp2 = None

    def init_gui(self):
        """
        Initialize the GUI components of iSLAT.
        This function sets up the main window, menus, and other GUI elements.
        """
        if not hasattr(self, "root"):
            self.root = tk.Tk()
            self.root.title("iSLAT - Infrared Spectral Line Analysis Tool")
            self.root.resizable(True, True)

        #self.root.mainloop()

        if not hasattr(self, "GUI"):
            self.GUI = GUI(
                master=self.root,
                molecule_data=self.molecules_dict,
                wave_data=self.wave_data,
                flux_data=self.flux_data,
                config=self.user_settings,
                islat_class_ref=self
            )
        
        self.GUI.start()

    def init_molecules(self, mole_save_data=None):
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()
        if not hasattr(self, "user_saved_molecules"):
                self.user_saved_molecules = read_from_user_csv()
        
        if mole_save_data is None:
            mole_save_data = self.user_saved_molecules
        else:
            mole_save_data = mole_save_data

        new_molecules = []
        for mol in mole_save_data.values():
            mol_name = mol["Molecule Name"]
            if mol_name not in self.molecules_dict:
                new_molecule = Molecule(
                    user_save_data=mol,
                    wavelength_range=self.wavelength_range,
                    initial_molecule_parameters = self.initial_molecule_parameters.get(mol_name, self.molecules_parameters_default)
                )
                new_molecules.append(new_molecule)

        if new_molecules:
            self.molecules_dict.add_molecules(new_molecules)

        # Initialize the active molecule based on user settings
        active_molecule_name = self.user_settings.get("default_active_molecule", "H2O")
        if active_molecule_name in self.molecules_dict:
            self._active_molecule = self.molecules_dict[active_molecule_name]
        else:
            print(f"Active molecule '{active_molecule_name}' not found in the dictionary. Defaulting to 'H2O'.")
            self._active_molecule = self.molecules_dict.get("H2O", None)

    def run(self):
        """
        Run the iSLAT application.
        This function starts the main event loop of the Tkinter application.
        """
        # Start the main event loop
        self.savedata = read_save_data()
        self.init_molecules()
        #self.add_molecule_from_hitran()
        self.load_spectrum()
        self.init_gui()
    
    def add_molecule_from_hitran(self, refresh = True, hitran_files = None, molecule_names = None, base_molecules = None, isotopes = None):
        """
        Adds one or more molecules to the iSLAT instance from HITRAN files.
        If molecule_names are not provided, they will be extracted from the file names.
        
        Parameters:
        -----------
        refresh : bool
            Whether to refresh the GUI after adding molecules
        hitran_files : str or list
            Single file path or list of file paths. If None, opens file dialog for multiple selection
        molecule_names : str or list
            Single molecule name or list of molecule names corresponding to files
        base_molecules : str or list
            Single base molecule or list of base molecules (currently unused)
        isotopes : int or list
            Single isotope or list of isotopes (currently unused)
        """
        if hitran_files is None:
            hitran_files = filedialog.askopenfilenames(
                title='Choose HITRAN Data Files (select multiple with Ctrl/Cmd)', 
                filetypes=[('PAR Files', '*.par')], 
                initialdir=os.path.abspath("HITRANdata")
            )
        
        if not hitran_files:
            print("No HITRAN files selected.")
            return
        
        # Convert single file to list for consistent processing
        if isinstance(hitran_files, str):
            hitran_files = [hitran_files]
        
        # Convert molecule_names to list if provided as single string
        if molecule_names is not None and isinstance(molecule_names, str):
            molecule_names = [molecule_names]
        
        new_molecules = []
        
        for i, hitran_file in enumerate(hitran_files):
            # Get molecule name for this file
            if molecule_names is not None and i < len(molecule_names):
                molecule_name = molecule_names[i]
            else:
                # Extract molecule name from file name
                molecule_file_name = os.path.basename(hitran_file)
                molecule_name = molecule_file_name
                # Clean up the molecule name for use as a Python identifier and display
                molecule_name = molecule_name.translate({ord(i): None for i in '_$^{}'})
                molecule_name = molecule_name.translate({ord(i): "_" for i in ' -'})
                if molecule_name and molecule_name[0].isdigit():
                    molecule_name = 'm_' + molecule_name
                molecule_name = molecule_name.upper()
            
            print(f"Loading molecule '{molecule_name}' from file: {hitran_file}")
            
            try:
                new_molecule = Molecule(
                    hitran_data=hitran_file,
                    name=molecule_name,
                    wavelength_range=self.wavelength_range,
                    initial_molecule_parameters=self.initial_molecule_parameters.get(molecule_name, self.molecules_parameters_default)
                )
                new_molecules.append(new_molecule)
                print(f"Successfully created molecule: {molecule_name}")
                
            except Exception as e:
                print(f"Error loading molecule '{molecule_name}' from {hitran_file}: {str(e)}")
                continue
        
        if new_molecules:
            self.molecules_dict.add_molecules(new_molecules)
            print(f"Added {len(new_molecules)} molecules to the dictionary.")
            
            if refresh:
                if hasattr(self, "GUI"):
                    if hasattr(self.GUI, "molecule_table"):
                        self.GUI.molecule_table.update_table()
                    if hasattr(self.GUI, "control_panel"):
                        self.GUI.control_panel.reload_molecule_dropdown()
                    if hasattr(self.GUI, "plot"):
                        self.GUI.plot.update_all_plots()
        else:
            print("No molecules were successfully loaded.")

    def check_HITRAN(self):
        """
        Checks that all expected HITRAN files are present,
        loads them using read_HITRAN_data, and optionally
        stores them in self.hitran_data.
        """
        print("\nChecking HITRAN files:")

        if self.user_settings["first_startup"] or self.user_settings["reload_default_files"]:
            print('First startup or reload_default_files is True. Downloading default HITRAN files ...')
            self.hitran_data = {}  # option: central dict holding lines by molecule name

            for mol, bm, iso in zip(self.mols, self.basem, self.isot):
                hitran_file = f"HITRANdata/data_Hitran_2020_{mol}.par"
                if not os.path.exists(hitran_file):
                    print(f"WARNING: HITRAN file for {mol} not found at {hitran_file}")
                    self.hitran_data[mol] = []
                    continue

                lines = read_HITRAN_data(hitran_file)
                if lines:
                    #print(f"Loaded HITRAN file for {mol}: {len(lines)} lines.")
                    self.hitran_data[mol] = {"lines": lines, "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                else:
                    #print(f"WARNING: HITRAN file for {mol} could not be parsed.")
                    self.hitran_data[mol] = []
        else:
            print('Not the first startup and reload_default_files is False. Skipping HITRAN files download.')

        print("Finished HITRAN file check.\n")
    
    def load_default_molecules(self, reset=True):
        """
        Loads default molecules into the molecules_dict.
        """
        print("Loading default molecules...")
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()

        if reset:
            # Clear existing molecules if reset is True
            self.molecules_dict.clear()
            print("Resetting molecules_dict to empty.")

        self.init_molecules(self.default_molecule_csv_data())

    def create_folders(self): # see if we need this one and/or add config for directories
        """
        create_folders() creates the necessary folders for saving data and models.
        This is typically done at the first launch of iSLAT.
        """
        # Create necessary folders, if they don't exist
        os.makedirs("SAVES", exist_ok=True)
        os.makedirs("MODELS", exist_ok=True)
        os.makedirs("LINESAVES", exist_ok=True)
        os.makedirs("HITRANdata", exist_ok=True)

    def load_spectrum(self, file_path=None):
        #filetypes = [('CSV Files', '*.csv'), ('TXT Files', '*.txt'), ('DAT Files', '*.dat')]
        spectra_directory = os.path.abspath("EXAMPLE-data")
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title='Choose Spectrum Data File', 
                #filetypes=filetypes, 
                initialdir=spectra_directory
            )

        if file_path:
            # Use the new read_spectral_data function
            df = read_spectral_data(file_path)
            
            if df.empty:
                print(f"Failed to load spectrum from {file_path}")
                return
            
            # Check if required columns exist
            required_columns = ['wave', 'flux']
            optional_columns = ['err', 'cont']
            
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Required columns {required_columns} not found in {file_path}")
                print(f"Available columns: {list(df.columns)}")
                return
            
            # Load required data
            self.wave_data = np.array(df['wave'].values) * self.user_settings.get("wave_data_scalar", 1.0)
            self.wave_data_original = self.wave_data.copy()
            self.flux_data = np.array(df['flux'].values) * self.user_settings.get("flux_data_scalar", 1.0)
            
            # Load optional data with defaults if not present
            if 'err' in df.columns:
                self.err_data = np.array(df['err'].values)
            else:
                # Create default error array (e.g., 10% of flux)
                self.err_data = np.abs(self.flux_data) * 0.1
                print("Warning: No 'err' column found. Using 10% of flux as default error.")
            
            if 'cont' in df.columns:
                self.continuum_data = np.array(df['cont'].values)
            else:
                # Create default continuum array (zeros or ones)
                self.continuum_data = np.ones_like(self.flux_data)
                print("Warning: No 'cont' column found. Using ones as default continuum.")
            
            print(f"Successfully loaded spectrum from {file_path}")
            print(f"  Wavelength range: {self.wave_data.min():.3f} - {self.wave_data.max():.3f}")
            print(f"  Data points: {len(self.wave_data)}")

            # Update any dependent components if spectrum is loaded after first start
            if hasattr(self, "GUI"):
                if hasattr(self.GUI, "plot"):
                    self.GUI.plot.update_all_plots()
                #if hasattr(self.GUI, "control_panel"):
                    #self.GUI.control_panel.update_controls()
            # If model spectrum or other calculations depend on spectrum, update them
            if hasattr(self, "update_model_spectrum"):
                self.update_model_spectrum()
        else:
            print("No file selected.")

    def update_model_spectrum(self):
        summed_flux = np.zeros_like(self.wave_data)
        for mol in self.molecules_dict.values():
            if mol.is_visible:
                mol_flux = mol.get_flux(self.wave_data)
                summed_flux += mol_flux
        self.sum_spectrum_flux = summed_flux

    def get_line_data_in_range(self, xmin, xmax):
        selected_mol = self.active_molecule
        if not selected_mol:
            return None
        lines_df = selected_mol.intensity.get_table
        subset = lines_df[(lines_df['lam'] >= xmin) & (lines_df['lam'] <= xmax)]
        if subset.empty:
            return None
        return (subset['lam'].values,
                subset['intens'].values,
                subset['e_up'].values,
                subset['a_stein'].values,
                subset['g_up'].values)
    
    @property
    def active_molecule(self):
        return self._active_molecule
    
    @active_molecule.setter
    def active_molecule(self, molecule):
        """
        Sets the active molecule based on the provided name or object.
        If the molecule is not found, it throws an error and does not update the active molecule.
        """
        try:
            if isinstance(molecule, Molecule):
                self._active_molecule = molecule
            elif isinstance(molecule, str):
                if molecule in self.molecules_dict:
                    self._active_molecule = self.molecules_dict[molecule]
                else:
                    raise ValueError(f"Molecule '{molecule}' not found in the dictionary.")
            else:
                raise TypeError("Active molecule must be a Molecule object or a string representing the molecule name.")
            
            if hasattr(self, "GUI") and hasattr(self.GUI, "plot"):
                self.GUI.plot.update_all_plots()
                #self.GUI.plot.update_line_inspection_plot()

        except (ValueError, TypeError) as e:
            print(f"Error setting active molecule: {e}")
        
    @property
    def display_range(self):
        """tuple: Display range for the spectrum plot."""
        return self._display_range
    
    @display_range.setter
    def display_range(self, value):
        """
        Sets the display range for the spectrum plot.
        The value should be a tuple of two floats representing the start and end wavelengths.
        """
        if isinstance(value, tuple) and len(value) == 2:
            self._display_range = value
            if hasattr(self, "GUI") and hasattr(self.GUI, "plot"):
                self.GUI.plot.match_display_range()
        else:
            raise ValueError("Display range must be a tuple of two floats (start, end).")
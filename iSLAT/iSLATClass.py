iSLAT_version = 'v5.00.00'
print(' ')
print('Loading iSLAT ' + iSLAT_version + ': Please Wait ...')

# Import necessary modules
import numpy as np
import pandas as pd
import os
import time

from .Modules.FileHandling.iSLATFileHandling import load_user_settings, read_default_molecule_parameters, read_initial_molecule_parameters, read_save_data, read_HITRAN_data, read_from_user_csv, read_default_csv, read_spectral_data

import iSLAT.Constants as c
from .Modules.GUI import *
from .Modules.DataTypes.Molecule import Molecule
from .Modules.DataTypes.MoleculeDict import MoleculeDict
from .Modules.Debug.DebugConfig import debug_config

class iSLAT:
    """
    iSLAT class to handle the iSLAT functionalities.
    This class is used to initialize the iSLAT application, load user settings, and manage the main functionalities.
    """

    def __init__(self):
        """
        Initialize the iSLAT application with minimal setup.
        """
        # === CORE STATE ===
        self._active_molecule = None
        self.GUI = None
        
        # Initialize collections
        self.molecules_dict = MoleculeDict()
        self.callbacks = {}
        
        # === CALLBACK SYSTEM ===
        self._active_molecule_change_callbacks = []
        
        # === LAZY LOADING FLAGS ===
        self._user_settings = None
        self._initial_molecule_parameters = None
        self._molecules_parameters_default = None
        self._default_molecule_csv_data = None
        self._user_saved_molecules = None
        self._molecules_data_default = None
        self._startup_optimizations_applied = False
        self._molecules_loaded = False  # Track if molecules have been initialized
        
        # === MOLECULE CONSTANTS ===
        # Define molecule constants (use tuples for immutability and performance)
        self.mols = ("H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N")
        self.basem = ("H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N")
        self.isot = (1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1)
        
        # === PHYSICAL PARAMETERS ===
        self.wavelength_range = c.WAVELENGTH_RANGE
        self._display_range = (23.52, 25.41)
        self._dist = c.DEFAULT_DISTANCE
        self._star_rv = c.DEFAULT_STELLAR_RV
        self._fwhm = c.DEFAULT_FWHM
        self._intrinsic_line_width = c.INTRINSIC_LINE_WIDTH

        # === DATA CONTAINERS ===
        self.hitran_data = {}
        self._hitran_file_cache = {}  # Cache for HITRAN file data to avoid re-reading
        self.deleted_molecules = []
        self.input_line_list = None
        self.output_line_measurements = None
        
        # === PERFORMANCE FLAGS ===
        self._use_parallel_processing = False
        self._defer_spectrum_rendering = False  # Flag to defer spectrum rendering during initialization
        self._batch_update_in_progress = False  # Flag to prevent redundant updates during batch operations

    # === LAZY LOADING PROPERTIES ===
    @property
    def user_settings(self):
        """Lazy load user settings only when needed."""
        if self._user_settings is None:
            self._user_settings = load_user_settings()
        return self._user_settings

    @property
    def initial_molecule_parameters(self):
        """Lazy load initial molecule parameters only when needed."""
        if self._initial_molecule_parameters is None:
            self._initial_molecule_parameters = read_initial_molecule_parameters()
        return self._initial_molecule_parameters

    @property
    def molecules_parameters_default(self):
        """Lazy load default molecule parameters only when needed."""
        if self._molecules_parameters_default is None:
            self._molecules_parameters_default = read_default_molecule_parameters()
        return self._molecules_parameters_default

    @property
    def default_molecule_csv_data(self):
        """Lazy load default molecule CSV data with safe error handling."""
        return self._safe_load_data(
            read_default_csv,
            '_default_molecule_csv_data', 
            "Failed to load default molecules"
        )
    
    @default_molecule_csv_data.setter
    def default_molecule_csv_data(self, value):
        self._default_molecule_csv_data = value

    @property
    def molecules_data_default(self):
        """Lazy load default molecules data only when needed."""
        if self._molecules_data_default is None:
            self._molecules_data_default = c.MOLECULES_DATA.copy()
        return self._molecules_data_default
    
    @property
    def user_saved_molecules(self):
        """Lazy load user saved molecules data with safe error handling."""
        return self._safe_load_data(
            read_from_user_csv, 
            '_user_saved_molecules',
            "Failed to load user molecules"
        )
        
    @user_saved_molecules.setter 
    def user_saved_molecules(self, value):
        """Set user saved molecules data."""
        self._user_saved_molecules = value

    # === INITIALIZATION METHODS ===
    def initialize_application(self, load_molecules=True, enable_optimizations=True):
        """Single method to fully initialize the application."""
        if enable_optimizations:
            self.apply_startup_optimizations()
        
        if load_molecules:
            self.init_molecules()
        
        return self

    def apply_startup_optimizations(self, force=False, minimal=True):
        """
        Apply startup optimizations with optional minimal mode for fast startup.
        
        Parameters
        ----------
        force : bool, default False
            Apply optimizations even if already applied.
        minimal : bool, default True
            If True, only apply minimal optimizations for fast startup.
        """
        if self._startup_optimizations_applied and not force:
            return
            
        # Initialize molecules_dict if not already done
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()
        
        if minimal:
            # Only do essential setup for fast startup
            self._setup_minimal_structures()
        else:
            # Apply full optimization methods
            self._optimize_imports()
            self._configure_performance_settings()
            self._setup_parallel_processing()
            
        self._startup_optimizations_applied = True

    def _setup_minimal_structures(self):
        """Set up only the minimal structures needed for GUI startup"""
        # Set basic parameters without heavy initialization
        self.molecules_dict._global_dist = self._dist
        self.molecules_dict._global_star_rv = self._star_rv
        self.molecules_dict._global_fwhm = self._fwhm
        self.molecules_dict._global_intrinsic_line_width = self._intrinsic_line_width

    def apply_full_optimizations(self):
        """Apply full optimizations when molecules are actually being loaded"""
        # Set up HITRAN file caching to avoid re-reading the same files
        if hasattr(self.molecules_dict, '_setup_hitran_cache'):
            self.molecules_dict._setup_hitran_cache(self._hitran_file_cache)
        elif hasattr(self.molecules_dict, '_hitran_cache'):
            self.molecules_dict._hitran_cache = self._hitran_file_cache
        
        # Pre-load commonly used HITRAN files based on spectrum range
        if hasattr(self, 'wave_data'):
            self._preload_hitran_files()
        
        # Apply MoleculeDict optimizations
        if hasattr(self.molecules_dict, 'optimize_startup_loading'):
            self.molecules_dict.optimize_startup_loading()
        
        # Ensure bulk calculation methods are used
        self._ensure_bulk_calculations()
            
        self._configure_performance_settings()
        self._setup_parallel_processing()
        
    def _preload_hitran_files(self):
        """Pre-load commonly used HITRAN files to avoid repeated file I/O"""
        spectrum_range = (self.wave_data.min(), self.wave_data.max())
        
        # Identify likely molecules based on wavelength range
        common_files = []
        if spectrum_range[0] < 30:  # IR range
            common_files = [
                "data_Hitran_2020_H2O.par",
                "data_Hitran_2020_CO2.par", 
                "data_Hitran_2020_CO.par",
                "data_Hitran_2020_OH.par",
                "data_Hitran_2020_HCN.par",
                "data_Hitran_2020_C2H2.par"
            ]
        
        # Set up file cache for the MoleculeDict to use
        for filename in common_files:
            filepath = f"DATAFILES/HITRANdata/{filename}"
            if os.path.exists(filepath):
                self._hitran_file_cache[filepath] = None  # Mark for caching
                
    def _ensure_bulk_calculations(self):
        """Ensure that intensity calculations use bulk methods for efficiency"""
        if hasattr(self.molecules_dict, 'enable_bulk_calculations'):
            self.molecules_dict.enable_bulk_calculations()
        
        # Set up bulk calculation parameters
        if hasattr(self.molecules_dict, 'set_bulk_calculation_mode'):
            self.molecules_dict.set_bulk_calculation_mode(True)
            
        # Ensure intensity caching is properly configured
        self._configure_intensity_caching()
    
    def _configure_intensity_caching(self):
        """Configure intensity caching to prevent unnecessary recalculations"""
        for molecule_name, molecule in self.molecules_dict.items():
            if hasattr(molecule, '_intensity_cache'):
                # Ensure cache is properly initialized
                if not isinstance(molecule._intensity_cache, dict):
                    molecule._intensity_cache = {'data': None, 'hash': None}
                    
            if hasattr(molecule, '_dirty_flags'):
                # Initialize dirty flags if not present
                if not isinstance(molecule._dirty_flags, dict):
                    molecule._dirty_flags = {'intensity': True, 'spectrum': True}

    def _optimize_imports(self):
        """Private method for import optimizations."""
        # Apply MoleculeDict optimizations
        if hasattr(self.molecules_dict, 'optimize_startup_loading'):
            self.molecules_dict.optimize_startup_loading()

    def _configure_performance_settings(self):
        """Private method for performance configuration."""
        # Set global parameters efficiently
        for attr, value in [
            ('global_dist', self._dist),
            ('global_star_rv', self._star_rv),
            ('global_fwhm', self._fwhm),
            ('global_intrinsic_line_width', self._intrinsic_line_width),
            ('global_wavelength_range', self.wavelength_range)
        ]:
            setattr(self.molecules_dict, attr, value)

    def _setup_parallel_processing(self):
        """Private method for parallel processing setup."""
        # Enable parallel processing for molecule loading if beneficial
        # Based on the output, 6 molecules take 4.5 seconds - parallel could help
        self._use_parallel_processing = True

    # === MOLECULE MANAGEMENT METHODS ===
    def init_molecules(self, mole_save_data=None, use_fast_method=None, spectrum_optimized=False):
        """
        Initialize molecules with automatic optimization and spectrum-aware loading.
        
        Parameters
        ----------
        mole_save_data : dict, list, or None
            Molecule data to load. If None, loads from user_saved_molecules.
        use_fast_method : bool or None
            Use optimized loading. If None, auto-determines based on data size.
        spectrum_optimized : bool, default False
            If True, optimizes loading for the loaded spectrum's wavelength range.
        """
        # Auto-determine fast method if not specified
        if use_fast_method is None:
            if mole_save_data is None:
                use_fast_method = spectrum_optimized  # Use fast method for spectrum-optimized loading
            elif isinstance(mole_save_data, (list, dict)):
                use_fast_method = len(mole_save_data) > 5 if spectrum_optimized else len(mole_save_data) > 10  # Lower threshold for spectrum-optimized
            else:
                use_fast_method = False
        
        # If spectrum is loaded, optimize for its wavelength range
        if spectrum_optimized and hasattr(self, 'wave_data'):
            spectrum_range = (self.wave_data.min(), self.wave_data.max())
            self.wavelength_range = spectrum_range
        
        # Use parallel processing for faster loading if enabled
        use_parallel = self._use_parallel_processing if hasattr(self, '_use_parallel_processing') else False
        
        if use_fast_method:
            return self._init_molecules_optimized(mole_save_data, use_parallel)
        else:
            return self._init_molecules_standard(mole_save_data)

    def _init_molecules_optimized(self, mole_save_data, use_parallel=False):
        """Private method for optimized molecule loading with optional parallel processing."""
        # Apply startup optimizations if not already done
        self.apply_startup_optimizations()
        
        # Lazy load user_saved_molecules if needed
        if mole_save_data is None:
            mole_save_data = self.user_saved_molecules

        # Validate and process input data
        if not mole_save_data:
            print("Warning: No molecule data provided to optimized initialization")
            return

        # Convert to list format efficiently
        molecules_list = self._prepare_molecules_list(mole_save_data)
        if not molecules_list:
            print("No new molecules to load.")
            return

        # Use optimized loading
        try:
            start_time = time.time()
            
            # Use the new optimized loading method with optional parallel processing
            results = self.molecules_dict.load_molecules_optimized(
                molecules_list, 
                self.initial_molecule_parameters,
                use_parallel=use_parallel
            )
            
            elapsed_time = time.time() - start_time
            print(f"✓ Loaded {len(results)} molecules in {elapsed_time:.3f}s")
            
            self._report_loading_results(results)
            self._set_initial_active_molecule()
                    
        except Exception as e:
            print(f"Error in optimized molecule initialization: {e}")
            # Fallback to regular loading
            print("Falling back to standard sequential loading...")
            self._init_molecules_standard(mole_save_data)

    def _init_molecules_standard(self, mole_save_data):
        """Private method for standard molecule loading."""
        # Lazy load user_saved_molecules if needed
        if mole_save_data is None:
            mole_save_data = self.user_saved_molecules

        # Initialize molecules_dict if needed
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()
            self._configure_performance_settings()
        
        # Validate and process input data
        if not mole_save_data:
            print("Warning: No molecule data provided to standard initialization")
            return

        molecules_list = self._prepare_molecules_list(mole_save_data)
        if not molecules_list:
            print("No new molecules to load.")
            return

        # Standard sequential loading 
        try:
            start_time = time.time()
            
            success_count = 0
            errors = []
            for mol_data in molecules_list:
                mol_name = mol_data.get("Molecule Name")
                try:
                    new_molecule = Molecule(
                        user_save_data=mol_data,
                        wavelength_range=self.wavelength_range,
                        initial_molecule_parameters=self.initial_molecule_parameters.get(mol_name, self.molecules_parameters_default)
                    )
                    self.molecules_dict[mol_name] = new_molecule
                    success_count += 1
                except Exception as e:
                    error_msg = f"Error creating molecule '{mol_name}': {e}"
                    errors.append(error_msg)
                    print(error_msg)
            
            elapsed_time = time.time() - start_time
            
            # Report results in consistent format
            results = {
                "success": success_count,
                "failed": len(errors),
                "errors": errors
            }
            self._report_loading_results(results)
            self._set_initial_active_molecule()
                    
        except Exception as e:
            print(f"Error in standard molecule initialization: {e}")
            raise

    def _prepare_molecules_list(self, mole_save_data):
        """Helper method to prepare molecules list from various input formats."""
        if isinstance(mole_save_data, dict):
            molecules_list = [mol for mol in mole_save_data.values() 
                            if mol.get("Molecule Name") and mol.get("Molecule Name") not in self.molecules_dict]
        elif isinstance(mole_save_data, list):
            molecules_list = [mol for mol in mole_save_data 
                            if mol.get("Molecule Name") and mol.get("Molecule Name") not in self.molecules_dict]
        else:
            print(f"Warning: Unsupported molecule data format: {type(mole_save_data)}")
            return []
        
        return molecules_list

    def _report_loading_results(self, results):
        """Helper method to report loading results in consistent format."""
        if results["success"] > 0:
            print(f"Loaded {results['success']} molecules")
            
        if results["failed"] > 0:
            print(f"Failed to load {results['failed']} molecules")
            # Only show first 3 errors to reduce clutter
            for error in results["errors"][:3]:
                print(f"  - {error}")
            if len(results["errors"]) > 3:
                print(f"  ... and {len(results['errors']) - 3} more errors")

    def _set_initial_active_molecule(self):
        """Set the initial active molecule based on user settings and available molecules."""
        active_molecule_name = self.user_settings.get("default_active_molecule", "H2O")
        
        if active_molecule_name in self.molecules_dict:
            self._active_molecule = self.molecules_dict[active_molecule_name]
        else:
            # Try to fall back to H2O
            if "H2O" in self.molecules_dict:
                print(f"Active molecule '{active_molecule_name}' not found. Defaulting to 'H2O'.")
                self._active_molecule = self.molecules_dict["H2O"]
            elif len(self.molecules_dict) > 0:
                # Use the first available molecule
                first_molecule = next(iter(self.molecules_dict.values()))
                print(f"Neither '{active_molecule_name}' nor 'H2O' found. Using '{first_molecule.name}'.")
                self._active_molecule = first_molecule
            else:
                print("No molecules available to set as active.")
                self._active_molecule = None

    def add_molecule_from_hitran(self, refresh=True, hitran_files=None, molecule_names=None, 
                                base_molecules=None, isotopes=None, use_parallel=False):
        """
        Adds one or more molecules to the iSLAT instance from HITRAN files with sequential loading by default.
        
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
        use_parallel : bool, default False
            Whether to use parallel loading (multiprocessing). Default is False for sequential loading.
        """
        if hitran_files is None:
            hitran_files = GUI.file_selector(title='Choose HITRAN Data Files (select multiple with Ctrl/Cmd)',
                                                  filetypes=[('PAR Files', '*.par')],
                                                  initialdir=os.path.abspath("DATAFILES/HITRANdata"))
            
        if not hitran_files:
            print("No HITRAN files selected.")
            return
        
        # Convert single file to list for consistent processing
        if isinstance(hitran_files, str):
            hitran_files = [hitran_files]
        
        # Convert molecule_names to list if provided as single string
        if molecule_names is not None and isinstance(molecule_names, str):
            molecule_names = [molecule_names]
        
        # Prepare molecule data for parallel processing
        molecules_data = []
        
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
            
            # Prepare molecule data dictionary for parallel processing
            mol_data = {
                "name": molecule_name,
                "Molecule Name": molecule_name,
                "file": hitran_file,
                "hitran_data": hitran_file,
                "label": molecule_name,
                "Molecule Label": molecule_name
            }
            molecules_data.append(mol_data)
        
        # Use sequential loading by default, parallel only if explicitly enabled
        success_count = 0
        use_parallel_loading = use_parallel or self.use_parallel_processing
        
        if use_parallel_loading and len(molecules_data) > 1:
            print(f"Loading {len(molecules_data)} HITRAN molecules using parallel method...")
            results = self.molecules_dict.load_molecules_ultra_fast(
                molecules_data, 
                self.initial_molecule_parameters,
                force_multiprocessing=True
            )
            success_count = results["success"]
            
            if results["failed"] > 0:
                print(f"Failed to load {results['failed']} molecules:")
                for error in results["errors"]:
                    print(f"  - {error}")
        else:
            # Sequential loading (default)
            print(f"Loading {len(molecules_data)} HITRAN molecule(s) sequentially...")
            start_time = time.time()
            
            for mol_data in molecules_data:
                molecule_name = mol_data["name"]
                hitran_file = mol_data["file"]
                
                print(f"Loading molecule '{molecule_name}' from file: {hitran_file}")
                
                try:
                    new_molecule = Molecule(
                        hitran_data=hitran_file,
                        name=molecule_name,
                        wavelength_range=self.wavelength_range,
                        initial_molecule_parameters=self.initial_molecule_parameters.get(molecule_name, self.molecules_parameters_default)
                    )
                    self.molecules_dict[molecule_name] = new_molecule
                    success_count += 1
                    print(f"Successfully created molecule: {molecule_name}")
                    
                except Exception as e:
                    print(f"Error loading molecule '{molecule_name}' from {hitran_file}: {str(e)}")
                    continue
            
            elapsed_time = time.time() - start_time
            print(f"Sequential loading completed in {elapsed_time:.3f}s")
        
        if success_count > 0:
            print(f"Successfully loaded {success_count} molecules.")
            
            # Use the optimized update system
            if refresh:
                self._update_gui_after_molecule_load()
        else:
            print("No molecules were successfully loaded.")

    def check_HITRAN(self):
        """
        Checks that all expected HITRAN files are present and loads them efficiently.
        Only loads when specifically requested to avoid startup delays.
        """
        if not self.user_settings.get("auto_load_hitran", False):
            print("HITRAN auto-loading disabled. Files will be loaded on demand.")
            return
            
        print("Checking HITRAN files:")

        if self.user_settings.get("first_startup", False) or self.user_settings.get("reload_default_files", False):
            print('First startup or reload_default_files is True. Loading default HITRAN files ...')
            
            for mol, bm, iso in zip(self.mols, self.basem, self.isot):
                hitran_file = f"HITRANdata/data_Hitran_2020_{mol}.par"
                if not os.path.exists(hitran_file):
                    print(f"WARNING: HITRAN file for {mol} not found at {hitran_file}")
                    self.hitran_data[mol] = {"lines": [], "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                    continue

                try:
                    lines = read_HITRAN_data(hitran_file)
                    if lines:
                        self.hitran_data[mol] = {"lines": lines, "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                    else:
                        self.hitran_data[mol] = {"lines": [], "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                except Exception as e:
                    print(f"ERROR: Failed to load HITRAN file for {mol}: {e}")
                    self.hitran_data[mol] = {"lines": [], "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
        else:
            print('Not the first startup and reload_default_files is False. Skipping HITRAN files loading.')

        print("Finished HITRAN file check.\n")
    
    def load_default_molecules(self, reset=True, use_parallel=False):
        """
        Loads default molecules into the molecules_dict with sequential loading by default.
        
        Parameters
        ----------
        reset : bool, optional
            If True, clears existing molecules before loading defaults. Default is True.
        use_parallel : bool, optional
            If True, uses parallel loading. Default is False for sequential loading.
        """
        print("Loading default molecules...")
        
        # Initialize molecules_dict if needed
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()
            # Set global parameters efficiently
            for attr, value in [
                ('global_dist', self._dist),
                ('global_star_rv', self._star_rv),
                ('global_fwhm', self._fwhm),
                ('global_intrinsic_line_width', self._intrinsic_line_width),
                ('global_wavelength_range', self.wavelength_range)
            ]:
                setattr(self.molecules_dict, attr, value)

        if reset:
            self.molecules_dict.clear()
            print("Resetting molecules_dict to empty.")

        try:
            # Lazy load default molecule data
            if self.default_molecule_csv_data is None:
                print("Loading default molecule CSV data...")
                self.default_molecule_csv_data = read_default_csv()
                
            if not self.default_molecule_csv_data:
                print("Error: Could not load default molecule CSV data.")
                return
            
            # Use sequential loading by default
            use_parallel_loading = use_parallel or self.use_parallel_processing
            self.init_molecules(self.default_molecule_csv_data, use_optimized_loading=use_parallel_loading)
            print(f"Successfully loaded {len(self.molecules_dict)} default molecules.")
            
            # Update GUI components if they exist
            if hasattr(self, "GUI") and self.GUI is not None:
                self._update_gui_after_molecule_load()
                
        except Exception as e:
            print(f"Error loading default molecules: {e}")
            raise

    def _update_gui_after_molecule_load(self):
        """
        Helper method to update GUI components after molecules are loaded.
        """
        try:
            # Update molecule table if it exists
            if (hasattr(self.GUI, "molecule_table") and self.GUI.molecule_table is not None):
                self.GUI.molecule_table.update_table()
            
            # Update control panel dropdown if it exists
            if (hasattr(self.GUI, "control_panel") and self.GUI.control_panel is not None and
                hasattr(self.GUI.control_panel, "reload_molecule_dropdown")):
                self.GUI.control_panel.reload_molecule_dropdown()
            
            '''# Update plots if they exist
            if (hasattr(self.GUI, "plot") and self.GUI.plot is not None):
                self.GUI.plot.update_all_plots()'''
                
        except Exception as e:
            print(f"Warning: Error updating GUI after molecule load: {e}")

    def _initialize_molecules_for_spectrum(self):
        """
        Initialize molecules optimized for the loaded spectrum's wavelength range.
        This method is called automatically after a spectrum is loaded.
        """
        if not hasattr(self, 'wave_data'):
            print("Warning: No spectrum data available for molecule optimization")
            return
            
        # Optimize wavelength range for the loaded spectrum
        spectrum_range = (self.wave_data.min(), self.wave_data.max())
        print(f"Initializing molecules for spectrum range: {spectrum_range[0]:.1f} - {spectrum_range[1]:.1f} µm")
        
        # Set optimized wavelength range before loading molecules
        self.wavelength_range = spectrum_range
        
        try:
            # Apply full optimizations now that we're loading molecules
            if not hasattr(self, '_full_optimizations_applied'):
                print("Applying optimizations for molecule loading...")
                self.apply_full_optimizations()
                self._full_optimizations_applied = True
            
            # Initialize molecules with spectrum-optimized settings
            start_time = time.time()
            
            # Use the most efficient initialization method with spectrum optimization
            self.init_molecules(use_fast_method=True, spectrum_optimized=True)
            
            elapsed_time = time.time() - start_time
            self._molecules_loaded = True
            
            print(f"✓ Molecule initialization completed in {elapsed_time:.3f}s")
            print(f"✓ Loaded {len(self.molecules_dict)} molecules optimized for spectrum")
            
            # Print performance summary
            self._print_performance_summary(elapsed_time)
                
        except Exception as e:
            print(f"Error initializing molecules for spectrum: {e}")
            self._molecules_loaded = False

    def _print_performance_summary(self, molecule_load_time):
        """Print a summary of performance optimizations and timings"""
        print(f"\n--- Performance Summary ---")
        print(f"Molecule loading time: {molecule_load_time:.3f}s")
        if hasattr(self, '_use_parallel_processing') and self._use_parallel_processing:
            print("✓ Parallel processing enabled")
        if hasattr(self, '_hitran_file_cache') and len(self._hitran_file_cache) > 0:
            print(f"✓ HITRAN file caching active ({len(self._hitran_file_cache)} files cached)")
        if hasattr(self, 'wavelength_range'):
            print(f"✓ Optimized for spectrum range: {self.wavelength_range[0]:.1f} - {self.wavelength_range[1]:.1f} µm")
        print("--- Ready for Analysis ---\n")

    def _safe_update_gui_components(self):
        """Safely update GUI components with error handling for destroyed widgets"""
        try:
            if not hasattr(self, "GUI") or self.GUI is None:
                return
                
            # Check if GUI still exists
            if hasattr(self.GUI, 'master') and hasattr(self.GUI.master, 'winfo_exists'):
                if not self.GUI.master.winfo_exists():
                    return
            
            # Update file interaction pane if it exists
            if hasattr(self.GUI, "file_interaction_pane") and hasattr(self, 'loaded_spectrum_name'):
                try:
                    self.GUI.file_interaction_pane.update_file_label(self.loaded_spectrum_name)
                except Exception:
                    pass  # Silently ignore GUI errors
                    
            # Don't call other plot updates that might access destroyed widgets
            # The main spectrum should already be displayed by update_model_plot()
                    
        except Exception:
            # Silently ignore all GUI-related errors during updates
            pass
    
    def load_molecules_manually(self):
        """
        Manually load molecules without spectrum optimization.
        This method can be called if the user wants to load molecules before loading a spectrum.
        """
        if self._molecules_loaded:
            print("Molecules are already loaded.")
            return
            
        print("Loading molecules in manual mode (not spectrum-optimized)...")
        
        try:
            # Apply full optimizations
            if not hasattr(self, '_full_optimizations_applied'):
                self.apply_full_optimizations()
                self._full_optimizations_applied = True
            
            start_time = time.time()
            
            # Initialize molecules with default settings
            self.init_molecules(use_fast_method=True, spectrum_optimized=False)
            
            elapsed_time = time.time() - start_time
            self._molecules_loaded = True
            
            print(f"Manual molecule loading completed in {elapsed_time:.3f}s")
            print(f"Loaded {len(self.molecules_dict)} molecules")
            
            # Update GUI status if GUI exists
            if hasattr(self, "GUI") and self.GUI is not None and hasattr(self.GUI, "data_field"):
                self.GUI.data_field.insert_text(f"Ready - {len(self.molecules_dict)} molecules loaded", clear_first=True)
                
        except Exception as e:
            print(f"Error in manual molecule loading: {e}")
            self._molecules_loaded = False
            
            # Update GUI with error status if GUI exists
            if hasattr(self, "GUI") and self.GUI is not None and hasattr(self.GUI, "data_field"):
                self.GUI.data_field.insert_text(f"Error loading molecules: {e}", clear_first=True)

    # === SPECTRUM METHODS ===
    def load_spectrum(self, file_path=None):
        """
        Load a spectrum from file or show file dialog.
        
        Parameters
        ----------
        file_path : str, optional
            Path to spectrum file. If None, shows file dialog.
            
        Returns
        -------
        bool
            True if spectrum loaded successfully, False otherwise.
            
        Raises
        ------
        FileNotFoundError
            If file_path doesn't exist.
        ValueError  
            If file format is not supported.
        """
        #filetypes = [('CSV Files', '*.csv'), ('TXT Files', '*.txt'), ('DAT Files', '*.dat')]
        spectra_directory = os.path.abspath("DATAFILES/EXAMPLE-data")
        if file_path is None:
            file_path = GUI.file_selector(
                title='Choose Spectrum Data File',
                initialdir=spectra_directory
            )

        if file_path:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Spectrum file not found: {file_path}")
                
                # Use the new read_spectral_data function
                df = read_spectral_data(file_path)
                
                if df.empty:
                    print(f"Failed to load spectrum from {file_path}")
                    return False
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return False
            except Exception as e:
                print(f"Error loading spectrum from {file_path}: {e}")
                return False
            
            # Check if required columns exist
            required_columns = ['wave', 'flux']
            optional_columns = ['err', 'cont']
            
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Required columns {required_columns} not found in {file_path}")
                print(f"Available columns: {list(df.columns)}")
                return False
            
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

            # Store the loaded file path and name
            self.loaded_spectrum_file = file_path
            self.loaded_spectrum_name = os.path.basename(file_path)

            # Initialize molecules after spectrum is loaded (most efficient approach)
            if not self._molecules_loaded:
                self._initialize_molecules_for_spectrum()
            else:
                # Update existing molecules with new wavelength range if needed
                spectrum_range = (self.wave_data.min(), self.wave_data.max())
                self.bulk_update_molecule_parameters({'wavelength_range': spectrum_range})
                print(f"Updated existing molecules for new wavelength range: {spectrum_range[0]:.3f} - {spectrum_range[1]:.3f}")

            # Initialize GUI after molecules are loaded
            if not hasattr(self, "GUI") or self.GUI is None:
                print("Initializing GUI with spectrum and molecules...")
                
                # Initialize GUI - spectrum display will happen automatically during init
                self.init_gui()
            else:
                # GUI already exists, just update the spectrum display
                print("Updating existing GUI with new spectrum...")
                if hasattr(self.GUI, "plot") and self.GUI.plot is not None:
                    self.GUI.plot.update_model_plot()
                    if hasattr(self.GUI.plot, 'canvas'):
                        self.GUI.plot.canvas.draw()
                        
                # Update file label
                if (hasattr(self.GUI, "file_interaction_pane") and 
                    hasattr(self, 'loaded_spectrum_name')):
                    self.GUI.file_interaction_pane.update_file_label(self.loaded_spectrum_name)
            # If model spectrum or other calculations depend on spectrum, update them
            if hasattr(self, "update_model_spectrum"):
                self.update_model_spectrum()
            
            return True
        else:
            print("No file selected.")
            return False

    def update_model_spectrum(self, force_recalculate=False, use_parallel=False):
        """
        Update model spectrum using sequential calculations by default.
        
        Parameters
        ----------
        force_recalculate : bool, default False
            If True, forces recalculation of all molecule intensities and spectra.
        use_parallel : bool, default False
            If True, uses parallel recalculation. Default is False for sequential processing.
            
        Returns
        -------
        None
            Updates self.sum_spectrum_flux with the calculated model spectrum.
        """
        if not hasattr(self, 'molecules_dict') or not hasattr(self, 'wave_data'):
            self.sum_spectrum_flux = np.array([])
            return
        
        if force_recalculate:
            use_parallel_calc = use_parallel or self.use_parallel_processing
            
            if use_parallel_calc:
                # Use parallel recalculation only if explicitly enabled
                print("Force recalculating all molecule spectra using parallel processing...")
                self.molecules_dict.bulk_recalculate_parallel()
            else:
                # Sequential recalculation (default)
                print("Force recalculating all molecule spectra sequentially...")
                self.molecules_dict.bulk_recalculate_sequential()
        
        try:
            # Use the optimized cached summed flux from MoleculeDict
            self.sum_spectrum_flux = self.molecules_dict.get_summed_flux(self.wave_data, visible_only=True)
            
            # Update individual molecule fluxes if needed
            self.molecules_dict.update_molecule_fluxes(self.wave_data)
            
        except Exception as e:
            print(f"Error updating model spectrum: {e}")
            self.sum_spectrum_flux = np.zeros_like(self.wave_data) if hasattr(self, 'wave_data') else np.array([])
    
    def bulk_update_molecule_parameters(self, parameter_dict, molecule_names=None, update_plots=True):
        """
        Bulk update parameters for multiple molecules using the optimized MoleculeDict methods.
        
        Parameters
        ----------
        parameter_dict : dict
            Dictionary of parameter names and values to update
        molecule_names : list, optional
            List of molecule names to update (None for all molecules)
        update_plots : bool, default True
            Whether to update plots after parameter changes
        """
        if not hasattr(self, 'molecules_dict'):
            print("No molecules_dict available for bulk update")
            return
        
        try:
            # Use the optimized bulk parameter update
            self.molecules_dict.bulk_update_parameters(parameter_dict, molecule_names)
            
            # Update model spectrum and plots if requested
            if update_plots:
                self.update_model_spectrum()
                
        except Exception as e:
            print(f"Error in bulk parameter update: {e}")
    
    # === PARAMETER METHODS ===
    def set_global_parameters(self, **kwargs):
        """
        Set global parameters with validation and callback support.
        
        Parameters
        ----------
        **kwargs : dict
            Global parameters to set (distance, fwhm, stellar_rv, etc.)
            
        Returns
        -------
        dict
            Dictionary of parameters that were actually changed
        """
        if not hasattr(self, 'molecules_dict'):
            return {}
        
        # Validate inputs first
        valid_params = {
            'distance', 'fwhm', 'stellar_rv', 'intrinsic_line_width', 
            'wavelength_range', 'temperature', 'column_density'
        }
        
        invalid_params = set(kwargs.keys()) - valid_params
        if invalid_params:
            raise ValueError(f"Invalid parameters: {invalid_params}")
        
        # Validate value types and ranges
        for param, value in kwargs.items():
            if param in ['distance', 'fwhm'] and value <= 0:
                raise ValueError(f"{param} must be positive, got {value}")
            if param == 'temperature' and value < 0:
                raise ValueError(f"Temperature cannot be negative, got {value}")
        
        parameters_changed = {}
        
        try:
            # Update global parameters using the new system
            for param_name, value in kwargs.items():
                old_value = None
                if param_name == 'distance':
                    old_value = getattr(self, '_dist', None)
                    if old_value != value:
                        self.molecules_dict.global_dist = value
                        self._dist = value
                        parameters_changed[param_name] = {'old': old_value, 'new': value}
                elif param_name == 'fwhm':
                    old_value = getattr(self, '_fwhm', None)
                    if old_value != value:
                        self.molecules_dict.global_fwhm = value
                        self._fwhm = value
                        parameters_changed[param_name] = {'old': old_value, 'new': value}
                elif param_name == 'stellar_rv':
                    old_value = getattr(self, '_star_rv', None)
                    if old_value != value:
                        self.molecules_dict.global_star_rv = value
                        self._star_rv = value
                        parameters_changed[param_name] = {'old': old_value, 'new': value}
                elif param_name == 'intrinsic_line_width':
                    old_value = getattr(self, '_intrinsic_line_width', None)
                    if old_value != value:
                        self.molecules_dict.global_intrinsic_line_width = value
                        self._intrinsic_line_width = value
                        parameters_changed[param_name] = {'old': old_value, 'new': value}
                elif param_name == 'wavelength_range':
                    old_value = getattr(self, 'wavelength_range', None)
                    if old_value != value:
                        self.molecules_dict.global_wavelength_range = value
                        self.wavelength_range = value
                        parameters_changed[param_name] = {'old': old_value, 'new': value}
            
            # Trigger callbacks for changed parameters
            if parameters_changed:
                self._trigger_callbacks('parameter_changed', parameters_changed)
            
            # Update model spectrum and plots
            self.update_model_spectrum()
            
            return parameters_changed
            
        except Exception as e:
            print(f"Error setting global parameters: {e}")
            return {}

    # === CALLBACK SYSTEM ===
    def register_callback(self, event_type, callback_func):
        """Register a callback for specific events."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        if callback_func not in self.callbacks[event_type]:
            self.callbacks[event_type].append(callback_func)

    def unregister_callback(self, event_type, callback_func):
        """Remove a callback for specific events.""" 
        if event_type in self.callbacks:
            self.callbacks[event_type] = [cb for cb in self.callbacks[event_type] if cb != callback_func]

    def _trigger_callbacks(self, event_type, *args, **kwargs):
        """Trigger all callbacks for an event type."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error in {event_type}: {e}")

    # === UTILITY METHODS ===
    def _safe_load_data(self, loader_func, cache_attr, error_message):
        """Helper method to safely load and cache data."""
        try:
            if not hasattr(self, cache_attr) or getattr(self, cache_attr) is None:
                data = loader_func()
                setattr(self, cache_attr, data)
            return getattr(self, cache_attr)
        except Exception as e:
            print(f"{error_message}: {e}")
            return None
    
    @property
    def active_molecule(self):
        return self._active_molecule
    
    @active_molecule.setter
    def active_molecule(self, molecule):
        """
        Sets the active molecule based on the provided name or object.
        """
        old_molecule = getattr(self, '_active_molecule', None)
        old_name = getattr(old_molecule, 'name', old_molecule)
        new_name = getattr(molecule, 'name', molecule) if hasattr(molecule, 'name') else molecule
        
        debug_config.info("active_molecule", f"Setting active molecule from {old_name} to {new_name}")
        
        try:
            if isinstance(molecule, Molecule):
                self._active_molecule = molecule
            elif isinstance(molecule, str):
                if hasattr(self, 'molecules_dict') and molecule in self.molecules_dict:
                    self._active_molecule = self.molecules_dict[molecule]
                else:
                    raise ValueError(f"Molecule '{molecule}' not found in the dictionary.")
            else:
                raise TypeError("Active molecule must be a Molecule object or a string representing the molecule name.")
            
            # Trigger callbacks using the new unified callback system
            self._trigger_callbacks('active_molecule_changed', old_molecule, self._active_molecule)
            
            # Also maintain backwards compatibility with old callback system
            debug_config.verbose("active_molecule", f"Notifying {len(self._active_molecule_change_callbacks)} callbacks of change")
            self._notify_active_molecule_change(old_molecule, self._active_molecule)
                
        except Exception as e:
            debug_config.error("active_molecule", f"Error setting active molecule: {e}")
            # Don't change the active molecule if there's an error
        
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
    
    def add_active_molecule_change_callback(self, callback):
        """Add a callback function to be called when active molecule changes"""
        self._active_molecule_change_callbacks.append(callback)
    
    def remove_active_molecule_change_callback(self, callback):
        """Remove a callback function for active molecule changes"""
        if callback in self._active_molecule_change_callbacks:
            self._active_molecule_change_callbacks.remove(callback)
    
    def _notify_active_molecule_change(self, old_molecule, new_molecule):
        """Notify all callbacks that the active molecule has changed"""
        debug_config.verbose("active_molecule", f"Notifying {len(self._active_molecule_change_callbacks)} callbacks")
        for i, callback in enumerate(self._active_molecule_change_callbacks):
            try:
                callback_name = callback.__name__ if hasattr(callback, '__name__') else str(callback)
                debug_config.trace("active_molecule", f"Calling callback {i+1}: {callback_name}")
                callback(old_molecule, new_molecule)
                debug_config.trace("active_molecule", f"Callback {i+1} completed successfully")
            except Exception as e:
                debug_config.error("active_molecule", f"Error in callback {i+1}: {e}")
        debug_config.verbose("active_molecule", "All callbacks completed")

    # === GUI METHODS ===
    def init_gui(self):
        """
        Initialize the GUI components of iSLAT.
        This function sets up the main window, menus, and other GUI elements.
        """
        try:
            if not hasattr(self, "GUI") or self.GUI is None:
                # Import GUI class
                from .Modules.GUI import GUI as GUIClass
                
                self.GUI = GUIClass(
                    master=None,
                    molecule_data=getattr(self, 'molecules_dict', None),
                    wave_data=getattr(self, 'wave_data', None),
                    flux_data=getattr(self, 'flux_data', None),
                    config=self.user_settings,
                    islat_class_ref=self
                )
                
                if self.GUI is None:
                    raise RuntimeError("Failed to create GUI object")
            
            if hasattr(self.GUI, 'start') and callable(self.GUI.start):
                self.GUI.start()
                
                # Immediately display spectrum after GUI starts if we have data
                if (hasattr(self, 'wave_data') and hasattr(self, 'flux_data') and 
                    hasattr(self.GUI, "plot") and self.GUI.plot is not None):
                    try:
                        print("Displaying spectrum in GUI...")
                        self.GUI.plot.update_model_plot()
                        
                        # Force immediate canvas update to ensure spectrum is visible
                        if hasattr(self.GUI.plot, 'canvas'):
                            self.GUI.plot.canvas.draw()
                            
                        print("✓ Spectrum displayed successfully")
                        
                        # Update file label if available
                        if (hasattr(self.GUI, "file_interaction_pane") and 
                            hasattr(self, 'loaded_spectrum_name')):
                            self.GUI.file_interaction_pane.update_file_label(self.loaded_spectrum_name)
                            
                    except Exception as e:
                        print(f"Warning: Error displaying spectrum during GUI init: {e}")
                        
            else:
                raise AttributeError(f"GUI object does not have a callable 'start' method")
                
        except Exception as e:
            print(f"Error initializing GUI: {e}")
            import traceback
            traceback.print_exc()
            print("GUI initialization failed. Running in headless mode.")
            self.GUI = None
            raise

    def run(self):
        """
        Run the iSLAT application with deferred GUI initialization.
        The GUI will only be created after a spectrum is loaded and molecules are initialized.
        """
        try:
            # Load save data first (lightweight)
            if hasattr(self, 'savedata') or hasattr(read_save_data, '__call__'):
                try:
                    self.savedata = read_save_data()
                except Exception as e:
                    print(f"Warning: Could not load save data: {e}")
            
            # Apply minimal startup optimizations (lightweight)
            self.apply_startup_optimizations()
            
            # Show console prompts for spectrum selection
            print("\n" + "="*60)
            print("iSLAT v5.00.00 - Infrared Spectroscopic Line Analysis Tool")
            print("="*60)
            print("Welcome! Please load a spectrum to begin analysis.")
            print("The GUI will appear after you select a spectrum file.")
            print("="*60 + "\n")
            
            # Load spectrum - this will trigger molecule loading and GUI initialization
            spectrum_loaded = self.load_spectrum()
            
            if not spectrum_loaded:
                print("No spectrum was loaded. Exiting...")
                return
            
            # If we get here, spectrum is loaded, molecules are initialized, and GUI should be running
            print("iSLAT is now ready for analysis!")
            
        except Exception as e:
            print(f"Error during iSLAT initialization: {e}")
            print(f"Error type: {type(e).__name__}")
            raise
    
    # === REMAINING PROPERTIES ===
    @property
    def molecules_data_default(self):
        """Lazy load default molecules data only when needed"""
        if self._molecules_data_default is None:
            self._molecules_data_default = c.MOLECULES_DATA.copy()
        return self._molecules_data_default

    @property
    def use_parallel_processing(self):
        """Check if parallel processing is enabled"""
        return getattr(self, '_use_parallel_processing', False)
    
    @property
    def molecules_loaded(self):
        """Check if molecules have been loaded"""
        return getattr(self, '_molecules_loaded', False)
    
    def has_spectrum_loaded(self):
        """Check if spectrum data is available"""
        return hasattr(self, 'wave_data') and hasattr(self, 'flux_data')
    
    def should_defer_rendering(self):
        """Check if spectrum rendering should be deferred during initialization"""
        return getattr(self, '_defer_spectrum_rendering', False)
    
    def _update_gui_after_initialization(self):
        """Perform a single, optimized GUI update after initialization to avoid redundant rendering"""
        if not hasattr(self, "GUI") or not hasattr(self.GUI, "plot"):
            return
            
        # Set a flag to prevent redundant updates during this batch update
        self._batch_update_in_progress = True
        
        try:
            # First, ensure spectrum data is available for model updates
            if hasattr(self, 'wave_data') and hasattr(self, 'flux_data'):
                # Update model plot once (this will render the spectrum)
                self.GUI.plot.update_model_plot()
                print(f"✓ Spectrum displayed with {len(self.wave_data)} data points")
            
            # Update population diagram if we have an active molecule
            if (hasattr(self, 'active_molecule') and self.active_molecule and
                hasattr(self.GUI.plot, 'plot_renderer')):
                try:
                    self.GUI.plot.plot_renderer.render_population_diagram(self.active_molecule)
                except Exception as e:
                    print(f"Warning: Could not update population diagram: {e}")
            
            # Update line inspection plot only if we have a valid display range
            if (hasattr(self, 'display_range') and self.display_range and 
                len(self.display_range) == 2):
                try:
                    xmin, xmax = self.display_range
                    # Avoid accessing data_field during line inspection to prevent TclError
                    self.GUI.plot.plot_spectrum_around_line(xmin, xmax, highlight_strongest=False)
                except Exception as e:
                    print(f"Warning: Could not update line inspection: {e}")
                    
        except Exception as e:
            print(f"Warning: Error during GUI update: {e}")
        finally:
            # Always clear the batch update flag
            self._batch_update_in_progress = False
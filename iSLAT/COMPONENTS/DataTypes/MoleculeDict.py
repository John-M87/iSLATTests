from iSLAT.COMPONENTS.DataTypes.Molecule import Molecule
import iSLAT.Constants as default_parms
import numpy as np

class MoleculeDict(dict):
    """A dictionary to store Molecule objects with their names as keys, and to perform operations on the collection of molecules."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fluxes = {}
        
        # Caching system for combined fluxes
        self._summed_flux_cache = {}
        self._cache_wave_data_hash = None
        
        # Global parameters that affect all molecules
        self._global_dist = default_parms.DEFAULT_DISTANCE
        self._global_star_rv = default_parms.DEFAULT_STELLAR_RV
        self._global_fwhm = default_parms.DEFAULT_FWHM
        self._global_intrinsic_line_width = default_parms.INTRINSIC_LINE_WIDTH
        self._global_wavelength_range = default_parms.WAVELENGTH_RANGE
        self._global_model_line_width = default_parms.MODEL_LINE_WIDTH
        self._global_model_pixel_res = default_parms.MODEL_PIXEL_RESOLUTION
        
        # Callbacks to notify when global parameters change
        self._global_parameter_change_callbacks = []

    def add_molecule(self, mol_entry, intrinsic_line_width=None, wavelength_range=None, model_pixel_res=None, model_line_width=None, distance=None, hitran_data=None):
        """Add a new molecule to the dictionary using molecule entry data."""
        mol_name = mol_entry["name"]

        # Use global parameters if not specifically provided
        effective_intrinsic_line_width = intrinsic_line_width if intrinsic_line_width is not None else self._global_intrinsic_line_width
        effective_wavelength_range = wavelength_range if wavelength_range is not None else self._global_wavelength_range
        effective_model_pixel_res = model_pixel_res if model_pixel_res is not None else self._global_model_pixel_res
        effective_model_line_width = model_line_width if model_line_width is not None else self._global_model_line_width
        effective_distance = distance if distance is not None else self._global_dist

        # Create a Molecule instance
        molecule = Molecule(
            name=mol_name,
            filepath=mol_entry["file"],
            displaylabel=mol_entry["label"],
            color=self.save_file_data.get(mol_name, {}).get("Color"),
            initial_molecule_parameters=self.initial_molecule_parameters.get(mol_name, {}),
            wavelength_range=effective_wavelength_range,
            broad=effective_intrinsic_line_width,
            model_pixel_res=effective_model_pixel_res,
            model_line_width=effective_model_line_width,
            distance=effective_distance,
            fwhm=self._global_fwhm,
            stellar_rv=self._global_star_rv,
            radius=self.save_file_data.get(mol_name, {}).get("Rad", None),
            temp=self.save_file_data.get(mol_name, {}).get("Temp", None),
            n_mol=self.save_file_data.get(mol_name, {}).get("N_Mol", None),
            is_visible=self.save_file_data.get(mol_name, {}).get("Vis", True),
            hitran_data=hitran_data
        )

        # Store the molecule in the dictionary
        self[mol_name] = molecule

        print(f"Molecule Initialized: {mol_name}")
        
        # Update fluxes if the molecule has plot data
        if hasattr(molecule, 'plot_flux'):
            self.fluxes[mol_name] = molecule.plot_flux
            
        return molecule

    def add_molecules(self, *molecules):
        """Add multiple molecules to the dictionary."""
        #print("Here is what I got chief:")
        #print(molecules)
        molecules = molecules[0]
        for mol in molecules:
            #print("Here is the current mol:", mol)
            if isinstance(mol, Molecule):
                #print("Adding molecule:", mol)
                self[mol.name] = mol
            else:
                raise TypeError("Expected a Molecule instance.")

    def load_molecules_data(self, molecules_data, initial_molecule_parameters, save_file_data, wavelength_range, intrinsic_line_width, model_pixel_res, model_line_width, distance, hitran_data):
        """Load multiple molecules data into the dictionary."""
        self.initial_molecule_parameters = initial_molecule_parameters
        self.save_file_data = save_file_data
        for mol_entry in molecules_data:
            self.add_molecule(
                mol_entry,
                intrinsic_line_width=intrinsic_line_width,
                wavelength_range=wavelength_range,
                model_pixel_res=model_pixel_res,
                model_line_width=model_line_width,
                distance=distance,
                hitran_data = hitran_data[mol_entry["name"]] if mol_entry["name"] in hitran_data else None
            )
    
    def clear(self):
        """Clear the dictionary of all molecules."""
        super().clear()
        self.fluxes.clear()
        print("MoleculeDict cleared.")

    def update_molecule_fluxes(self, wave_data=None):
        """Update stored fluxes for all molecules with current wave_data - with caching"""
        if wave_data is None:
            return
            
        # Create cache key for wave_data
        wave_data_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else str(wave_data)
        
        # Only update if wave_data changed
        if self._cache_wave_data_hash == wave_data_hash:
            return
            
        for mol_name, molecule in self.items():
            if hasattr(molecule, 'prepare_plot_data'):
                molecule.prepare_plot_data(wave_data)
                if hasattr(molecule, 'plot_flux'):
                    self.fluxes[mol_name] = molecule.plot_flux
        
        # Clear summed flux cache when wave data changes
        self._summed_flux_cache.clear()
        self._cache_wave_data_hash = wave_data_hash
    
    def get_summed_flux(self, wave_data, visible_only=True):
        """Get summed flux for all visible molecules with caching"""
        
        # Create cache key
        wave_data_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else str(wave_data)
        visible_molecules = [name for name, mol in self.items() if mol.is_visible] if visible_only else list(self.keys())
        cache_key = (wave_data_hash, tuple(sorted(visible_molecules)))
        
        # Check cache
        if cache_key in self._summed_flux_cache:
            return self._summed_flux_cache[cache_key]
        
        # Calculate summed flux
        summed_flux = np.zeros_like(wave_data)
        
        for mol_name in visible_molecules:
            if mol_name in self:
                molecule = self[mol_name]
                # Ensure molecule has up-to-date plot data
                molecule.prepare_plot_data(wave_data)
                if hasattr(molecule, 'plot_flux'):
                    summed_flux += molecule.plot_flux
        
        # Cache result
        self._summed_flux_cache[cache_key] = summed_flux
        return summed_flux
    
    def _clear_flux_caches(self):
        """Clear all flux caches when parameters change"""
        self._summed_flux_cache.clear()
        self._cache_wave_data_hash = None
        for molecule in self.values():
            if hasattr(molecule, '_clear_flux_caches'):
                molecule._clear_flux_caches()
    
    def get_molecule_flux(self, mol_name):
        """Get the flux for a specific molecule"""
        if mol_name in self.fluxes:
            return self.fluxes[mol_name]
        elif mol_name in self and hasattr(self[mol_name], 'plot_flux'):
            return self[mol_name].plot_flux
        return None

    def add_global_parameter_change_callback(self, callback):
        """Add a callback function to be called when global parameters change"""
        self._global_parameter_change_callbacks.append(callback)
    
    def remove_global_parameter_change_callback(self, callback):
        """Remove a callback function"""
        if callback in self._global_parameter_change_callbacks:
            self._global_parameter_change_callbacks.remove(callback)
    
    def _notify_global_parameter_change(self, parameter_name, old_value, new_value):
        """Notify all callbacks that a global parameter has changed"""
        for callback in self._global_parameter_change_callbacks:
            try:
                callback(parameter_name, old_value, new_value)
            except Exception as e:
                print(f"Error in global parameter change callback: {e}")
    
    # Global parameter properties with change notification
    @property
    def global_dist(self):
        return self._global_dist
    
    @global_dist.setter
    def global_dist(self, value):
        old_value = self._global_dist
        self._global_dist = float(value)
        self._notify_global_parameter_change('dist', old_value, self._global_dist)
        self._clear_flux_caches()
        # Update all molecule instances
        for molecule in self.values():
            molecule.distance = self._global_dist
    
    @property
    def global_star_rv(self):
        return self._global_star_rv
    
    @global_star_rv.setter
    def global_star_rv(self, value):
        old_value = self._global_star_rv
        self._global_star_rv = float(value)
        self._notify_global_parameter_change('star_rv', old_value, self._global_star_rv)
        self._clear_flux_caches()
        # Update all molecule instances
        for molecule in self.values():
            molecule.stellar_rv = self._global_star_rv
    
    @property
    def global_fwhm(self):
        return self._global_fwhm
    
    @global_fwhm.setter
    def global_fwhm(self, value):
        old_value = self._global_fwhm
        self._global_fwhm = float(value)
        self._notify_global_parameter_change('fwhm', old_value, self._global_fwhm)
        self._clear_flux_caches()
        # Update model parameters that depend on FWHM
        self._update_model_parameters()
        # Update all molecule instances
        for molecule in self.values():
            molecule.fwhm = self._global_fwhm
    
    @property
    def global_intrinsic_line_width(self):
        return self._global_intrinsic_line_width
    
    @global_intrinsic_line_width.setter
    def global_intrinsic_line_width(self, value):
        old_value = self._global_intrinsic_line_width
        self._global_intrinsic_line_width = float(value)
        self._notify_global_parameter_change('intrinsic_line_width', old_value, self._global_intrinsic_line_width)
        self._clear_flux_caches()
        # Update all molecule instances
        for molecule in self.values():
            molecule.intrinsic_line_width = self._global_intrinsic_line_width
    
    @property
    def global_wavelength_range(self):
        return self._global_wavelength_range
    
    @global_wavelength_range.setter
    def global_wavelength_range(self, value):
        old_value = self._global_wavelength_range
        self._global_wavelength_range = tuple(value)
        self._notify_global_parameter_change('wavelength_range', old_value, self._global_wavelength_range)
        self._clear_flux_caches()
        # Update model parameters that depend on wavelength range
        self._update_model_parameters()
        # Update all molecule instances
        for molecule in self.values():
            molecule.wavelength_range = self._global_wavelength_range
            molecule._recreate_spectrum()
    
    def _update_model_parameters(self):
        """Update model parameters when wavelength range or fwhm changes"""
        self._global_model_line_width = default_parms.SPEED_OF_LIGHT_KMS / self._global_fwhm
        self._global_model_pixel_res = (np.mean(self._global_wavelength_range) / default_parms.SPEED_OF_LIGHT_KMS * self._global_fwhm) / default_parms.PIXELS_PER_FWHM
        
        # Update all molecule instances to use new model parameters
        for molecule in self.values():
            molecule.model_line_width = self._global_model_line_width
            molecule.model_pixel_res = self._global_model_pixel_res
            # Recreate spectrum with new parameters
            molecule._recreate_spectrum()
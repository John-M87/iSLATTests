from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import os

from .Molecule import Molecule
import iSLAT.Constants as default_parms

class MoleculeDict(dict):
    """
    A dictionary to store Molecule objects with their names as keys.
    Provides efficient operations on collections of molecules with caching and lazy evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Flux storage and caching
        self.fluxes: Dict[str, np.ndarray] = {}
        self._summed_flux_cache: Dict[Tuple, np.ndarray] = {}
        self._cache_wave_data_hash: Optional[str] = None
        
        # Global parameters that affect all molecules
        self._global_dist: float = default_parms.DEFAULT_DISTANCE
        self._global_star_rv: float = default_parms.DEFAULT_STELLAR_RV
        self._global_fwhm: float = default_parms.DEFAULT_FWHM
        self._global_intrinsic_line_width: float = default_parms.INTRINSIC_LINE_WIDTH
        self._global_wavelength_range: Tuple[float, float] = default_parms.WAVELENGTH_RANGE
        self._global_model_line_width: float = default_parms.MODEL_LINE_WIDTH
        self._global_model_pixel_res: float = default_parms.MODEL_PIXEL_RESOLUTION
        
        # Callbacks to notify when global parameters change
        self._global_parameter_change_callbacks: List[Callable] = []

    def add_molecule(self, mol_entry: Dict[str, Any], intrinsic_line_width: Optional[float] = None, 
                     wavelength_range: Optional[Tuple[float, float]] = None, 
                     model_pixel_res: Optional[float] = None, model_line_width: Optional[float] = None, 
                     distance: Optional[float] = None, hitran_data: Optional[Any] = None) -> Molecule:
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
            color=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Color"),
            initial_molecule_parameters=getattr(self, 'initial_molecule_parameters', {}).get(mol_name, {}),
            wavelength_range=effective_wavelength_range,
            broad=effective_intrinsic_line_width,
            model_pixel_res=effective_model_pixel_res,
            model_line_width=effective_model_line_width,
            distance=effective_distance,
            fwhm=self._global_fwhm,
            stellar_rv=self._global_star_rv,
            radius=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Rad", None),
            temp=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Temp", None),
            n_mol=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("N_Mol", None),
            is_visible=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Vis", True),
            hitran_data=hitran_data
        )

        # Store the molecule in the dictionary
        self[mol_name] = molecule

        print(f"Molecule Initialized: {mol_name}")
        
        # Update fluxes if the molecule has plot data
        if hasattr(molecule, 'plot_flux'):
            self.fluxes[mol_name] = molecule.plot_flux
            
        return molecule

    def add_molecules(self, *molecules) -> None:
        """Add multiple molecules to the dictionary."""
        molecules = molecules[0]
        for mol in molecules:
            if isinstance(mol, Molecule):
                self[mol.name] = mol
            else:
                raise TypeError("Expected a Molecule instance.")

    def load_molecules_data(self, molecules_data: List[Dict[str, Any]], 
                           initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                           save_file_data: Dict[str, Dict[str, Any]], 
                           wavelength_range: Tuple[float, float], 
                           intrinsic_line_width: float, 
                           model_pixel_res: float, 
                           model_line_width: float, 
                           distance: float, 
                           hitran_data: Dict[str, Any]) -> None:
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
                hitran_data=hitran_data[mol_entry["name"]] if mol_entry["name"] in hitran_data else None
            )
    
    def clear(self):
        """Clear the dictionary of all molecules."""
        super().clear()
        self.fluxes.clear()
        print("MoleculeDict cleared.")

    def update_molecule_fluxes(self, wave_data: Optional[np.ndarray] = None) -> None:
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
    
    def get_summed_flux(self, wave_data: np.ndarray, visible_only: bool = True) -> np.ndarray:
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
    
    def get_molecule_flux(self, mol_name: str) -> Optional[np.ndarray]:
        """Get the flux for a specific molecule"""
        if mol_name in self.fluxes:
            return self.fluxes[mol_name]
        elif mol_name in self and hasattr(self[mol_name], 'plot_flux'):
            return self[mol_name].plot_flux
        return None

    def add_global_parameter_change_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when global parameters change"""
        self._global_parameter_change_callbacks.append(callback)
    
    def remove_global_parameter_change_callback(self, callback: Callable) -> None:
        """Remove a callback function"""
        if callback in self._global_parameter_change_callbacks:
            self._global_parameter_change_callbacks.remove(callback)
    
    def _notify_global_parameter_change(self, parameter_name: str, old_value: Any, new_value: Any) -> None:
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
    
    def get_ndarray_of_attributes(self, attribute_name: str) -> np.ndarray:
        """Get a numpy array of a specific attribute for all molecules."""
        return np.array([getattr(mol, attribute_name, None) for mol in self.values()])
    
    def get_ndarray_of_line_attributes(self, attribute_name: str) -> np.ndarray:
        """Get a numpy array of a specific line attribute for all molecules."""
        return np.array([mol.lines.get_ndarray_of_attribute(attribute_name) for mol in self.values() if hasattr(mol, 'lines')])

    # Enhanced bulk parameter update methods
    def bulk_update_parameter(self, parameter_name: str, value: Any, molecule_names: Optional[List[str]] = None) -> None:
        """
        Update a single parameter for multiple molecules efficiently.
        
        Args:
            parameter_name: Name of the parameter to update
            value: New value for the parameter
            molecule_names: List of molecule names to update (None for all)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        updated_molecules = []
        
        # Batch update with thread safety
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                old_value = getattr(molecule, parameter_name, None)
                
                # Update the parameter directly (bypassing setter to avoid individual cache invalidations)
                if hasattr(molecule, f'_{parameter_name}'):
                    setattr(molecule, f'_{parameter_name}', float(value))
                elif parameter_name == 'intrinsic_line_width':
                    molecule.broad = float(value)
                elif hasattr(molecule, parameter_name):
                    setattr(molecule, parameter_name, value)
                
                updated_molecules.append((molecule, parameter_name, old_value, value))
        
        # Batch invalidation and notification
        self._batch_invalidate_caches(updated_molecules, parameter_name)
        
        print(f"Bulk updated {parameter_name} to {value} for {len(updated_molecules)} molecules")
    
    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], molecule_names: Optional[List[str]] = None) -> None:
        """
        Update multiple parameters for multiple molecules efficiently.
        
        Args:
            parameter_dict: Dictionary of parameter names and values
            molecule_names: List of molecule names to update (None for all)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        updated_molecules = []
        
        # Batch update with efficient parameter setting
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                molecule.bulk_update_parameters(parameter_dict, skip_notification=True)
                updated_molecules.append(molecule)
        
        # Batch cache invalidation
        if updated_molecules:
            self._batch_invalidate_caches_multiple(updated_molecules, parameter_dict.keys())
        
        print(f"Bulk updated {len(parameter_dict)} parameters for {len(updated_molecules)} molecules")
    
    def bulk_set_temperature(self, temperature: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update temperature for multiple molecules."""
        self.bulk_update_parameter('temp', temperature, molecule_names)
    
    def bulk_set_radius(self, radius: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update radius for multiple molecules."""
        self.bulk_update_parameter('radius', radius, molecule_names)
    
    def bulk_set_column_density(self, n_mol: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update column density for multiple molecules."""
        self.bulk_update_parameter('n_mol', n_mol, molecule_names)
    
    def bulk_set_distance(self, distance: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update distance for multiple molecules."""
        self.bulk_update_parameter('distance', distance, molecule_names)
    
    def bulk_set_fwhm(self, fwhm: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update FWHM for multiple molecules."""
        self.bulk_update_parameter('fwhm', fwhm, molecule_names)
    
    def bulk_set_intrinsic_line_width(self, width: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update intrinsic line width for multiple molecules."""
        self.bulk_update_parameter('intrinsic_line_width', width, molecule_names)
    
    def bulk_set_visibility(self, is_visible: bool, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update visibility for multiple molecules."""
        self.bulk_update_parameter('is_visible', is_visible, molecule_names)
    
    def force_recalculate_all(self, molecule_names: Optional[List[str]] = None) -> None:
        """
        Force recalculation of intensity and spectrum for specified molecules.
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        recalculated_count = 0
        
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                # Force invalidation of all caches
                molecule._intensity_valid = False
                molecule._spectrum_valid = False
                molecule._clear_flux_caches()
                molecule._invalidate_parameter_hash()
                recalculated_count += 1
        
        # Clear global caches
        self._clear_flux_caches()
        
        print(f"Forced recalculation for {recalculated_count} molecules")
    
    def _batch_invalidate_caches(self, updated_molecules: List[Tuple], parameter_name: str) -> None:
        """
        Efficiently invalidate caches for batch-updated molecules.
        
        Args:
            updated_molecules: List of (molecule, param_name, old_value, new_value) tuples
            parameter_name: Name of the parameter that was updated
        """
        # Determine what needs to be invalidated based on parameter type
        invalidate_intensity = parameter_name in ['temp', 'n_mol', 'fwhm', 'intrinsic_line_width']
        invalidate_spectrum = True  # Most parameters affect spectrum
        
        for molecule, param_name, old_value, new_value in updated_molecules:
            if invalidate_intensity:
                molecule._intensity_valid = False
            
            if invalidate_spectrum:
                molecule._spectrum_valid = False
            
            molecule._clear_flux_caches()
            molecule._invalidate_parameter_hash()
            
            # Send notification for this molecule
            molecule._notify_my_parameter_change(param_name, old_value, new_value)
        
        # Clear global caches once
        self._clear_flux_caches()
    
    def _batch_invalidate_caches_multiple(self, updated_molecules: List, parameter_names: List[str]) -> None:
        """
        Efficiently invalidate caches for molecules with multiple parameter updates.
        
        Args:
            updated_molecules: List of updated molecule objects
            parameter_names: List of parameter names that were updated
        """
        # Determine what needs to be invalidated
        invalidate_intensity = any(param in ['temp', 'n_mol', 'fwhm', 'intrinsic_line_width'] 
                                 for param in parameter_names)
        
        for molecule in updated_molecules:
            if invalidate_intensity:
                molecule._intensity_valid = False
            
            molecule._spectrum_valid = False
            molecule._clear_flux_caches()
            molecule._invalidate_parameter_hash()
        
        # Clear global caches once
        self._clear_flux_caches()
    
    def get_parameter_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all parameters for all molecules.
        
        Returns:
            Dictionary with molecule names as keys and parameter dictionaries as values
        """
        summary = {}
        
        for mol_name, molecule in self.items():
            summary[mol_name] = {
                'temp': molecule.temp,
                'radius': molecule.radius,
                'n_mol': molecule.n_mol,
                'distance': molecule.distance,
                'fwhm': molecule.fwhm,
                'intrinsic_line_width': molecule.intrinsic_line_width,
                'is_visible': molecule.is_visible,
                'stellar_rv': molecule.star_rv,
                'wavelength_range': molecule.wavelength_range,
                'model_pixel_res': molecule.model_pixel_res,
                'model_line_width': molecule.model_line_width
            }
        
        return summary
    
    def apply_parameter_template(self, template: Dict[str, Any], molecule_names: Optional[List[str]] = None) -> None:
        """
        Apply a parameter template to multiple molecules.
        
        Args:
            template: Dictionary of parameter names and values to apply
            molecule_names: List of molecule names to apply template to (None for all)
        """
        self.bulk_update_parameters(template, molecule_names)
        print(f"Applied parameter template to {len(molecule_names or self.keys())} molecules")
    
    # Multiprocessing support for molecule loading
    @staticmethod
    def _create_molecule_worker(args):
        """
        Worker function for creating molecules in parallel.
        
        Args:
            args: Tuple of (mol_data, global_params, init_params)
        
        Returns:
            Tuple of (success, molecule_or_error, mol_name)
        """
        try:
            mol_data, global_params, init_params = args
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            
            if not mol_name:
                return False, "Missing molecule name", None
            
            # Import Molecule here to avoid pickling issues
            from .Molecule import Molecule
            
            # Create molecule with provided parameters
            molecule = Molecule(
                user_save_data=mol_data if "Molecule Name" in mol_data else None,
                hitran_data=mol_data.get("hitran_data") if "hitran_data" in mol_data else None,
                name=mol_name,
                filepath=mol_data.get("file") or mol_data.get("File Path"),
                displaylabel=mol_data.get("label") or mol_data.get("Molecule Label", mol_name),
                wavelength_range=global_params.get("wavelength_range"),
                distance=global_params.get("distance"),
                fwhm=global_params.get("fwhm"),
                stellar_rv=global_params.get("stellar_rv"),
                broad=global_params.get("intrinsic_line_width"),
                model_pixel_res=global_params.get("model_pixel_res"),
                model_line_width=global_params.get("model_line_width"),
                temp=mol_data.get("Temp"),
                radius=mol_data.get("Rad"),
                n_mol=mol_data.get("N_Mol"),
                color=mol_data.get("Color"),
                is_visible=mol_data.get("Vis", True),
                initial_molecule_parameters=init_params.get(mol_name, {})
            )
            
            return True, molecule, mol_name
            
        except Exception as e:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            return False, str(e), mol_name
    
    def load_molecules_parallel(self, molecules_data: List[Dict[str, Any]], 
                               initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                               max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Load multiple molecules in parallel using multiprocessing.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
            max_workers: Maximum number of worker processes (None for auto-detect)
        
        Returns:
            Dictionary with loading statistics and results
        """
        if not molecules_data:
            return {"success": 0, "failed": 0, "molecules": []}
        
        # Prepare global parameters
        global_params = {
            "wavelength_range": self._global_wavelength_range,
            "distance": self._global_dist,
            "fwhm": self._global_fwhm,
            "stellar_rv": self._global_star_rv,
            "intrinsic_line_width": self._global_intrinsic_line_width,
            "model_pixel_res": self._global_model_pixel_res,
            "model_line_width": self._global_model_line_width
        }
        
        # Prepare arguments for workers
        worker_args = [
            (mol_data, global_params, initial_molecule_parameters)
            for mol_data in molecules_data
        ]
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(len(molecules_data), mp.cpu_count())
        
        print(f"Loading {len(molecules_data)} molecules using {max_workers} worker processes...")
        
        results = {
            "success": 0,
            "failed": 0,
            "molecules": [],
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Use ProcessPoolExecutor for CPU-bound molecule creation
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_args = {
                    executor.submit(self._create_molecule_worker, args): args[0]
                    for args in worker_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_args):
                    mol_data = future_to_args[future]
                    try:
                        success, result, mol_name = future.result()
                        
                        if success:
                            # Add molecule to dictionary
                            self[mol_name] = result
                            results["molecules"].append(mol_name)
                            results["success"] += 1
                            print(f"✓ Successfully loaded molecule: {mol_name}")
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"{mol_name}: {result}")
                            print(f"✗ Failed to load molecule '{mol_name}': {result}")
                            
                    except Exception as e:
                        mol_name = mol_data.get("Molecule Name", "Unknown")
                        results["failed"] += 1
                        results["errors"].append(f"{mol_name}: {str(e)}")
                        print(f"✗ Error processing molecule '{mol_name}': {e}")
        
        except Exception as e:
            print(f"Error in parallel molecule loading: {e}")
            # Fall back to sequential loading
            return self._load_molecules_sequential(molecules_data, initial_molecule_parameters)
        
        elapsed_time = time.time() - start_time
        print(f"Parallel loading completed in {elapsed_time:.2f}s")
        print(f"Successfully loaded: {results['success']}, Failed: {results['failed']}")
        
        # Clear and rebuild caches after loading
        self._clear_flux_caches()
        
        return results
    
    def _load_molecules_sequential(self, molecules_data: List[Dict[str, Any]], 
                                  initial_molecule_parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback sequential molecule loading method.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
        
        Returns:
            Dictionary with loading statistics and results
        """
        print("Falling back to sequential molecule loading...")
        
        results = {
            "success": 0,
            "failed": 0,
            "molecules": [],
            "errors": []
        }
        
        for mol_data in molecules_data:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            if not mol_name:
                results["failed"] += 1
                results["errors"].append("Unknown: Missing molecule name")
                continue
            
            try:
                success, result, name = self._create_molecule_worker((
                    mol_data, 
                    {
                        "wavelength_range": self._global_wavelength_range,
                        "distance": self._global_dist,
                        "fwhm": self._global_fwhm,
                        "stellar_rv": self._global_star_rv,
                        "intrinsic_line_width": self._global_intrinsic_line_width,
                        "model_pixel_res": self._global_model_pixel_res,
                        "model_line_width": self._global_model_line_width
                    },
                    initial_molecule_parameters
                ))
                
                if success:
                    self[mol_name] = result
                    results["molecules"].append(mol_name)
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{mol_name}: {result}")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{mol_name}: {str(e)}")
        
        return results
    
    def bulk_recalculate_sequential(self, molecule_names: Optional[List[str]] = None) -> None:
        """
        Recalculate intensity and spectrum for multiple molecules sequentially (default method).
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        if not molecule_names:
            return
        
        print(f"Recalculating {len(molecule_names)} molecules sequentially...")
        start_time = time.time()
        success_count = 0
        
        for mol_name in molecule_names:
            try:
                if mol_name in self:
                    molecule = self[mol_name]
                    # Force invalidation and recalculation
                    molecule._intensity_valid = False
                    molecule._spectrum_valid = False
                    molecule._clear_flux_caches()
                    molecule._invalidate_parameter_hash()
                    
                    # Trigger recalculation by accessing properties
                    if hasattr(molecule, 'calculate_intensity'):
                        molecule.calculate_intensity()
                    
                    success_count += 1
                else:
                    print(f"Molecule '{mol_name}' not found")
            except Exception as e:
                print(f"Error recalculating '{mol_name}': {str(e)}")
        
        # Clear global caches
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Sequential recalculation completed in {elapsed_time:.2f}s")
        print(f"Successfully recalculated {success_count}/{len(molecule_names)} molecules")
    
    def bulk_recalculate_parallel(self, molecule_names: Optional[List[str]] = None, 
                                 max_workers: Optional[int] = None) -> None:
        """
        Recalculate intensity and spectrum for multiple molecules in parallel.
        This method is available but not used by default for better stability.
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
            max_workers: Maximum number of worker threads (None for auto-detect)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        if not molecule_names:
            return
        
        if max_workers is None:
            max_workers = min(len(molecule_names), mp.cpu_count())
        
        print(f"Recalculating {len(molecule_names)} molecules using {max_workers} worker threads...")
        
        def recalculate_molecule(mol_name):
            """Worker function for recalculating a single molecule"""
            try:
                if mol_name in self:
                    molecule = self[mol_name]
                    # Force invalidation and recalculation
                    molecule._intensity_valid = False
                    molecule._spectrum_valid = False
                    molecule._clear_flux_caches()
                    molecule._invalidate_parameter_hash()
                    
                    # Trigger recalculation by accessing properties
                    if hasattr(molecule, 'calculate_intensity'):
                        molecule.calculate_intensity()
                    
                    return True, mol_name
                else:
                    return False, f"Molecule '{mol_name}' not found"
            except Exception as e:
                return False, f"Error recalculating '{mol_name}': {str(e)}"
        
        start_time = time.time()
        success_count = 0
        
        # Use ThreadPoolExecutor for I/O-bound recalculation tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(recalculate_molecule, mol_name): mol_name
                for mol_name in molecule_names
            }
            
            for future in as_completed(future_to_name):
                mol_name = future_to_name[future]
                try:
                    success, result = future.result()
                    if success:
                        success_count += 1
                    else:
                        print(f"Failed to recalculate {mol_name}: {result}")
                except Exception as e:
                    print(f"Error recalculating {mol_name}: {e}")
        
        # Clear global caches
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Parallel recalculation completed in {elapsed_time:.2f}s")
        print(f"Successfully recalculated {success_count}/{len(molecule_names)} molecules")
    
    def load_molecules_ultra_fast(self, molecules_data: List[Dict[str, Any]], 
                                 initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                                 max_workers: Optional[int] = None, 
                                 force_multiprocessing: bool = False) -> Dict[str, Any]:
        """
        Load multiple molecules with sequential loading by default, multiprocessing only when forced.
        
        This method uses sequential loading by default for better compatibility and stability.
        Multiprocessing is only used when explicitly requested via force_multiprocessing=True.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
            max_workers: Maximum number of worker processes (None for auto-detect)
            force_multiprocessing: If True, forces multiprocessing even for small datasets
        
        Returns:
            Dictionary with loading statistics and results
        """
        if not molecules_data:
            return {"success": 0, "failed": 0, "errors": []}
        
        print(f"Starting molecule loading for {len(molecules_data)} molecules...")
        start_time = time.time()
        
        # Use multiprocessing only if explicitly forced AND conditions are met
        use_multiprocessing = force_multiprocessing and self._should_use_multiprocessing(molecules_data, max_workers)
        
        results = {
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        if use_multiprocessing:
            print(f"Using multiprocessing for {len(molecules_data)} molecules (forced)...")
            # Use the existing parallel loading method
            parallel_results = self.load_molecules_parallel(
                molecules_data, 
                initial_molecule_parameters, 
                max_workers
            )
            results.update(parallel_results)
        else:
            print(f"Using sequential loading for {len(molecules_data)} molecules...")
            # Use sequential loading (default and safer)
            for mol_data in molecules_data:
                success = self._load_single_molecule_ultra_fast(mol_data, initial_molecule_parameters)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to load {mol_data.get('Molecule Name', 'unknown')}")
        
        elapsed_time = time.time() - start_time
        print(f"Ultra-fast loading completed in {elapsed_time:.3f}s")
        print(f"Success: {results['success']}, Failed: {results['failed']}")
        
        return results
    
    def _should_use_multiprocessing(self, molecules_data: List[Dict[str, Any]], 
                                   max_workers: Optional[int] = None) -> bool:
        """
        Determine whether to use multiprocessing based on workload characteristics.
        
        With ultra-fast loading, multiprocessing overhead may outweigh benefits for small datasets.
        """
        num_molecules = len(molecules_data)
        
        # Don't use multiprocessing for very few molecules
        if num_molecules < 3:
            return False
            
        # Estimate total workload based on file sizes (if available)
        total_estimated_lines = 0
        large_files_count = 0
        
        for mol_data in molecules_data:
            # Try to estimate file size/complexity
            file_path = mol_data.get("hitran_data") or mol_data.get("File Path")
            if file_path and os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    # Rough estimate: 80 bytes per line on average for .par files
                    estimated_lines = file_size // 80
                    total_estimated_lines += estimated_lines
                    
                    # Consider files with >50k lines as "large"
                    if estimated_lines > 50000:
                        large_files_count += 1
                except:
                    # If we can't estimate, assume medium size
                    total_estimated_lines += 25000
        
        # Use multiprocessing if:
        # 1. We have multiple large files (>50k lines each), OR
        # 2. Total estimated lines > 100k AND more than 2 molecules
        if large_files_count >= 2:
            return True
        elif total_estimated_lines > 100000 and num_molecules > 2:
            return True
        else:
            return False
    
    def _load_single_molecule_ultra_fast(self, mol_data: Dict[str, Any], 
                                        initial_molecule_parameters: Dict[str, Dict[str, Any]]) -> bool:
        """
        Load a single molecule using ultra-fast optimized methods.
        """
        try:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            
            if not mol_name:
                print("Error: Missing molecule name")
                return False
            
            # Create molecule with optimized loading
            molecule = Molecule(
                user_save_data=mol_data if "Molecule Name" in mol_data else None,
                hitran_data=mol_data.get("hitran_data") if "hitran_data" in mol_data else None,
                name=mol_name,
                filepath=mol_data.get("file") or mol_data.get("File Path"),
                displaylabel=mol_data.get("label") or mol_data.get("Molecule Label", mol_name),
                wavelength_range=self._global_wavelength_range,
                distance=self._global_dist,
                fwhm=self._global_fwhm,
                stellar_rv=self._global_star_rv,
                broad=self._global_intrinsic_line_width,
                model_pixel_res=self._global_model_pixel_res,
                model_line_width=self._global_model_line_width,
                temp=mol_data.get("Temp"),
                radius=mol_data.get("Rad"),
                n_mol=mol_data.get("N_Mol"),
                color=mol_data.get("Color"),
                is_visible=mol_data.get("Vis", True),
                initial_molecule_parameters=initial_molecule_parameters.get(mol_name, {})
            )
            
            # Add to dictionary
            self[mol_name] = molecule
            print(f"Successfully loaded molecule: {mol_name}")
            return True
            
        except Exception as e:
            print(f"Error loading molecule '{mol_name}': {e}")
            return False

    # Multiprocessing support for molecule loading
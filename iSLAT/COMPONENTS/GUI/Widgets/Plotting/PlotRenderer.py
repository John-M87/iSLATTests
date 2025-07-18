from typing import Optional, List, Dict, Any, Tuple, Union, TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import iSLAT.Constants as c

# Import actual data types for proper type hinting
if TYPE_CHECKING:
    from iSLAT.COMPONENTS.DataTypes.Molecule import Molecule
    from iSLAT.COMPONENTS.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.COMPONENTS.DataTypes.MoleculeLineList import MoleculeLineList
    from iSLAT.COMPONENTS.DataTypes.MoleculeLine import MoleculeLine
    from iSLAT.COMPONENTS.DataTypes.ir_model.intensity import Intensity
    from iSLAT.COMPONENTS.DataTypes.ir_model.spectrum import Spectrum

class PlotRenderer:
    """
    Handles all plot rendering and visual updates for the iSLAT spectroscopy tool.
    
    This class provides comprehensive rendering of:
    - Observed spectrum data with error bars
    - Individual molecule model spectra
    - Summed model spectra
    - Population diagrams with caching
    - Line inspection plots
    - Saved line markers
    
    Features:
    - Efficient molecule visibility filtering
    - Cached flux calculations for performance
    - Population diagram caching to avoid redundant calculations
    - Memory-conscious plotting with line limit management
    - Batch operations for better performance with many molecules
    """
    
    def __init__(self, plot_manager: Any) -> None:
        self.plot_manager = plot_manager
        self.islat = plot_manager.islat
        self.theme: Dict[str, Any] = plot_manager.theme
        
        self.fig = plot_manager.fig
        self.ax1 = plot_manager.ax1
        self.ax2 = plot_manager.ax2
        self.ax3 = plot_manager.ax3
        self.canvas = plot_manager.canvas
        
        self.model_lines: List[Line2D] = []
        self.active_lines: List[Line2D] = []
        
        self._plot_cache = {
            'population_diagram': {'molecule_id': None, 'param_hash': None, 'wave_range': None},
            'spectrum_plots': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._register_molecule_change_callbacks()
    
    # Helper methods for common operations
    def _convert_visibility_to_bool(self, is_visible_raw: Any) -> bool:
        """Convert various visibility representations to boolean"""
        if isinstance(is_visible_raw, str):
            return is_visible_raw.lower() in ('true', '1', 'yes', 'on')
        return bool(is_visible_raw)
    
    def _get_molecule_display_name(self, molecule: 'Molecule') -> str:
        """Get display name for a molecule"""
        return getattr(molecule, 'displaylabel', getattr(molecule, 'name', 'unknown'))
    
    def _get_molecule_identifier(self, molecule: 'Molecule') -> Optional[str]:
        """Get unique identifier for a molecule"""
        return getattr(molecule, 'name', getattr(molecule, 'displaylabel', None)) if molecule else None
    
    def _get_theme_value(self, key: str, default: Any = None) -> Any:
        """Get theme value with fallback"""
        return self.theme.get(key, default)
    
    def _get_molecule_color(self, molecule: 'Molecule') -> str:
        """Get color for molecule from theme or molecule properties"""
        mol_name = self._get_molecule_display_name(molecule)
        
        # Check molecule's own color first
        if hasattr(molecule, 'color') and molecule.color:
            return molecule.color
            
        # Then check theme colors
        molecule_colors = self._get_theme_value('molecule_colors', {})
        if mol_name in molecule_colors:
            return molecule_colors[mol_name]
            
        # Default fallback
        return self._get_theme_value('default_molecule_color', 'blue')
    
    def _safe_get_molecule_attribute(self, molecule: 'Molecule', attr_name: str, default_value: Any = None) -> Any:
        """Safely get molecule attribute with error handling"""
        try:
            return getattr(molecule, attr_name, default_value)
        except Exception as e:
            print(f"Error accessing {attr_name} for molecule {self._get_molecule_display_name(molecule)}: {e}")
            return default_value
    
    def _is_cache_valid_for_molecule(self, molecule: 'Molecule', cache_type: str) -> bool:
        """Check if cache is valid for a specific molecule and cache type"""
        try:
            if molecule is None:
                return False
                
            if hasattr(molecule, 'is_cache_valid'):
                return molecule.is_cache_valid(cache_type)
            
            # Fallback: assume cache is valid if no validation method exists
            return True
        except Exception as e:
            print(f"Error validating {cache_type} cache: {e}")
            return False
        
    def _register_molecule_change_callbacks(self) -> None:
        try:
            from iSLAT.COMPONENTS.DataTypes.Molecule import Molecule
            if hasattr(Molecule, 'add_molecule_parameter_change_callback'):
                Molecule.add_molecule_parameter_change_callback(self._handle_molecule_parameter_change)
        except Exception as e:
            print(f"Could not register for molecule parameter changes: {e}")
    
    def _handle_molecule_parameter_change(self, molecule_name: str, parameter_name: str, old_value: Any, new_value: Any) -> None:
        """Callback when a molecule parameter changes - invalidate relevant caches"""
        try:
            if old_value == new_value:
                return
                
            # Invalidate spectrum cache for this molecule
            if molecule_name in self._plot_cache['spectrum_plots']:
                del self._plot_cache['spectrum_plots'][molecule_name]
                
            # Invalidate population diagram if it's for this molecule
            pop_cache = self._plot_cache['population_diagram']
            if pop_cache['molecule_id'] == molecule_name:
                pop_cache['molecule_id'] = None
                pop_cache['param_hash'] = None
                
        except Exception as e:
            print(f"Error handling molecule parameter change: {e}")
    
    def cleanup_callbacks(self) -> None:
        """Cleanup callbacks when renderer is destroyed"""
        try:
            from iSLAT.COMPONENTS.DataTypes.Molecule import Molecule
            if hasattr(Molecule, 'remove_molecule_parameter_change_callback'):
                Molecule.remove_molecule_parameter_change_callback(self._handle_molecule_parameter_change)
        except Exception:
            pass
    
    def clear_all_plots(self) -> None:
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.model_lines.clear()
        self.active_lines.clear()
        self._plot_cache = {
            'population_diagram': {'molecule_id': None, 'param_hash': None, 'wave_range': None},
            'spectrum_plots': {},
            'cache_hits': self._plot_cache.get('cache_hits', 0),
            'cache_misses': self._plot_cache.get('cache_misses', 0)
        }
        
    def clear_model_lines(self) -> None:
        """Clear only the model spectrum lines from the main plot"""
        for line in self.model_lines:
            if line in self.ax1.lines:
                line.remove()
        self.model_lines.clear()
        
    def render_main_spectrum_plot(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                                 molecules: Union[List['Molecule'], 'MoleculeDict'], 
                                 summed_flux: Optional[np.ndarray] = None, 
                                 error_data: Optional[np.ndarray] = None) -> None:
        """Render the main spectrum plot with observed data, model spectra, and sum"""
        # Store current view limits
        current_xlim = self.ax1.get_xlim() if hasattr(self.ax1, 'get_xlim') else None
        current_ylim = self.ax1.get_ylim() if hasattr(self.ax1, 'get_ylim') else None
        
        # Clear the plot
        self.ax1.clear()
        
        # Early return if no data
        if wave_data is None or len(wave_data) == 0:
            self.ax1.set_title("No spectrum data loaded")
            return
        
        # Plot observed spectrum
        self._plot_observed_spectrum(wave_data, flux_data, error_data)
        
        # Plot individual molecule spectra
        if molecules:
            self.render_visible_molecules(wave_data, molecules)
            
        # Plot summed spectrum
        if summed_flux is not None and len(summed_flux) > 0:
            self._plot_summed_spectrum(wave_data, summed_flux)
        
        # Configure plot appearance
        self._configure_main_plot_appearance()
        
        # Restore view limits if they existed
        if current_xlim and current_ylim:
            if current_xlim != (0.0, 1.0) and current_ylim != (0.0, 1.0):
                self.ax1.set_xlim(current_xlim)
                self.ax1.set_ylim(current_ylim)
        
    def _plot_observed_spectrum(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                               error_data: Optional[np.ndarray] = None) -> None:
        """Plot the observed spectrum data"""
        if flux_data is not None and len(flux_data) > 0:
            if error_data is not None and len(error_data) == len(flux_data):
                # Plot with error bars
                self.ax1.errorbar(
                    wave_data, 
                    flux_data,
                    yerr=error_data,
                    fmt='-', 
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=self._get_theme_value("zorder_observed", 3),
                    elinewidth=0.5,
                    capsize=0
                )
            else:
                # Plot without error bars
                self.ax1.plot(
                    wave_data, 
                    flux_data,
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=self._get_theme_value("zorder_observed", 3)
                )
    
    def _plot_summed_spectrum(self, wave_data: np.ndarray, summed_flux: np.ndarray) -> None:
        """Plot the summed model spectrum"""
        if len(summed_flux) > 0 and np.any(summed_flux > 0):
            self.ax1.fill_between(
                wave_data,
                0,
                summed_flux,
                color=self._get_theme_value("summed_spectra_color", "lightgray"),
                alpha=1.0,
                label='Sum',
                zorder=self._get_theme_value("zorder_summed", 1)
            )
    
    def _configure_main_plot_appearance(self) -> None:
        """Configure the appearance of the main plot"""
        self.ax1.set_xlabel('Wavelength (μm)')
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax1.set_title("Full Spectrum with Line Inspection")
        
        # Only show legend if there are labeled items
        handles, labels = self.ax1.get_legend_handles_labels()
        if handles:
            self.ax1.legend()
        
    def render_line_inspection_plot(self, line_wave: Optional[np.ndarray], 
                                   line_flux: Optional[np.ndarray], 
                                   line_label: Optional[str] = None) -> None:
        """Render the line inspection subplot"""
        self.ax2.clear()
        
        if line_wave is not None and line_flux is not None:
            # Plot data in selected range
            self.ax2.plot(line_wave, line_flux, 
                         color=self._get_theme_value("foreground", "black"), 
                         linewidth=1, 
                         label="Observed")
            
            self.ax2.set_xlabel("Wavelength (μm)")
            self.ax2.set_ylabel("Flux (Jy)")
            self.ax2.set_title("Line inspection plot")
            
            # Show legend if there are labeled items
            handles, labels = self.ax2.get_legend_handles_labels()
            if handles:
                self.ax2.legend()
    
    def _get_molecule_parameters_hash(self, molecule: 'Molecule') -> Optional[int]:
        if molecule is None:
            return None
        
        try:
            if hasattr(molecule, 'get_parameter_hash'):
                return molecule.get_parameter_hash('full')
            elif hasattr(molecule, '_compute_full_parameter_hash'):
                return molecule._compute_full_parameter_hash()
            else:
                return hash((molecule.name, molecule.temp, molecule.radius, molecule.n_mol, 
                           molecule.distance, molecule.fwhm, molecule.broad))
        except Exception:
            return None
    
    def invalidate_population_diagram_cache(self) -> None:
        self._plot_cache['population_diagram'] = {'molecule_id': None, 'param_hash': None, 'wave_range': None}
    
    def render_population_diagram(self, molecule: 'Molecule', wave_range: Optional[Tuple[float, float]] = None) -> None:
        molecule_id = self._get_molecule_identifier(molecule)
        molecule_param_hash = self._get_molecule_parameters_hash(molecule)
        
        cache = self._plot_cache['population_diagram']
        cache_hit = (cache['molecule_id'] == molecule_id and 
                    cache['param_hash'] == molecule_param_hash and 
                    cache['wave_range'] == wave_range and
                    molecule_id is not None and
                    molecule_param_hash is not None)
        
        if cache_hit:
            self._plot_cache['cache_hits'] += 1
            return
        
        self._plot_cache['cache_misses'] += 1
        
        cache['molecule_id'] = molecule_id
        cache['param_hash'] = molecule_param_hash
        cache['wave_range'] = wave_range
        
        self.ax3.clear()
        
        if molecule is None:
            self.ax3.set_title("No molecule selected")
            return
            
        try:
            int_pars = self.get_intensity_data(molecule)
            if int_pars is None:
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f"{mol_label} - No intensity data")
                return

            wavelength = int_pars['lam']
            intens_mod = int_pars['intens']
            Astein_mod = int_pars['a_stein']
            gu = int_pars['g_up']
            eu = int_pars['e_up']

            radius = getattr(molecule, 'radius', 1.0)
            distance = getattr(molecule, 'distance', getattr(self.islat, 'global_dist', 140.0))
            
            area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
            dist = distance * c.PARSEC_CM
            beam_s = area / dist ** 2
            F = intens_mod * beam_s
            frequency = c.SPEED_OF_LIGHT_MICRONS / wavelength
            rd_yax = np.log(4 * np.pi * F / (Astein_mod * c.PLANCK_CONSTANT * frequency * gu))
            threshold = np.nanmax(F) / 100

            # Set limits with bounds checking
            valid_rd = rd_yax[F > threshold]
            valid_eu = eu[F > threshold]
            
            if len(valid_rd) > 0 and len(valid_eu) > 0:
                self.ax3.set_ylim(np.nanmin(valid_rd), np.nanmax(rd_yax) + 0.5)
                self.ax3.set_xlim(np.nanmin(eu) - 50, np.nanmax(valid_eu))

                # Populating the population diagram graph with the lines
                self.ax3.scatter(eu, rd_yax, s=0.5, color=self._get_theme_value("scatter_main_color", '#838B8B'))

                # Set labels
                self.ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
                self.ax3.set_xlabel(r'$E_{u}$ (K)')
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f'{mol_label} Population diagram', fontsize='medium')
            else:
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f"{mol_label} - No valid data for population diagram")
            
        except Exception as e:
            print(f"Error rendering population diagram: {e}")
            mol_label = self._get_molecule_display_name(molecule)
            self.ax3.set_title(f"{mol_label} - Error in calculation")
    
    def plot_saved_lines(self, saved_lines: pd.DataFrame) -> None:
        """Plot saved lines on the main spectrum"""
        if saved_lines.empty:
            return

        for index, line in saved_lines.iterrows():
            #print("Line:", line)
            #print("Index:", index)
            # Plot vertical lines at saved positions
            if 'lam' in line:
                self.ax1.axvline(
                    line['lam'], 
                    color=self._get_theme_value("saved_line_color", self._get_theme_value("saved_line_color_one", "red")),
                    alpha=0.7, 
                    linestyle=':', 
                    label=f"Saved: {line.get('label', 'Line')}"
                )
            elif 'xmin' in line and 'xmax' in line:
                # Plot wavelength range
                self.ax1.axvspan(
                    line['xmin'], 
                    line['xmax'], 
                    alpha=0.2, 
                    color=self._get_theme_value("saved_line_color_two", "coral"),
                    label=f"Saved Range: {line.get('label', 'Range')}"
                )
        # make sure that a refresh of the plot is triggered
        self.update_plot_display()
    
    def highlight_line_selection(self, xmin: float, xmax: float) -> None:
        """Highlight a selected wavelength range"""
        # Remove previous highlights
        for patch in self.ax1.patches:
            if hasattr(patch, '_islat_highlight'):
                patch.remove()
        
        # Add new highlight
        highlight = self.ax1.axvspan(xmin, xmax, alpha=0.3, color=self._get_theme_value("highlighted_line_color", "yellow"))
        highlight._islat_highlight = True
    
    def update_plot_display(self) -> None:
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_plot_refresh(self) -> None:
        """Force a complete plot refresh"""
        self.canvas.draw()
    
    def plot_vertical_lines(self, wavelengths: List[float], heights: Optional[List[float]] = None, 
                           colors: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> None:
        """Plot vertical lines at specified wavelengths"""
        if heights is None:
            # Get current y-limits for line height
            ylim = self.ax2.get_ylim()
            height = ylim[1] - ylim[0]
            heights = [height] * len(wavelengths)
        
        if colors is None:
            colors = ['green'] * len(wavelengths)
        
        if labels is None:
            labels = [None] * len(wavelengths)
        
        for i, (wave, height, color, label) in enumerate(zip(wavelengths, heights, colors, labels)):
            # Plot vertical line from bottom to specified height
            self.ax2.axvline(wave, color=color, alpha=0.7, linewidth=1, 
                           linestyle='-', picker=True, label=label)
            
            # Add scatter point at top of line for picking
            self.ax2.scatter([wave], [height], color=color, s=20, 
                           alpha=0.8, picker=True, zorder=5)
    
    def get_visible_molecules(self, molecules: Union['MoleculeDict', List['Molecule']]) -> List['Molecule']:
        """Get visible molecules using the most efficient method available"""
        
        print(f"PlotRenderer: molecules type = {type(molecules)}")
        print(f"PlotRenderer: hasattr(molecules, 'get_visible_molecules_fast') = {hasattr(molecules, 'get_visible_molecules_fast')}")
        print(f"PlotRenderer: hasattr(molecules, 'values') = {hasattr(molecules, 'values')}")
        print(f"PlotRenderer: hasattr(molecules, '__iter__') = {hasattr(molecules, '__iter__')}")
        
        if hasattr(molecules, 'get_visible_molecules_fast'):
            # MoleculeDict with fast access
            visible_names = molecules.get_visible_molecules_fast()
            visible_molecules = [molecules[name] for name in visible_names if name in molecules]
            print(f"PlotRenderer: get_visible_molecules_fast(): {len(visible_molecules)}/{len(molecules)} molecules visible: {visible_names}")
            return visible_molecules
        elif hasattr(molecules, 'values'):
            # Regular dict-like object - check each molecule's is_visible attribute
            print("PlotRenderer: Using values() path for dict-like object")
            visible_molecules = []
            for mol in molecules.values():
                is_visible_raw = getattr(mol, 'is_visible', False)
                mol_name = getattr(mol, 'name', 'unknown')
                
                # Use consolidated conversion method
                is_visible = self._convert_visibility_to_bool(is_visible_raw)
                    
                print(f"PlotRenderer: Checking {mol_name}: is_visible_raw = {is_visible_raw} -> is_visible = {is_visible}")
                if is_visible:
                    visible_molecules.append(mol)
            print(f"PlotRenderer: values() path result: {len(visible_molecules)}/{len(molecules)} molecules visible")
            return visible_molecules
        elif hasattr(molecules, '__iter__'):
            # List-like object
            print("PlotRenderer: Using __iter__ path for list-like object")
            visible_molecules = []
            total_count = 0
            for mol in molecules:
                total_count += 1
                is_visible_raw = getattr(mol, 'is_visible', False)
                mol_name = getattr(mol, 'name', 'unknown')
                mol_id = id(mol)
                
                # Use consolidated conversion method
                is_visible = self._convert_visibility_to_bool(is_visible_raw)
                
                print(f"PlotRenderer: Checking {mol_name} (id:{mol_id}): is_visible_raw = {is_visible_raw} (type: {type(is_visible_raw)}) -> is_visible = {is_visible}")
                
                if is_visible:
                    visible_molecules.append(mol)
                    print(f"PlotRenderer: ADDED {mol_name} to visible_molecules list (list length now: {len(visible_molecules)})")
                else:
                    print(f"PlotRenderer: SKIPPED {mol_name} (not visible)")
                
            print(f"PlotRenderer: Final visible_molecules list length: {len(visible_molecules)}")
            print(f"PlotRenderer: __iter__ path result: {len(visible_molecules)}/{total_count} molecules visible")
            return visible_molecules
        else:
            # Single molecule
            is_visible_raw = getattr(molecules, 'is_visible', False)
            
            # Use consolidated conversion method
            is_visible = self._convert_visibility_to_bool(is_visible_raw)
                
            result = [molecules] if is_visible else []
            print(f"PlotRenderer: Single molecule visibility: is_visible_raw = {is_visible_raw} -> is_visible = {is_visible} -> {'visible' if is_visible else 'hidden'}")
            return result
    
    def render_visible_molecules(self, wave_data: np.ndarray, molecules: Union['MoleculeDict', List['Molecule']]) -> None:
        """Render molecules using available methods"""
        if not molecules:
            return
        
        print(f"PlotRenderer: render_visible_molecules called with {len(molecules) if hasattr(molecules, '__len__') else 'unknown count'} molecules")
        
        # Get visible molecules
        visible_molecules = self.get_visible_molecules(molecules)
        if not visible_molecules:
            print("PlotRenderer: No visible molecules, returning early")
            return
        
        print(f"PlotRenderer: About to render {len(visible_molecules)} visible molecules: {[getattr(m, 'name', 'unknown') for m in visible_molecules]}")
        
        # Use rendering for each molecule
        for mol in visible_molecules:
            mol_name = getattr(mol, 'name', 'unknown')
            print(f"PlotRenderer: Rendering molecule: {mol_name}")
            self.render_individual_molecule_spectrum(mol, wave_data)
    
    def optimize_plot_memory_usage(self) -> None:
        """Optimize memory usage for plotting operations"""
        # Limit the number of cached model lines
        if len(self.model_lines) > 50:
            # Remove oldest lines from plot
            for line in self.model_lines[:25]:
                if line in self.ax1.lines:
                    line.remove()
            self.model_lines = self.model_lines[25:]
        
        # Clear inactive lines
        self.active_lines = [line for line in self.active_lines if line in self.ax2.lines]
    
    def batch_update_molecule_colors(self, molecule_color_map: Dict[str, str]) -> None:
        """Update molecule colors in batch for better performance"""
        for line in self.model_lines:
            label = line.get_label()
            if label in molecule_color_map:
                line.set_color(molecule_color_map[label])
        
        # Update canvas once at the end
        self.canvas.draw_idle()
    
    def get_plot_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics for debugging"""
        return {
            'model_lines_count': len(self.model_lines),
            'active_lines_count': len(self.active_lines),
            'cache_hits': self._plot_cache.get('cache_hits', 0),
            'cache_misses': self._plot_cache.get('cache_misses', 0),
            'cache_hit_ratio': (
                self._plot_cache.get('cache_hits', 0) / 
                max(1, self._plot_cache.get('cache_hits', 0) + self._plot_cache.get('cache_misses', 0))
            ),
            'cached_spectrum_plots': len(self._plot_cache.get('spectrum_plots', {})),
            'memory_optimized': hasattr(self, '_memory_optimized'),
            'line_intensity_threshold_percent': self.get_line_intensity_threshold() * 100
        }
    
    def clear_active_lines(self, active_lines_list: List[Any]) -> None:
        """
        Properly clear active lines by removing matplotlib artists first.
        
        Parameters
        ----------
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
        """
        for line_data in active_lines_list:
            if len(line_data) >= 2:
                line_artist = line_data[0]  # Line artist (vlines)
                scatter_artist = line_data[1]  # Scatter artist
                
                # Remove line artist if it exists
                if line_artist is not None:
                    try:
                        line_artist.remove()
                    except (ValueError, AttributeError):
                        pass
                
                # Remove scatter artist if it exists
                if scatter_artist is not None:
                    try:
                        scatter_artist.remove()
                    except (ValueError, AttributeError):
                        pass
        
        # Clear the list after removing all artists
        active_lines_list.clear()
    
    def render_active_lines_in_population_diagram(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], active_lines_list: List[Any]) -> None:
        """
        Render active lines as scatter points in the population diagram.
        
        Lines are filtered based on intensity threshold from user settings.
        Only lines with intensity above threshold_percent of the strongest line are rendered.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
        active_lines_list : List[Any]
            List to store active line data for interaction
        """
        if not line_data:
            return
        
        # Get threshold and filter lines
        threshold_percent = self.get_line_intensity_threshold()
        filtered_line_data = self.filter_lines_by_threshold(line_data, threshold_percent)
        
        if not filtered_line_data:
            print(f"No lines above threshold ({threshold_percent*100:.1f}% of strongest line) for population diagram")
            return
            
        print(f"Rendering {len(filtered_line_data)}/{len(line_data)} lines in population diagram above threshold ({threshold_percent*100:.1f}%)")
        
        # Get max intensity from original data for consistent percentage calculation
        original_intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(original_intensities) if original_intensities else 1.0
        
        # Calculate rd_yax values for each filtered line and add scatter points
        for idx, (line, intensity, tau_val) in enumerate(filtered_line_data):
            if all(x is not None for x in [intensity, line.a_stein, line.g_up, line.lam]):
                # Get molecule properties safely
                molecule = getattr(self.islat, 'active_molecule', None)
                if molecule is None:
                    continue
                    
                radius = getattr(molecule, 'radius', 1.0)
                distance = getattr(molecule, 'distance', getattr(self.islat, 'global_dist', 140.0))
                
                # Calculate rd_yax
                area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
                dist = distance * c.PARSEC_CM
                beam_s = area / dist ** 2
                F = intensity * beam_s
                freq = c.SPEED_OF_LIGHT_MICRONS / line.lam
                rd_yax = np.log(4 * np.pi * F / (line.a_stein * c.PLANCK_CONSTANT * freq * line.g_up))
                
                # Create scatter point
                sc = self.ax3.scatter(line.e_up, rd_yax, s=30, 
                                     color=self._get_theme_value("scatter_main_color", 'green'), 
                                     edgecolors='black', picker=True)
                
                # Store line information
                value_data = {
                    'lam': line.lam,
                    'e': line.e_up,
                    'a': line.a_stein,
                    'g': line.g_up,
                    'rd_yax': rd_yax,
                    'inten': intensity,
                    'up_lev': line.lev_up if line.lev_up else 'N/A',
                    'low_lev': line.lev_low if line.lev_low else 'N/A',
                    'tau': tau_val if tau_val is not None else 'N/A',
                    'intensity_percent': (intensity / max_intensity) * 100  # Store percentage for debugging
                }
                
                # Update existing entry or create new one
                if idx < len(active_lines_list):
                    # Update existing entry with scatter artist
                    active_lines_list[idx][1] = sc  # Set scatter artist
                    active_lines_list[idx][2].update(value_data)  # Update value data
                else:
                    # Create new entry: [line_artist, scatter_artist, value_data]
                    active_lines_list.append([None, sc, value_data])
    
    def render_active_lines_in_line_inspection(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], active_lines_list: List[Any], 
                                              max_y: float) -> None:
        """
        Render active lines as vertical lines in the line inspection plot.
        
        Lines are filtered based on intensity threshold from user settings.
        Only lines with intensity above threshold_percent of the strongest line are rendered.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
        active_lines_list : List[Any]
            List to store active line data for interaction
        max_y : float
            Maximum y value for scaling line heights
        """
        if not line_data:
            return
        
        # Get threshold and filter lines
        threshold_percent = self.get_line_intensity_threshold()
        filtered_line_data = self.filter_lines_by_threshold(line_data, threshold_percent)
        
        if not filtered_line_data:
            print(f"No lines above threshold ({threshold_percent*100:.1f}% of strongest line)")
            return
            
        print(f"Rendering {len(filtered_line_data)}/{len(line_data)} lines above threshold ({threshold_percent*100:.1f}%)")
        
        # Get max intensity from original data for consistent scaling
        original_intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(original_intensities) if original_intensities else 1.0
        
        # Plot vertical lines for each filtered molecular line and create/update active_lines entries
        for idx, (line, intensity, tau_val) in enumerate(filtered_line_data):
            # Calculate line height
            lineheight = 0
            if max_intensity > 0:
                lineheight = (intensity / max_intensity) * max_y
            
            if lineheight > 0:
                # Create vertical line
                vline = self.ax2.vlines(line.lam, 0, lineheight,
                                       color=self._get_theme_value("active_scatter_line_color", "green"), 
                                       linestyle='dashed', linewidth=1, picker=True)
                
                # Add text label
                text = self.ax2.text(line.lam, lineheight,
                                   f"{line.e_up:.0f},{line.a_stein:.3f}", 
                                   fontsize='x-small', 
                                   color=self._get_theme_value("active_scatter_line_color", "green"), 
                                   rotation=45)
                
                # Create value data for this line
                value_data = {
                    'lam': line.lam,
                    'e': line.e_up,
                    'a': line.a_stein,
                    'g': line.g_up,
                    'inten': intensity,
                    'up_lev': line.lev_up if line.lev_up else 'N/A',
                    'low_lev': line.lev_low if line.lev_low else 'N/A',
                    'tau': tau_val if tau_val is not None else 'N/A',
                    'text_obj': text,
                    'lineheight': lineheight,
                    'intensity_percent': (intensity / max_intensity) * 100  # Store percentage for debugging
                }
                
                # Add new entry to active_lines or update existing one
                if idx < len(active_lines_list):
                    # Update existing entry
                    active_lines_list[idx][0] = vline  # Set line artist
                    active_lines_list[idx][2].update(value_data)  # Update value data
                else:
                    # Create new entry: [line_artist, scatter_artist, value_data]
                    active_lines_list.append([vline, None, value_data])
    
    def highlight_strongest_line(self, active_lines_list: List[Any]) -> Any:
        """
        Find and highlight the strongest line in the active lines.
        
        Parameters
        ----------
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
            
        Returns
        -------
        Any
            The strongest line triplet or None
        """
        if not active_lines_list:
            return None
            
        # Reset all lines to green first
        for line, scatter, value in active_lines_list:
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
                scatter.set_zorder(1)  # Reset z-order
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('green')
        
        # Find the line with the highest intensity
        highest_intensity = -float('inf')
        strongest_triplet = None
        
        for line, scatter, value in active_lines_list:
            intensity = value.get('inten', 0) if value else 0
            if intensity > highest_intensity:
                highest_intensity = intensity
                strongest_triplet = [line, scatter, value]
        
        # Highlight the strongest line in orange
        if strongest_triplet is not None:
            line, scatter, value = strongest_triplet
            if line is not None:
                line.set_color('orange')
            if scatter is not None:
                scatter.set_facecolor('orange')
                scatter.set_zorder(10)  # Bring to front
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('orange')
        
        return strongest_triplet
    
    def handle_line_pick_event(self, picked_artist: Any, active_lines_list: List[Any]) -> Any:
        """
        Handle line pick events and highlight the selected line.
        
        Parameters
        ----------
        picked_artist : Any
            The matplotlib artist that was picked
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
            
        Returns
        -------
        Any
            The value data of the picked line or None
        """
        picked_value = None
        
        # Find which entry in active_lines was picked and reset colors
        for line, scatter, value in active_lines_list:
            is_picked = (picked_artist is line or picked_artist is scatter)
            
            # Reset all to green first
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('green')
            
            # If this was the picked item, highlight in orange
            if is_picked:
                picked_value = value
                if line is not None:
                    line.set_color('orange')
                if scatter is not None:
                    scatter.set_facecolor('orange')
                if 'text_obj' in value and value['text_obj'] is not None:
                    value['text_obj'].set_color('orange')
        
        return picked_value
    
    def get_molecule_spectrum_data(self, molecule: 'Molecule', wave_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get molecule spectrum data efficiently using the molecule's internal caching system.
        
        This method leverages the molecule's optimized caching to avoid redundant calculations.
        """
        try:
            molecule_name = self._get_molecule_display_name(molecule)
            
            # Method 1: Use molecule's prepare_plot_data method (preferred)
            if hasattr(molecule, 'prepare_plot_data') and callable(molecule.prepare_plot_data):
                try:
                    result = molecule.prepare_plot_data(wave_data)
                    if result is not None and len(result) == 2:
                        return result[0], result[1]  # (lam, flux)
                    elif hasattr(molecule, 'plot_lam') and hasattr(molecule, 'plot_flux'):
                        if molecule.plot_lam is not None and molecule.plot_flux is not None:
                            return molecule.plot_lam, molecule.plot_flux
                except Exception as e:
                    print(f"Error in prepare_plot_data for {molecule_name}: {e}")
            
            # Method 2: Use molecule's get_flux method (uses interpolated flux cache)
            if hasattr(molecule, 'get_flux') and callable(molecule.get_flux):
                try:
                    flux = molecule.get_flux(wave_data)
                    if flux is not None and len(flux) > 0:
                        return wave_data.copy(), flux
                except Exception as e:
                    print(f"Error in get_flux for {molecule_name}: {e}")
            
            # Method 3: Use spectrum object directly if molecule's internal methods fail
            if hasattr(molecule, 'spectrum') and molecule.spectrum is not None:
                spectrum = molecule.spectrum
                if hasattr(spectrum, 'lamgrid') and hasattr(spectrum, 'flux_jy'):
                    lam = spectrum.lamgrid
                    flux = spectrum.flux_jy  # Use Jy units for consistency
                    if lam is not None and flux is not None and len(lam) > 0 and len(flux) > 0:
                        # Interpolate to requested wavelength grid
                        interpolated_flux = np.interp(wave_data, lam, flux, left=0, right=0)
                        return wave_data.copy(), interpolated_flux
            
            print(f"No valid spectrum data found for molecule {molecule_name}")
            return None, None
            
        except Exception as e:
            print(f"Error getting molecule spectrum: {e}")
            return None, None
    
    def get_molecule_line_data(self, molecule: 'Molecule', xmin: float, xmax: float) -> List[Tuple['MoleculeLine', float, Optional[float]]]:
        """
        Get molecule lines in wavelength range.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object
        xmin, xmax : float
            Wavelength range
            
        Returns
        -------
        List[Tuple['MoleculeLine', float, Optional[float]]]
            List of (MoleculeLine, intensity, tau) tuples
        """
        try:
            # Method 1: Use intensity API
            if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                intensity_obj = molecule.intensity
                if hasattr(intensity_obj, 'get_lines_in_range_with_intensity'):
                    return intensity_obj.get_lines_in_range_with_intensity(xmin, xmax)
            
            # Method 2: Use MoleculeLineList directly
            if hasattr(molecule, 'lines') and molecule.lines is not None:
                lines = molecule.lines
                if hasattr(lines, 'get_lines_in_range'):
                    lines_in_range = lines.get_lines_in_range(xmin, xmax)
                    # Try to get corresponding intensities
                    if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                        intensity_obj = molecule.intensity
                        if hasattr(intensity_obj, 'intensity') and intensity_obj.intensity is not None:
                            intensities = intensity_obj.intensity
                            tau_values = getattr(intensity_obj, 'tau', None)
                            
                            result = []
                            for i, line in enumerate(lines_in_range):
                                intensity = intensities[i] if i < len(intensities) else 0.0
                                tau = tau_values[i] if tau_values is not None and i < len(tau_values) else None
                                result.append((line, intensity, tau))
                            return result
                    else:
                        # Return lines with zero intensity
                        return [(line, 0.0, None) for line in lines_in_range]
            
            return []
            
        except Exception as e:
            print(f"Error getting molecule lines: {e}")
            return []
    
    def render_individual_molecule_spectrum(self, molecule: 'Molecule', wave_data: np.ndarray, 
                                         plot_name: Optional[str] = None) -> bool:
        """
        Render a single molecule spectrum leveraging intelligent caching.
        
        This method uses the molecule's internal caching system and validates
        cache coherence before attempting to plot.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object with internal caching
        wave_data : np.ndarray
            Wavelength array
        plot_name : Optional[str]
            Custom name for plotting
            
        Returns
        -------
        bool
            True if successfully plotted, False otherwise
        """
        try:
            # Validate and sync cache state with molecule
            self.sync_molecule_cache(molecule)
            
            print("Hey whats up man")
            # Get spectrum data using enhanced caching
            plot_lam, plot_flux = self.get_molecule_spectrum_data(molecule, wave_data)
            
            if plot_lam is None or plot_flux is None:
                return False
            
            # Check if we actually have meaningful flux data
            if len(plot_flux) == 0 or np.all(plot_flux == 0):
                return False
            
            # Get molecule properties
            mol_name = plot_name or self._get_molecule_display_name(molecule)
            color = self._get_molecule_color(molecule)
            label = getattr(molecule, 'displaylabel', mol_name)
            
            # Plot the spectrum
            line, = self.ax1.plot(
                plot_lam,
                plot_flux,
                linestyle='--',
                color=color,
                alpha=0.7,
                linewidth=self._get_theme_value("model_plot_line_width", 1.5),
                label=label,
                zorder=self._get_theme_value("zorder_model", 2)
            )
            
            self.model_lines.append(line)
            return True
            
        except Exception as e:
            print(f"Error plotting molecule {self._get_molecule_display_name(molecule)}: {e}")
            return False
    
    def get_intensity_data(self, molecule: 'Molecule') -> Optional[pd.DataFrame]:
        """
        Get intensity table leveraging molecule's internal caching system.
        
        This method uses the molecule's internal intensity caching and validation
        to avoid redundant intensity calculations.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object with internal caching
            
        Returns
        -------
        Optional[pd.DataFrame]
            Intensity table with columns: lam, intens, a_stein, g_up, e_up, etc.
        """
        try:
            if not hasattr(molecule, 'intensity') or molecule.intensity is None:
                return None
                
            intensity_obj = molecule.intensity
            
            # Use molecule's intelligent intensity calculation (leverages internal caching)
            if hasattr(molecule, 'calculate_intensity'):
                molecule.calculate_intensity()  # This uses molecule's internal caching
            elif hasattr(molecule, '_ensure_intensity_initialized'):
                molecule._ensure_intensity_initialized()
            
            # Method 1: Use get_table property/method
            if hasattr(intensity_obj, 'get_table'):
                table = intensity_obj.get_table
                if table is not None:
                    # Reset index for consistent access
                    if hasattr(table, 'index'):
                        table.index = range(len(table.index))
                    return table
            
            # Method 2: Build from MoleculeLineList and intensity arrays
            if hasattr(molecule, 'lines') and molecule.lines is not None:
                lines = molecule.lines
                
                # Get line data efficiently
                if hasattr(lines, 'get_pandas_table'):
                    line_df = lines.get_pandas_table()
                    if line_df is not None:
                        # Add intensity data
                        if hasattr(intensity_obj, 'intensity') and intensity_obj.intensity is not None:
                            line_df = line_df.copy()
                            line_df['intens'] = intensity_obj.intensity[:len(line_df)]
                            
                            # Add tau if available
                            if hasattr(intensity_obj, 'tau') and intensity_obj.tau is not None:
                                line_df['tau'] = intensity_obj.tau[:len(line_df)]
                            
                            return line_df
                
                # Fallback: Build from lines_as_namedtuple
                elif hasattr(lines, 'lines_as_namedtuple'):
                    line_data = lines.lines_as_namedtuple
                    if line_data is not None and hasattr(intensity_obj, 'intensity'):
                        intensity_array = intensity_obj.intensity
                        tau_array = getattr(intensity_obj, 'tau', None)
                        
                        # Build DataFrame
                        data = {
                            'lam': line_data.lam,
                            'intens': intensity_array[:len(line_data.lam)],
                            'a_stein': line_data.a_stein,
                            'g_up': line_data.g_up,
                            'e_up': line_data.e_up,
                            'e_low': getattr(line_data, 'e_low', None),
                            'g_low': getattr(line_data, 'g_low', None)
                        }
                        
                        if tau_array is not None:
                            data['tau'] = tau_array[:len(line_data.lam)]
                        
                        return pd.DataFrame(data)
            
            return None
            
        except Exception as e:
            print(f"Error getting intensity table: {e}")
            return None
        
    def validate_cache_coherence(self, molecule: 'Molecule') -> bool:
        try:
            if molecule is None:
                return True
                
            molecule_name = self._get_molecule_display_name(molecule)
            
            if hasattr(molecule, 'is_cache_valid') and not molecule.is_cache_valid('spectrum'):
                if molecule_name in self._plot_cache['spectrum_plots']:
                    del self._plot_cache['spectrum_plots'][molecule_name]
                return False
            
            return True
        except Exception as e:
            print(f"Error validating cache coherence: {e}")
            return False
    
    def sync_molecule_cache(self, molecule: 'Molecule') -> None:
        try:
            if molecule is None:
                return
                
            molecule_name = self._get_molecule_display_name(molecule)
            
            if not self.validate_cache_coherence(molecule):
                current_hash = self._get_molecule_parameters_hash(molecule)
                self._plot_cache['spectrum_plots'][molecule_name] = current_hash
                
                pop_cache = self._plot_cache['population_diagram']
                if pop_cache['molecule_id'] == molecule_name:
                    pop_cache['param_hash'] = current_hash
                
        except Exception as e:
            print(f"Error syncing with molecule cache: {e}")
            
    def get_cache_debug_info(self, molecule: 'Molecule' = None) -> Dict[str, Any]:
        """
        Get detailed cache debugging information.
        
        Parameters
        ----------
        molecule : Molecule, optional
            Specific molecule to get debug info for
            
        Returns
        -------
        Dict[str, Any]
            Debug information about cache states
        """
        debug_info = {
            'plot_cache_stats': self.get_plot_performance_stats(),
            'population_diagram_cache': self._plot_cache['population_diagram'].copy(),
            'spectrum_plots_cache': self._plot_cache['spectrum_plots'].copy(),
            'line_threshold_settings': {
                'threshold_percent': self.get_line_intensity_threshold() * 100,
                'threshold_value': self.get_line_intensity_threshold()
            }
        }
        
        if molecule is not None:
            molecule_name = self._get_molecule_display_name(molecule)
            debug_info['molecule_specific'] = {
                'name': molecule_name,
                'current_param_hash': self._get_molecule_parameters_hash(molecule),
                'cached_param_hash': self._plot_cache['spectrum_plots'].get(molecule_name),
                'cache_coherent': self.validate_cache_coherence(molecule),
                'molecule_spectrum_valid': getattr(molecule, '_spectrum_valid', 'unknown'),
                'molecule_intensity_valid': getattr(molecule, '_intensity_valid', 'unknown'),
                'molecule_has_plot_data': (
                    hasattr(molecule, 'plot_lam') and molecule.plot_lam is not None and
                    hasattr(molecule, 'plot_flux') and molecule.plot_flux is not None
                )
            }
            
        return debug_info
    
    def get_line_intensity_threshold(self) -> float:
        """
        Get the line intensity threshold from user settings.
        
        Returns
        -------
        float
            Threshold as a percentage (0.0 to 1.0) of the strongest line intensity.
            Default is 0.3 (30%) if not specified in settings.
        """
        return getattr(self.islat, 'user_settings', {}).get('line_threshold', 0.3)
    
    def set_line_intensity_threshold(self, threshold_percent: float) -> None:
        """
        Set the line intensity threshold dynamically.
        
        This updates the user settings and can be used for runtime adjustments.
        
        Parameters
        ----------
        threshold_percent : float
            Threshold as a percentage (0.0 to 1.0) of the strongest line intensity.
        """
        if not hasattr(self.islat, 'user_settings'):
            print("Warning: user_settings not available, cannot set threshold")
            return
            
        # Validate threshold value
        threshold_percent = max(0.0, min(1.0, threshold_percent))
        
        # Update user settings
        self.islat.user_settings['line_threshold'] = threshold_percent
        
        print(f"Line intensity threshold updated to {threshold_percent*100:.1f}%")
        
        # Optionally trigger a plot refresh if plots are currently visible
        # This could be enhanced to automatically refresh active plots
    
    def filter_lines_by_threshold(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], 
                                 threshold_percent: Optional[float] = None) -> List[Tuple['MoleculeLine', float, Optional[float]]]:
        """
        Filter line data based on intensity threshold.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
        threshold_percent : Optional[float]
            Threshold percentage (0.0 to 1.0). If None, uses user settings.
            
        Returns
        -------
        List[Tuple['MoleculeLine', float, Optional[float]]]
            Filtered list of lines above threshold
        """
        if not line_data:
            return []
            
        if threshold_percent is None:
            threshold_percent = self.get_line_intensity_threshold()
        
        # Extract intensities for threshold calculation
        intensities = [intensity for _, intensity, _ in line_data]
        if not intensities:
            return []
            
        max_intensity = max(intensities)
        if max_intensity <= 0:
            return []
            
        # Calculate threshold intensity
        threshold_intensity = max_intensity * threshold_percent
        
        # Filter lines based on threshold
        filtered_lines = [(line, intensity, tau_val) for line, intensity, tau_val in line_data 
                         if intensity >= threshold_intensity]
        
        return filtered_lines
    
    def get_threshold_debug_info(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]]) -> Dict[str, Any]:
        """
        Get debugging information about threshold filtering.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
            
        Returns
        -------
        Dict[str, Any]
            Debug information about threshold filtering
        """
        if not line_data:
            return {'total_lines': 0, 'filtered_lines': 0, 'threshold_percent': 0}
            
        threshold_percent = self.get_line_intensity_threshold()
        filtered_lines = self.filter_lines_by_threshold(line_data, threshold_percent)
        
        intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(intensities) if intensities else 0
        min_intensity = min(intensities) if intensities else 0
        threshold_intensity = max_intensity * threshold_percent if max_intensity > 0 else 0
        
        return {
            'total_lines': len(line_data),
            'filtered_lines': len(filtered_lines),
            'threshold_percent': threshold_percent * 100,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'threshold_intensity': threshold_intensity,
            'lines_below_threshold': len(line_data) - len(filtered_lines)
        }
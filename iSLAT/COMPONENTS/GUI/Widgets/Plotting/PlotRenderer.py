import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.widgets import SpanSelector
#from iSLAT.iSLATDefaultInputParms import dist, au, pc, ccum, hh
import iSLAT.Constants as c

class PlotRenderer:
    """Pure plotting logic - handles all plot rendering and visual updates"""
    
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.islat = plot_manager.islat
        self.theme = plot_manager.theme
        
        # Plot references
        self.fig = plot_manager.fig
        self.ax1 = plot_manager.ax1  # Main spectrum plot
        self.ax2 = plot_manager.ax2  # Line inspection plot
        self.ax3 = plot_manager.ax3  # Population diagram
        self.canvas = plot_manager.canvas
        
        # Visual state
        self.model_lines = []
        self.active_lines = []
        
        # Cache for population diagram to prevent unnecessary re-rendering
        self._pop_diagram_cache = {
            'molecule_id': None,
            'parameters_hash': None,
            'wave_range': None
        }
        
    def clear_all_plots(self):
        """Clear all plots and reset visual state"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.model_lines.clear()
        self.active_lines.clear()
        # Clear population diagram cache
        self._pop_diagram_cache = {
            'molecule_id': None,
            'parameters_hash': None,
            'wave_range': None
        }
        
    def clear_model_lines(self):
        """Clear only the model spectrum lines from the main plot"""
        for line in self.model_lines:
            if line in self.ax1.lines:
                line.remove()
        self.model_lines.clear()
        
    def render_main_spectrum_plot(self, wave_data, flux_data, molecules, summed_flux=None, error_data=None):
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
            self._plot_model_spectra(wave_data, molecules)
            
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
        
    def _plot_observed_spectrum(self, wave_data, flux_data, error_data=None):
        """Plot the observed spectrum data"""
        if flux_data is not None and len(flux_data) > 0:
            if error_data is not None and len(error_data) == len(flux_data):
                # Plot with error bars
                self.ax1.errorbar(
                    wave_data, 
                    flux_data,
                    yerr=error_data,
                    fmt='-', 
                    color=self.theme.get("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=self.theme.get("zorder_observed", 3),
                    elinewidth=0.5,
                    capsize=0
                )
            else:
                # Plot without error bars
                self.ax1.plot(
                    wave_data, 
                    flux_data,
                    color=self.theme.get("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=self.theme.get("zorder_observed", 3)
                )
    
    def _plot_model_spectra(self, wave_data, molecules):
        """Plot individual molecule model spectra"""
        # Double-check visibility using the molecule's own is_visible attribute
        truly_visible = []
        for mol in molecules:
            # Check for is_visible attribute and ensure it's True
            if getattr(mol, 'is_visible', False):
                truly_visible.append(mol)
        
        if not truly_visible:
            return
        
        # Sort molecules by peak intensity for better visual layering
        mol_intensities = []
        for mol in truly_visible:
            mol_name = getattr(mol, 'name', getattr(mol, 'displaylabel', 'unknown'))
            try:
                # Prepare plot data or use molecule's spectrum directly
                if hasattr(mol, 'prepare_plot_data'):
                    mol.prepare_plot_data(wave_data)
                    if hasattr(mol, 'plot_flux') and hasattr(mol, 'plot_lam') and len(mol.plot_flux) > 0:
                        peak_intensity = np.max(mol.plot_flux) if len(mol.plot_flux) > 0 else 0
                        mol_intensities.append((peak_intensity, mol, mol.plot_lam, mol.plot_flux))
                    else:
                        # Fallback to interpolating spectrum data
                        if hasattr(mol, 'spectrum') and mol.spectrum is not None:
                            mol_flux_interp = np.interp(
                                wave_data,
                                mol.spectrum.lamgrid,
                                mol.spectrum.flux_jy,
                                left=0.0, right=0.0
                            )
                            peak_intensity = np.max(mol_flux_interp) if len(mol_flux_interp) > 0 else 0
                            mol_intensities.append((peak_intensity, mol, wave_data, mol_flux_interp))
                        else:
                            print(f"Warning: {mol_name} has no spectrum data")
                            continue
                else:
                    # Direct interpolation approach
                    if hasattr(mol, 'spectrum') and mol.spectrum is not None:
                        mol_flux_interp = np.interp(
                            wave_data,
                            mol.spectrum.lamgrid,
                            mol.spectrum.flux_jy,
                            left=0.0, right=0.0
                        )
                        peak_intensity = np.max(mol_flux_interp) if len(mol_flux_interp) > 0 else 0
                        mol_intensities.append((peak_intensity, mol, wave_data, mol_flux_interp))
                    else:
                        print(f"Warning: {mol_name} has no spectrum data")
                        continue
            except Exception as e:
                print(f"Error preparing plot data for {mol_name}: {e}")
                continue
        
        # Sort by peak intensity (lowest first for better visibility)
        mol_intensities.sort(key=lambda x: x[0])
        
        for peak_intensity, mol, plot_lam, plot_flux in mol_intensities:
            # Get proper molecule name
            mol_name = getattr(mol, 'name', getattr(mol, 'displaylabel', 'unknown'))
            
            if peak_intensity > 0:  # Only plot if there's actual flux
                try:
                    # Use the molecule's color if available, otherwise get from theme
                    color = getattr(mol, 'color', None)
                    if color is None:
                        # Get color from theme, or use default
                        color = self.theme.get('molecule_colors', {}).get(mol_name, 'blue')
                    
                    # Get proper label
                    label = getattr(mol, 'displaylabel', mol_name)

                    # Plot with style 
                    line, = self.ax1.plot(
                        plot_lam,
                        plot_flux,
                        linestyle='--',
                        color=color,
                        alpha=0.7,
                        linewidth=1,
                        label=label,
                        zorder=self.theme.get("zorder_model", 2)
                    )
                    self.model_lines.append(line)
                except Exception as e:
                    print(f"Error plotting {mol_name}: {e}")
                    continue
    
    def _plot_summed_spectrum(self, wave_data, summed_flux):
        """Plot the summed model spectrum"""
        if len(summed_flux) > 0 and np.any(summed_flux > 0):
            self.ax1.fill_between(
                wave_data,
                0,
                summed_flux,
                color=self.theme.get("summed_spectra_color", "lightgray"),
                alpha=1.0,
                label='Sum',
                zorder=self.theme.get("zorder_summed", 1)
            )
    
    def _configure_main_plot_appearance(self):
        """Configure the appearance of the main plot"""
        self.ax1.set_xlabel('Wavelength (μm)')
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax1.set_title("Full Spectrum with Line Inspection")
        
        # Only show legend if there are labeled items
        handles, labels = self.ax1.get_legend_handles_labels()
        if handles:
            self.ax1.legend()
        
    def render_line_inspection_plot(self, line_wave, line_flux, line_label=None):
        """Render the line inspection subplot"""
        self.ax2.clear()
        
        if line_wave is not None and line_flux is not None:
            # Plot data in selected range
            self.ax2.plot(line_wave, line_flux, 
                         color=self.theme.get("foreground", "black"), 
                         linewidth=1, 
                         label="Observed")
            
            self.ax2.set_xlabel("Wavelength (μm)")
            self.ax2.set_ylabel("Flux (Jy)")
            self.ax2.set_title("Line inspection plot")
            
            # Show legend if there are labeled items
            handles, labels = self.ax2.get_legend_handles_labels()
            if handles:
                self.ax2.legend()
    
    def _get_molecule_parameters_hash(self, molecule):
        """Generate a hash of molecule parameters to detect changes"""
        if molecule is None:
            return None
        
        try:
            # Collect key parameters that affect population diagram
            params = []
            
            # Basic molecule properties
            mol_name = getattr(molecule, 'name', getattr(molecule, 'displaylabel', 'unknown'))
            params.append(mol_name)
            
            # Physical parameters from intensity calculation
            if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                intensity_params = [
                    getattr(molecule.intensity, 't_kin', None),
                    getattr(molecule.intensity, 'n_mol', None), 
                    getattr(molecule.intensity, 'dv', None)
                ]
                params.extend(intensity_params)
                
                # Also check if intensity has been calculated (tau and intensity arrays)
                has_calculated_intensity = (
                    hasattr(molecule.intensity, 'tau') and 
                    molecule.intensity.tau is not None and
                    hasattr(molecule.intensity, 'intensity') and 
                    molecule.intensity.intensity is not None
                )
                params.append(has_calculated_intensity)
            else:
                params.extend([None, None, None, False])
            
            # Geometric parameters
            params.extend([
                getattr(molecule, 'radius', None),
                getattr(molecule, 'distance', None)
            ])
            
            # Convert to string and hash
            param_str = str(tuple(params))
            return hash(param_str)
        except Exception as e:
            print(f"Error generating parameter hash: {e}")
            # If hashing fails, return None to force re-render
            return None
    
    def invalidate_population_diagram_cache(self):
        """Force the population diagram to re-render on next call"""
        self._pop_diagram_cache = {
            'molecule_id': None,
            'parameters_hash': None, 
            'wave_range': None
        }
    
    def render_population_diagram(self, molecule, wave_range=None):
        """Render the population diagram for the active molecule"""
        # Check cache to avoid unnecessary re-rendering
        molecule_id = getattr(molecule, 'name', getattr(molecule, 'displaylabel', None)) if molecule else None
        parameters_hash = self._get_molecule_parameters_hash(molecule)
        
        # Check if we can skip rendering
        cache = self._pop_diagram_cache
        if (cache['molecule_id'] == molecule_id and 
            cache['parameters_hash'] == parameters_hash and 
            cache['wave_range'] == wave_range and
            molecule_id is not None):
            # Same molecule and parameters, skip re-rendering
            print(f"Population diagram cache hit for {molecule_id} - skipping render")
            return
        
        # Update cache
        cache['molecule_id'] = molecule_id
        cache['parameters_hash'] = parameters_hash
        cache['wave_range'] = wave_range
        
        self.ax3.clear()
        print(f"Actually rendering population diagram for {molecule_id}")
        
        if molecule is None:
            self.ax3.set_title("No molecule selected")
            return
            
        try:
            # Check if molecule has intensity data
            if not hasattr(molecule, 'intensity') or molecule.intensity is None:
                self.ax3.set_title(f"{getattr(molecule, 'displaylabel', 'Molecule')} - No intensity data")
                return
                
            # Get the intensity table
            int_pars = molecule.intensity.get_table
            int_pars.index = range(len(int_pars.index))

            # Parsing the components of the lines in int_pars
            wavelength = int_pars['lam']
            intens_mod = int_pars['intens']
            Astein_mod = int_pars['a_stein']
            gu = int_pars['g_up']
            eu = int_pars['e_up']

            # Calculating the y-axis for the population diagram
            # Get molecule radius safely
            radius = getattr(molecule, 'radius', 1.0)  # Default to 1.0 AU if not available
            distance = getattr(molecule, 'distance', getattr(self.islat, 'global_dist', 140.0))  # Default distance
            
            area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2  # In cm^2
            dist = distance * c.PARSEC_CM  # In cm
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
                self.ax3.scatter(eu, rd_yax, s=0.5, color=self.theme.get("scatter_main_color", '#838B8B'))

                # Set labels
                self.ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
                self.ax3.set_xlabel(r'$E_{u}$ (K)')
                mol_label = getattr(molecule, 'displaylabel', getattr(molecule, 'name', 'Molecule'))
                self.ax3.set_title(f'{mol_label} Population diagram', fontsize='medium')
            else:
                mol_label = getattr(molecule, 'displaylabel', getattr(molecule, 'name', 'Molecule'))
                self.ax3.set_title(f"{mol_label} - No valid data for population diagram")
            
        except Exception as e:
            print(f"Error rendering population diagram: {e}")
            mol_label = getattr(molecule, 'displaylabel', getattr(molecule, 'name', 'Molecule'))
            self.ax3.set_title(f"{mol_label} - Error in calculation")
    
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines on the main spectrum"""
        if not saved_lines:
            return
            
        for line in saved_lines:
            # Plot vertical lines at saved positions
            if 'wavelength' in line:
                self.ax1.axvline(
                    line['wavelength'], 
                    color=self.theme.get("saved_line_color", "orange"),
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
                    color=self.theme.get("saved_line_color_two", "coral"),
                    label=f"Saved Range: {line.get('label', 'Range')}"
                )
    
    def highlight_line_selection(self, xmin, xmax):
        """Highlight a selected wavelength range"""
        # Remove previous highlights
        for patch in self.ax1.patches:
            if hasattr(patch, '_islat_highlight'):
                patch.remove()
        
        # Add new highlight
        highlight = self.ax1.axvspan(xmin, xmax, alpha=0.3, color=self.theme.get("highlighted_line_color", "yellow"))
        highlight._islat_highlight = True
    
    def update_plot_display(self):
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_plot_refresh(self):
        """Force a complete plot refresh"""
        self.canvas.draw()
    
    def plot_vertical_lines(self, wavelengths, heights=None, colors=None, labels=None):
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
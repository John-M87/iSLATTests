import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector


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
        
    def clear_all_plots(self):
        """Clear all plots and reset visual state"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.model_lines.clear()
        self.active_lines.clear()
        
    def clear_model_lines(self):
        """Clear only the model spectrum lines from the main plot"""
        for line in self.model_lines:
            if line in self.ax1.lines:
                line.remove()
        self.model_lines.clear()
        
    def render_main_spectrum_plot(self, wave_data, flux_data, visible_molecules, summed_flux=None, error_data=None):
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
        if visible_molecules:
            self._plot_model_spectra(wave_data, visible_molecules)
            
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
        """Plot the observed spectrum data - matching original style"""
        if flux_data is not None and len(flux_data) > 0:
            if error_data is not None and len(error_data) == len(flux_data):
                # Plot with error bars - match original style
                self.ax1.errorbar(
                    wave_data, 
                    flux_data,
                    yerr=error_data,
                    fmt='-', 
                    color=self.theme.get("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=1,
                    elinewidth=0.5,
                    capsize=0
                )
            else:
                # Plot without error bars - match original style
                self.ax1.plot(
                    wave_data, 
                    flux_data,
                    color=self.theme.get("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=1
                )
    
    def _plot_model_spectra(self, wave_data, visible_molecules):
        """Plot individual molecule model spectra - matching original style"""
        # Double-check visibility using the molecule's own is_visible attribute
        truly_visible = []
        for mol in visible_molecules:
            # Check for is_visible attribute and ensure it's True (matching original logic)
            if getattr(mol, 'is_visible', False):
                truly_visible.append(mol)
        
        if not truly_visible:
            return
        
        # Sort molecules by peak intensity for better visual layering
        mol_intensities = []
        for mol in truly_visible:
            try:
                # Prepare plot data or use molecule's spectrum directly
                if hasattr(mol, 'prepare_plot_data'):
                    mol.prepare_plot_data(wave_data)
                    if hasattr(mol, 'plot_flux') and len(mol.plot_flux) > 0:
                        peak_intensity = np.max(mol.plot_flux)
                        mol_intensities.append((peak_intensity, mol, mol.plot_lam, mol.plot_flux))
                    else:
                        # Fallback to interpolating spectrum data
                        mol_flux_interp = np.interp(
                            wave_data,
                            mol.spectrum.lamgrid,
                            mol.spectrum.flux_jy,
                            left=0.0, right=0.0
                        )
                        peak_intensity = np.max(mol_flux_interp) if len(mol_flux_interp) > 0 else 0
                        mol_intensities.append((peak_intensity, mol, wave_data, mol_flux_interp))
                else:
                    # Direct interpolation approach like original
                    mol_flux_interp = np.interp(
                        wave_data,
                        mol.spectrum.lamgrid,
                        mol.spectrum.flux_jy,
                        left=0.0, right=0.0
                    )
                    peak_intensity = np.max(mol_flux_interp) if len(mol_flux_interp) > 0 else 0
                    mol_intensities.append((peak_intensity, mol, wave_data, mol_flux_interp))
            except Exception as e:
                print(f"Error preparing plot data for {getattr(mol, 'name', 'unknown')}: {e}")
                continue
        
        # Sort by peak intensity (lowest first for better visibility)
        mol_intensities.sort(key=lambda x: x[0])
        
        for peak_intensity, mol, plot_lam, plot_flux in mol_intensities:
            if peak_intensity > 0:  # Only plot if there's actual flux
                try:
                    # Use the molecule's color if available, otherwise get from theme
                    color = getattr(mol, 'color', None)
                    if color is None:
                        # Get color from theme like original, or use default
                        color = self.theme.get('molecule_colors', {}).get(getattr(mol, 'name', 'unknown'), 'blue')
                    
                    # Get proper label
                    label = getattr(mol, 'displaylabel', getattr(mol, 'name', 'Unknown'))
                    
                    # Plot with style matching original (solid line, not dashed)
                    line, = self.ax1.plot(
                        plot_lam,
                        plot_flux,
                        linestyle='-',  # Solid line like original
                        color=color,
                        alpha=0.7,      # Match original alpha
                        linewidth=1,    # Match original linewidth
                        label=label,
                        zorder=3
                    )
                    self.model_lines.append(line)
                except Exception as e:
                    print(f"Error plotting {getattr(mol, 'name', 'unknown molecule')}: {e}")
                    continue
    
    def _plot_summed_spectrum(self, wave_data, summed_flux):
        """Plot the summed model spectrum - matching original style with gray fill"""
        if len(summed_flux) > 0 and np.any(summed_flux > 0):
            # Use original gray fill styling like the old MainPlot
            self.ax1.fill_between(
                wave_data,
                0,
                summed_flux,
                color='lightgray',
                alpha=1.0,
                label='Sum',
                zorder=2  # Behind individual molecule lines but above background
            )
    
    def _configure_main_plot_appearance(self):
        """Configure the appearance of the main plot - matching original style"""
        self.ax1.set_xlabel('Wavelength (μm)')
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax1.set_title("Full Spectrum with Line Inspection")
        
        # Only show legend if there are labeled items
        handles, labels = self.ax1.get_legend_handles_labels()
        if handles:
            self.ax1.legend()
        
        # Don't add grid by default to match original
        
    def render_line_inspection_plot(self, line_wave, line_flux, line_label=None):
        """Render the line inspection subplot - matching original style"""
        self.ax2.clear()
        
        if line_wave is not None and line_flux is not None:
            # Plot data in selected range - match original style
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
    
    def render_population_diagram(self, molecule, wave_range=None):
        """Render the population diagram for the active molecule - matching original style"""
        self.ax3.clear()
        
        if molecule is None:
            self.ax3.set_title("No molecule selected")
            return
            
        try:
            # Use the original calculation method from the old MainPlot
            from iSLAT.iSLATDefaultInputParms import dist, au, pc, ccum, hh
            
            # Get the intensity table - same as original
            int_pars = molecule.intensity.get_table
            int_pars.index = range(len(int_pars.index))

            # Parsing the components of the lines in int_pars - exact same as original
            wl = int_pars['lam']
            intens_mod = int_pars['intens']
            Astein_mod = int_pars['a_stein']
            gu = int_pars['g_up']
            eu = int_pars['e_up']

            # Calculating the y-axis for the population diagram - exact same as original
            area = np.pi * (molecule.radius * au * 1e2) ** 2  # In cm^2
            Dist = dist * pc
            beam_s = area / Dist ** 2
            F = intens_mod * beam_s
            freq = ccum / wl
            rd_yax = np.log(4 * np.pi * F / (Astein_mod * hh * freq * gu))
            threshold = np.nanmax(F) / 100

            # Set limits exactly as original
            self.ax3.set_ylim(np.nanmin(rd_yax[F > threshold]), np.nanmax(rd_yax) + 0.5)
            self.ax3.set_xlim(np.nanmin(eu) - 50, np.nanmax(eu[F > threshold]))

            # Populating the population diagram graph with the lines - exact same as original but with picker
            self.ax3.scatter(eu, rd_yax, s=30, color='green', edgecolors='black', picker=True)
            
            # Set labels exactly as original
            self.ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
            self.ax3.set_xlabel(r'$E_{u}$ (K)')
            self.ax3.set_title(f'{molecule.displaylabel} Population diagram', fontsize='medium')
            
        except Exception as e:
            print(f"Error rendering population diagram: {e}")
            self.ax3.set_title(f"{molecule.displaylabel} - Error in calculation")
    
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines on the main spectrum"""
        if not saved_lines:
            return
            
        for line in saved_lines:
            # Plot vertical lines at saved positions
            if 'wavelength' in line:
                self.ax1.axvline(
                    line['wavelength'], 
                    color='orange', 
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
                    color='coral',
                    label=f"Saved Range: {line.get('label', 'Range')}"
                )
    
    def highlight_line_selection(self, xmin, xmax):
        """Highlight a selected wavelength range"""
        # Remove previous highlights
        for patch in self.ax1.patches:
            if hasattr(patch, '_islat_highlight'):
                patch.remove()
        
        # Add new highlight
        highlight = self.ax1.axvspan(xmin, xmax, alpha=0.3, color='yellow')
        highlight._islat_highlight = True
    
    def update_plot_display(self):
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_plot_refresh(self):
        """Force a complete plot refresh"""
        self.canvas.draw()
    
    def plot_vertical_lines(self, wavelengths, heights=None, colors=None, labels=None):
        """Plot vertical lines at specified wavelengths - matching original style"""
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

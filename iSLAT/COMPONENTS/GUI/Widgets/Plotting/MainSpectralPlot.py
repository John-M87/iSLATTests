import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk


class MainSpectralPlot:
    """Handles the main spectrum plot with observed data, model spectra, and sum"""
    
    def __init__(self, parent_frame, theme):
        self.parent_frame = parent_frame
        self.theme = theme
        
        # Create matplotlib figure and axis
        self.figure = plt.Figure(figsize=(5, 3.5))
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Visual state tracking
        self.model_lines = []
        self.active_lines = []
        
    def clear_plot(self):
        """Clear the main plot and reset visual state"""
        self.axis.clear()
        self.model_lines.clear()
        self.active_lines.clear()
        
    def clear_model_lines(self):
        """Clear only the model spectrum lines from the main plot"""
        for line in self.model_lines:
            if line in self.axis.lines:
                line.remove()
        self.model_lines.clear()
        
    def render_spectrum_plot(self, wave_data, flux_data, molecules, summed_flux=None, error_data=None):
        """Render the main spectrum plot with observed data, model spectra, and sum"""
        # Store current view limits
        current_xlim = self.axis.get_xlim() if hasattr(self.axis, 'get_xlim') else None
        current_ylim = self.axis.get_ylim() if hasattr(self.axis, 'get_ylim') else None
        
        # Clear the plot
        self.clear_plot()
        
        # Early return if no data
        if wave_data is None or len(wave_data) == 0:
            self.axis.set_title("No spectrum data loaded")
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
        self._configure_plot_appearance()
        
        # Restore view limits if they existed
        if current_xlim and current_ylim:
            if current_xlim != (0.0, 1.0) and current_ylim != (0.0, 1.0):
                self.axis.set_xlim(current_xlim)
                self.axis.set_ylim(current_ylim)
        
    def _plot_observed_spectrum(self, wave_data, flux_data, error_data=None):
        """Plot the observed spectrum data"""
        if flux_data is not None and len(flux_data) > 0:
            if error_data is not None and len(error_data) == len(flux_data):
                # Plot with error bars
                self.axis.errorbar(
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
                self.axis.plot(
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
                    # Direct interpolation approach check if this works right
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
                        # Get color from theme like, or use default
                        color = self.theme.get('molecule_colors', {}).get(getattr(mol, 'name', 'unknown'), 'blue')
                    
                    # Get proper label
                    label = getattr(mol, 'displaylabel', getattr(mol, 'name', 'Unknown'))
                    
                    # Plot with style 
                    line, = self.axis.plot(
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
                    print(f"Error plotting {getattr(mol, 'name', 'unknown molecule')}: {e}")
                    continue
    
    def _plot_summed_spectrum(self, wave_data, summed_flux):
        """Plot the summed model spectrum"""
        if len(summed_flux) > 0 and np.any(summed_flux > 0):
            self.axis.fill_between(
                wave_data,
                0,
                summed_flux,
                color='lightgray',
                alpha=1.0,
                label='Sum',
                zorder=self.theme.get("zorder_summed", 1)
            )
    
    def _configure_plot_appearance(self):
        """Configure the appearance of the main plot"""
        self.axis.set_xlabel('Wavelength (μm)')
        self.axis.set_ylabel('Flux density (Jy)')
        self.axis.set_title("Full Spectrum with Line Inspection")
        
        # Only show legend if there are labeled items
        handles, labels = self.axis.get_legend_handles_labels()
        if handles:
            self.axis.legend()
    
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines on the main spectrum"""
        if not saved_lines:
            return
            
        for line in saved_lines:
            # Plot vertical lines at saved positions
            if 'wavelength' in line:
                self.axis.axvline(
                    line['wavelength'], 
                    color='orange', 
                    alpha=0.7, 
                    linestyle=':', 
                    label=f"Saved: {line.get('label', 'Line')}"
                )
            elif 'xmin' in line and 'xmax' in line:
                # Plot wavelength range
                self.axis.axvspan(
                    line['xmin'], 
                    line['xmax'], 
                    alpha=0.2, 
                    color='coral',
                    label=f"Saved Range: {line.get('label', 'Range')}"
                )
    
    def highlight_line_selection(self, xmin, xmax):
        """Highlight a selected wavelength range"""
        # Remove previous highlights
        for patch in self.axis.patches:
            if hasattr(patch, '_islat_highlight'):
                patch.remove()
        
        # Add new highlight
        highlight = self.axis.axvspan(xmin, xmax, alpha=0.3, color='yellow')
        highlight._islat_highlight = True
    
    def update_display(self):
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_refresh(self):
        """Force a complete plot refresh"""
        self.canvas.draw()

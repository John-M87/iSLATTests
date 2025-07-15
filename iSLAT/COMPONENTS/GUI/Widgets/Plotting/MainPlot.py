import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

import iSLAT.Constants as c
from iSLAT.COMPONENTS.DataTypes.Molecule import Molecule

# Import the new modular classes
from .PlotRenderer import PlotRenderer
from iSLAT.COMPONENTS.DataProcessing.DataProcessor import DataProcessor
from iSLAT.COMPONENTS.GUI.InteractionHandler import InteractionHandler
from iSLAT.COMPONENTS.DataProcessing.FittingEngine import FittingEngine
from iSLAT.COMPONENTS.DataProcessing.LineAnalyzer import LineAnalyzer

class iSLATPlot:
    """
    Main plotting class for iSLAT spectroscopy tool.
    
    This class coordinates between specialized modules to provide comprehensive
    plotting functionality including spectrum visualization, line analysis,
    population diagrams, and interactive features.
    
    Architecture:
    - PlotRenderer: Handles matplotlib rendering and visual updates
    - DataProcessor: Manages data processing and caching operations  
    - InteractionHandler: Processes mouse/keyboard interactions
    - FittingEngine: Handles line fitting operations
    - LineAnalyzer: Provides line detection and analysis capabilities
    
    Data Sources:
    - Uses MoleculeLine objects for line data access
    - Leverages Spectrum and Intensity classes for computation
    - Integrates with MoleculeDict for parameter management
    """
    def __init__(self, parent_frame, wave_data, flux_data, theme, islat_class_ref):
        self.theme = theme
        self.islat = islat_class_ref

        self.active_lines = []  # List of (line, scatter) tuples for active molecular lines

        self.fig = plt.Figure(figsize=(10, 7))
        gs = GridSpec(2, 2, height_ratios=[2, 3], figure=self.fig)
        self.ax1 = self.full_spectrum = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.line_inspection = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.population_diagram = self.fig.add_subplot(gs[1, 1])

        self.ax1.set_title("Full Spectrum with Line Inspection")
        self.ax2.set_title("Line inspection plot")
        self.ax3.set_title(f"{self.islat.active_molecule.displaylabel} Population diagram")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent_frame)
        
        # Apply theme to matplotlib figure and toolbar
        self._apply_plot_theming()

        # Initialize the modular classes
        self.plot_renderer = PlotRenderer(self)
        self.data_processor = DataProcessor(self)
        self.interaction_handler = InteractionHandler(self)
        self.fitting_engine = FittingEngine(self.islat)
        self.line_analyzer = LineAnalyzer(self.islat)

        # Legacy attribute for compatibility - actual line management is in PlotRenderer
        self.model_lines = []  # This will be kept in sync with plot_renderer.model_lines

        # Set up interaction handler callbacks
        self.interaction_handler.set_span_select_callback(self.onselect)
        self.interaction_handler.set_click_callback(self.on_click)
        
        self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        self.selected_wave = None
        self.selected_flux = None
        self.fit_result = None

        # Initial data and model computation using new data structures
        self.summed_flux = np.array([])
        
        # Register callbacks for parameter and molecule changes
        self._register_update_callbacks()
        
        # Use update coordination if available
        if hasattr(self.islat, '_update_coordinator') and self.islat._update_coordinator:
            self.islat.request_update('model_spectrum')
            self.islat.request_update('plots')
        else:
            # Direct updates using optimized molecular data structures
            if hasattr(self.islat.molecules_dict, 'get_summed_flux_optimized'):
                # Use the new optimized flux calculation if available
                self.islat.molecules_dict.update_molecule_fluxes(self.islat.wave_data)
            else:
                # Standard approach for compatibility
                visible_molecules = [mol for mol in self.islat.molecules_dict.values() if mol.is_visible]
                self.data_processor.compute_summed_flux(
                    self.islat.wave_data, 
                    visible_molecules, 
                    visible_only=True
                )
            self.islat.update_model_spectrum()
            self.update_all_plots()
        
        # Set initial zoom range to display_range if available
        self._set_initial_zoom_range()

    def _apply_plot_theming(self):
        """Apply theme colors to matplotlib figure and toolbar"""
        try:
            # Set figure background color
            self.fig.patch.set_facecolor(self.theme.get("background", "#181A1B"))
            
            # Set axes background colors and text colors
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_facecolor(self.theme.get("graph_fill_color", "#282C34"))
                ax.tick_params(colors=self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")), which='both')
                ax.xaxis.label.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                ax.yaxis.label.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                ax.title.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                
                # Set spine colors
                for spine in ax.spines.values():
                    spine.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                    
                # Set grid colors if grid is enabled
                ax.grid(True, color=self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")), alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Apply theme to toolbar if possible
            if hasattr(self.toolbar, 'configure'):
                try:
                    self.toolbar.configure(bg=self.theme.get("toolbar", "#23272A"))
                except:
                    pass
            
            # Try to style toolbar buttons
            if hasattr(self.toolbar, 'winfo_children'):
                for child in self.toolbar.winfo_children():
                    try:
                        if hasattr(child, 'configure'):
                            child.configure(
                                bg=self.theme.get("toolbar", "#23272A"),
                                fg=self.theme.get("foreground", "#F0F0F0"),
                                activebackground=self.theme.get("selection_color", "#00FF99"),
                                activeforeground=self.theme.get("foreground", "#F0F0F0")
                            )
                    except:
                        pass
                    
            # Apply theme to canvas
            if hasattr(self.canvas.get_tk_widget(), 'configure'):
                try:
                    self.canvas.get_tk_widget().configure(bg=self.theme.get("background", "#181A1B"))
                except:
                    pass
                    
        except Exception as e:
            print(f"Could not apply plot theming: {e}")

    def _register_update_callbacks(self):
        """Register callbacks to handle parameter and molecule changes"""
        Molecule.add_molecule_parameter_change_callback(self.on_molecule_parameter_changed)
        
        # Register for global parameter changes if molecules_dict exists
        if hasattr(self.islat, 'molecules_dict'):
            self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_changed)

        # Register for active molecule changes
        self.islat.add_active_molecule_change_callback(self._on_active_molecule_changed)
    
    def _on_active_molecule_changed(self, old_molecule, new_molecule):
        """Handle active molecule changes"""
        self.on_active_molecule_changed()
    
    def _on_global_parameter_changed(self, parameter_name, old_value, new_value):
        """Handle global parameter changes that affect all molecules"""
        # Refresh plots when global parameters change
        self.update_all_plots()

    def match_display_range(self):
        # Sync plot xlim to islat.display_range if set, else update islat.display_range from plot
        if hasattr(self.islat, 'display_range'):
            # If display_range is set elsewhere, update plot xlim
            if self.islat.display_range:
                wmin, wmax = self.islat.display_range
                self.ax1.set_xlim(wmin, wmax)
            else:
                # If not set, initialize from current plot xlim
                self.islat.display_range = tuple(self.ax1.get_xlim())
        else:
            # If islat has no display_range attribute, do nothing
            return

        # Connect callback to update islat.display_range when user changes xlim
        def on_xlim_changed(ax):
            # Only update if changed by user (not programmatically)
            new_xlim = list(ax.get_xlim())
            if self.islat.display_range != new_xlim:
                self.islat.display_range = new_xlim

        # Avoid multiple connections
        if not hasattr(self, '_xlim_callback_connected'):
            self.ax1.callbacks.connect('xlim_changed', on_xlim_changed)
            self._xlim_callback_connected = True

        # Adjust y-limits as before
        wmin, wmax = self.ax1.get_xlim()
        mask = (self.islat.wave_data >= wmin) & (self.islat.wave_data <= wmax)
        range_flux_cnts = self.islat.flux_data[mask]
        if range_flux_cnts.size == 0:
            fig_height = np.nanmax(self.islat.flux_data)
            fig_bottom_height = 0
        else:
            fig_height = np.nanmax(range_flux_cnts)
            fig_bottom_height = np.nanmin(range_flux_cnts)
        self.ax1.set_ylim(ymin=fig_bottom_height, ymax=fig_height + (fig_height / 8))

        self.canvas.draw_idle()

    def _set_initial_zoom_range(self):
        """Set the initial zoom range based on display_range or data range"""
        # Use display_range if available
        if hasattr(self.islat, 'display_range') and self.islat.display_range:
            xmin, xmax = self.islat.display_range
            self.ax1.set_xlim(xmin, xmax)
        elif hasattr(self.islat, 'wave_data') and self.islat.wave_data is not None:
            # Fallback to full data range
            self.ax1.set_xlim(self.islat.wave_data.min(), self.islat.wave_data.max())
        
        # Update display to match and set optimal y-limits
        self.match_display_range()
        self.canvas.draw_idle()

    def make_span_selector(self):
        """Creates a SpanSelector for the main plot to select a region for line inspection."""
        self.span = self.interaction_handler.create_span_selector(self.ax1, self.theme["selection_color"])

    def update_all_plots(self):
        """
        Updates all plots in the GUI.
        This method leverages the molecular data model for updates.
        """
        self.update_model_plot()
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        self.plot_spectrum_around_line()

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.
        Uses data processing through the molecular data structures.
        """
        #print("Debug: Updating model plot...")
        
        # Calculate summed flux using MoleculeDict methods if available
        if hasattr(self.islat.molecules_dict, 'get_summed_flux_optimized'):
            summed_flux = self.islat.molecules_dict.get_summed_flux_optimized(
                self.islat.wave_data, visible_only=True
            )
            # Fix: get_visible_molecules_fast might be returning keys, not objects
            try:
                visible_molecules_raw = list(self.islat.molecules_dict.get_visible_molecules_fast())
                # Check if we got strings (keys) instead of molecule objects
                if visible_molecules_raw and isinstance(visible_molecules_raw[0], str):
                    #print("Debug: get_visible_molecules_fast returned keys, converting to objects")
                    visible_molecules = [self.islat.molecules_dict[key] for key in visible_molecules_raw if key in self.islat.molecules_dict]
                else:
                    visible_molecules = visible_molecules_raw
            except Exception as e:
                #print(f"Debug: Error with get_visible_molecules_fast: {e}, falling back to standard method")
                visible_molecules = [mol for mol in self.islat.molecules_dict.values() if mol.is_visible]
        else:
            # Fallback to standard data processor
            visible_molecules = [mol for mol in self.islat.molecules_dict.values() if mol.is_visible]
            summed_flux = self.data_processor.compute_summed_flux(
                self.islat.wave_data, 
                visible_molecules, 
                visible_only=True
            )
        
        '''#print(f"Debug: Found {len(visible_molecules)} visible molecules")
        for mol in visible_molecules:
            # Better molecule name detection
            mol_name = 'unknown'
            mol_type = type(mol).__name__
            if isinstance(mol, str):
                mol_name = mol
                #print(f"Warning: Got string instead of molecule object: {mol}")
                continue
            elif hasattr(mol, 'name') and mol.name:
                mol_name = mol.name
            elif hasattr(mol, 'displaylabel') and mol.displaylabel:
                mol_name = mol.displaylabel
            
            is_visible = getattr(mol, 'is_visible', 'unknown')
            #print(f"Debug: Visible molecule: {mol_name}, is_visible={is_visible}, type={mol_type}")'''
        
        # Filter out any string objects that might have slipped through
        actual_molecules = [mol for mol in visible_molecules if not isinstance(mol, str)]
        if len(actual_molecules) != len(visible_molecules):
            #print(f"Warning: Filtered out {len(visible_molecules) - len(actual_molecules)} string objects")
            visible_molecules = actual_molecules
        
        # Delegate rendering to PlotRenderer for clean separation of concerns
        self.plot_renderer.render_main_spectrum_plot(
            self.islat.wave_data,
            self.islat.flux_data,
            molecules=visible_molecules,
            summed_flux=summed_flux,
            error_data=getattr(self.islat, 'err_data', None)
        )
        
        # Recreate span selector and redraw
        self.make_span_selector()
        self.canvas.draw_idle()
        
        # Synchronize legacy attributes for backwards compatibility
        self.sync_model_lines()

    def onselect(self, xmin, xmax):
        self.current_selection = (xmin, xmax)
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.selected_wave = self.islat.wave_data[mask]
        self.selected_flux = self.islat.flux_data[mask]
        self.islat.selected_wave = self.selected_wave
        self.islat.selected_flux = self.selected_flux
        self.last_xmin = xmin
        self.last_xmax = xmax

        if len(self.selected_wave) < 5:
            self.ax2.clear()
            self.canvas.draw_idle()
            return

        self.plot_spectrum_around_line(
            xmin=xmin,
            xmax=xmax
        )

    def plot_spectrum_around_line(self, xmin=None, xmax=None, highlight_strongest=True):
        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None

        if xmin is None or xmax is None:
            # If no selection but we need to update population diagram due to molecule/parameter changes
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()
            return

        # Get lines in range using the MoleculeLine API
        try:
            line_data = self.plot_renderer.get_molecule_lines_efficiently(self.islat.active_molecule, xmin, xmax)
            if not line_data:
                # Clear active lines and update population diagram even if no lines in range
                self.clear_active_lines()
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.canvas.draw_idle()
                return
        except Exception as e:
            print(f"Warning: Could not get line data: {e}")
            # Clear active lines and update population diagram even if no lines in range
            self.clear_active_lines()
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()
            return

        # Clear previous active_lines before plotting
        self.clear_active_lines()
        self.plot_line_inspection(xmin, xmax, line_data, highlight_strongest=highlight_strongest)
        self.plot_population_diagram(line_data)
        self.canvas.mpl_connect('pick_event', self.on_pick_line)
    
    def on_pick_line(self, event):
        """Handle line pick events by delegating to PlotRenderer."""
        picked_value = self.plot_renderer.handle_line_pick_event(event.artist, self.active_lines)
        if picked_value:
            self._display_line_info(picked_value)
        self.canvas.draw_idle()

    def find_strongest_line_from_data(self):
        """
        Find strongest line directly from line data using the MoleculeLine API.
        Returns a dictionary with line information ready for display.
        """
        if not hasattr(self, 'current_selection') or self.current_selection is None:
            return None
            
        xmin, xmax = self.current_selection
        
        # Use the MoleculeLine approach
        try:
            lines_with_intensity = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
            if not lines_with_intensity:
                return None
                
            # Find the line with maximum intensity
            strongest_line, strongest_intensity, strongest_tau = max(lines_with_intensity, key=lambda x: x[1])
            
            # Create a dictionary with the line information
            line_info = {
                'lam': strongest_line.lam,
                'e': strongest_line.e_up, 
                'a': strongest_line.a_stein,
                'g': strongest_line.g_up,
                'inten': strongest_intensity,
                'up_lev': strongest_line.lev_up if strongest_line.lev_up else 'N/A',
                'low_lev': strongest_line.lev_low if strongest_line.lev_low else 'N/A',
                'tau': strongest_tau if strongest_tau is not None else 'N/A',
                'wavelength': strongest_line.lam,
                'intensity': strongest_intensity,
                'flux': strongest_intensity
            }
            
            return line_info
        except Exception as e:
            print(f"Warning: Could not find strongest line: {e}")
            return None

    def flux_integral_basic(self, wave_data, flux_data, err_data, xmin, xmax):
        """
        Calculate the flux integral in a given wavelength range.
        
        Parameters:
        -----------
        wave_data : array
            Wavelength array
        flux_data : array
            Flux array
        err_data : array or None
            Error array (optional)
        xmin, xmax : float
            Wavelength range
            
        Returns:
        --------
        line_flux : float
            Integrated flux
        line_err : float
            Error on integrated flux
        """
        mask = (wave_data >= xmin) & (wave_data <= xmax)
        if not np.any(mask):
            return 0.0, 0.0
            
        wave_region = wave_data[mask]
        flux_region = flux_data[mask]
        
        if len(wave_region) < 2:
            return 0.0, 0.0
            
        # Integrate using trapezoidal rule
        line_flux = np.trapz(flux_region, wave_region)
        
        # Calculate error if available
        if err_data is not None:
            err_region = err_data[mask]
            # Simple error propagation for integration
            line_err = np.sqrt(np.sum(err_region**2)) * (wave_region[-1] - wave_region[0]) / len(wave_region)
        else:
            line_err = 0.0
            
        return line_flux, line_err

    def highlight_strongest_line(self):
        """
        Highlight the strongest line by delegating to PlotRenderer.
        """
        strongest = self.plot_renderer.highlight_strongest_line(self.active_lines)
        if strongest is not None:
            # Display strongest line information in data field
            line, scatter, value = strongest
            if value:
                self._display_line_info(value)
        
        self.canvas.draw_idle()

    def _display_line_info(self, value, clear_data_field=True):
        """
        Helper method to display line information in the data field.
        """
        # Calculate flux integral in selected range
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            # Calculate flux integral
            err_data = getattr(self.islat, 'err_data', None)
            line_flux, line_err = self.flux_integral_basic(
                self.islat.wave_data, 
                self.islat.flux_data, 
                err_data, 
                xmin, 
                xmax
            )
        else:
            line_flux = [0.0]
        
        # Extract line information
        lam = value.get('lam', None)
        e_up = value.get('e', None)
        a_stein = value.get('a', None)
        g_up = value.get('g', None)
        inten = value.get('inten', None)
        up_lev = value.get('up_lev', 'N/A')
        low_lev = value.get('low_lev', 'N/A')
        tau_val = value.get('tau', 'N/A')
        
        # Format values to match original output
        wavelength_str = f"{lam:.6f}" if lam is not None else 'N/A'
        einstein_str = f"{a_stein:.3e}" if a_stein is not None else 'N/A'
        energy_str = f"{e_up:.0f}" if e_up is not None else 'N/A'
        tau_str = f"{tau_val:.3f}" if isinstance(tau_val, (float, int)) else str(tau_val)
        flux_str = f"{line_flux[0]:.3e}" if isinstance(line_flux, (list, tuple)) and len(line_flux) > 0 else f"{line_flux:.3e}"

        # Display line information in the original format
        info_str = (
            "\n--- Line Information ---\n"
            "Selected line:\n"
            f"Upper level = {up_lev}\n"
            f"Lower level = {low_lev}\n"
            f"Wavelength (μm) = {wavelength_str}\n"
            f"Einstein-A coeff. (1/s) = {einstein_str}\n"
            f"Upper level energy (K) = {energy_str}\n"
            f"Opacity = {tau_str}\n"
            f"Flux in sel. range (erg/s/cm2) = {flux_str}\n"
        )
        
        # Add the information without clearing the data field
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
            self.islat.GUI.data_field.insert_text(info_str, clear_first=clear_data_field)

    def plot_line_inspection(self, xmin=None, xmax=None, line_data=None, highlight_strongest=True):
        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None
        
        if xmin is None or xmax is None:
            self.canvas.draw_idle()
            return
        
        # Get line data using the molecular line API
        if line_data is None:
            try:
                line_data = self.plot_renderer.get_molecule_lines_efficiently(self.islat.active_molecule, xmin, xmax)
                if not line_data:
                    self.ax2.clear()
                    self.canvas.draw_idle()
                    return
            except Exception as e:
                print(f"Warning: Could not get line data: {e}")
                self.ax2.clear()
                self.canvas.draw_idle()
                return

        # First update the basic line inspection plot
        self.update_line_inspection_plot(xmin=xmin, xmax=xmax)
        
        # Get the max y value for scaling line heights
        data_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        data_region_y = self.islat.flux_data[data_mask]
        max_y = np.nanmax(data_region_y) if len(data_region_y) > 0 else 1.0

        # Clear and add vertical lines using PlotRenderer
        self.clear_active_lines()
        self.plot_renderer.render_active_lines_in_line_inspection(line_data, self.active_lines, max_y)

        self.canvas.draw_idle()

        # Highlight the strongest line
        if highlight_strongest:
            self.highlight_strongest_line()

    def plot_population_diagram(self, line_data):
        """
        Plot population diagram for the currently active lines in the selected region.
        Uses the MoleculeLine-based data structure.
        
        Parameters
        ----------
        line_data : list
            List of (MoleculeLine, intensity, tau) tuples
        """
        # First update the base population diagram with current molecule parameters
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        
        # Add active line scatter points
        if line_data:
            self.plot_renderer.render_active_lines_in_population_diagram(line_data, self.active_lines)
        
        self.canvas.draw_idle()
        self.highlight_strongest_line()

    def update_line_inspection_plot(self, xmin=None, xmax=None):
        """
        Update the line inspection plot showing data and active molecule model in the selected range.
        Uses data access patterns for efficiency.
        """
        self.ax2.clear()

        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None

        if xmin is None or xmax is None or (xmax - xmin) < 0.0001:
            self.canvas.draw_idle()
            return

        # Plot observed data in selected range
        data_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        observed_wave = self.islat.wave_data[data_mask]
        observed_flux = self.islat.flux_data[data_mask]
        
        self.ax2.plot(observed_wave, observed_flux, 
                     color=self.theme["foreground"], linewidth=1, label="Observed")

        # Plot the active molecule model using spectrum API
        active_molecule = self.islat.active_molecule
        if active_molecule is not None:
            # Use the optimized spectrum access from PlotRenderer
            model_wave, model_flux = self.plot_renderer.get_molecule_spectrum_efficiently(active_molecule, self.islat.wave_data)
            
            if model_wave is not None and model_flux is not None:
                # Filter to selected range
                model_mask = (model_wave >= xmin) & (model_wave <= xmax)
                
                if np.any(model_mask):
                    model_wave_range = model_wave[model_mask]
                    model_flux_range = model_flux[model_mask]
                    
                    if len(model_wave_range) > 0 and len(model_flux_range) > 0:
                        label = getattr(active_molecule, 'displaylabel', getattr(active_molecule, 'name', 'Model'))
                        color = getattr(active_molecule, 'color', 'blue')
                        self.ax2.plot(model_wave_range, model_flux_range, 
                                     color=color, linestyle="--", 
                                     linewidth=1, label=label)

        # Calculate max_y for plot scaling
        max_y = np.nanmax(observed_flux) if len(observed_flux) > 0 else 1.0

        # Plot fit results if available
        if self.fit_result is not None:
            self._plot_fit_results_in_range(xmin, xmax, max_y)

        # Set plot properties
        self.ax2.set_xlim(xmin, xmax)
        self.ax2.set_ylim(0, max_y * 1.1)
        self.ax2.legend()
        self.ax2.set_title("Line inspection plot")
        self.ax2.set_xlabel("Wavelength (μm)")
        self.ax2.set_ylabel("Flux (Jy)")

        self.canvas.draw_idle()
    
    def _plot_fit_results_in_range(self, xmin, xmax, max_y):
        """Helper method to plot fit results in the line inspection plot."""
        gauss_fit, fitted_wave, fitted_flux = self.fit_result
        if gauss_fit is not None:
            # Create x_fit array for plotting (use intersection of fitted_wave and current range)
            x_fit_mask = (fitted_wave >= xmin) & (fitted_wave <= xmax)
            x_fit = fitted_wave[x_fit_mask]
            
            if len(x_fit) > 0:
                # Plot the total fit line
                total_flux = gauss_fit.eval(x=x_fit)
                self.ax2.plot(x_fit, total_flux, 
                             color=self.theme.get("total_fit_line_color", "red"), 
                             linestyle='-', linewidth=1, label="Total Fit Line")

                # Plot individual component lines if it's a multi-component fit
                if self.fitting_engine.is_multi_component_fit():
                    components = self.fitting_engine.evaluate_fit_components(x_fit)
                    component_prefixes = self.fitting_engine.get_component_prefixes()
                    
                    for i, prefix in enumerate(component_prefixes):
                        if prefix in components:
                            component_flux = components[prefix]
                            self.ax2.plot(x_fit, component_flux, 
                                         linestyle='--', linewidth=1, 
                                         label=f"Component {i+1}")

        # Reset fit_result after plotting
        self.fit_result = None

    def toggle_legend(self):
        ax1_leg = self.ax1.get_legend()
        ax2_leg = self.ax2.get_legend()
        if ax1_leg is not None:
            vis = not ax1_leg.get_visible()
            ax1_leg.set_visible(vis)
        if ax2_leg is not None:
            vis = not ax2_leg.get_visible()
            ax2_leg.set_visible(vis)
        self.canvas.draw_idle()

    def flux_integral(self, lam, flux, err, lam_min, lam_max):
        """
        Calculate flux integral in the selected wavelength range.
        
        Parameters
        ----------
        lam : array
            Wavelength array in microns
        flux : array  
            Flux array in Jy
        err : array
            Error array in Jy
        lam_min : float
            Minimum wavelength in microns
        lam_max : float
            Maximum wavelength in microns
            
        Returns
        -------
        tuple
            (line_flux_meas, line_err_meas) in erg/s/cm^2
        """
        # Use vectorized operations for efficiency
        wavelength_mask = (lam >= lam_min) & (lam <= lam_max)
        
        if not np.any(wavelength_mask):
            return 0.0, 0.0
            
        lam_range = lam[wavelength_mask]
        flux_range = flux[wavelength_mask]
        
        if len(lam_range) < 2:
            return 0.0, 0.0
        
        # Convert to frequency space for proper integration
        freq_range = c.SPEED_OF_LIGHT_KMS / lam_range
        
        # Integrate in frequency space (reverse order for proper frequency ordering)
        line_flux_meas = np.trapz(flux_range[::-1], x=freq_range[::-1])
        line_flux_meas = -line_flux_meas * 1e-23  # Convert Jy*Hz to erg/s/cm^2
        
        # Calculate error propagation if error data provided
        if err is not None:
            err_range = err[wavelength_mask]
            line_err_meas = np.trapz(err_range[::-1], x=freq_range[::-1])
            line_err_meas = -line_err_meas * 1e-23
        else:
            line_err_meas = 0.0
            
        return line_flux_meas, line_err_meas

    def clear_active_lines(self) -> None:
        """
        Clear active lines by delegating to PlotRenderer.
        """
        self.plot_renderer.clear_active_lines(self.active_lines)

    def clear_model_lines(self):
        """
        Clear model spectrum lines from the main plot.
        Delegates to PlotRenderer for efficient line management.
        """
        self.plot_renderer.clear_model_lines()
        # Keep legacy attribute in sync
        self.model_lines.clear()
        self.canvas.draw_idle()
    
    def clear_all_plots(self):
        """
        Clear all plots and reset visual state.
        Delegates to PlotRenderer for comprehensive plot clearing.
        """
        self.plot_renderer.clear_all_plots()
        self.canvas.draw_idle()
    
    def invalidate_population_diagram_cache(self):
        """
        Force the population diagram to re-render on next call.
        Useful when molecule parameters change significantly.
        """
        self.plot_renderer.invalidate_population_diagram_cache()
    
    def optimize_plot_memory(self):
        """
        Optimize memory usage for plotting operations.
        Delegates to PlotRenderer for memory management.
        """
        self.plot_renderer.optimize_plot_memory()
    
    def get_plot_performance_stats(self):
        """
        Get performance statistics for debugging.
        Returns dict with plot performance metrics.
        """
        return self.plot_renderer.get_plot_performance_stats()

    def highlight_line_selection(self, xmin, xmax):
        """
        Highlight a selected wavelength range.
        Delegates to PlotRenderer for visual highlighting.
        """
        self.plot_renderer.highlight_line_selection(xmin, xmax)
        self.canvas.draw_idle()
    
    def plot_vertical_lines(self, wavelengths, heights=None, colors=None, labels=None):
        """
        Plot vertical lines at specified wavelengths.
        Delegates to PlotRenderer for efficient line plotting.
        """
        self.plot_renderer.plot_vertical_lines(wavelengths, heights, colors, labels)
        self.canvas.draw_idle()
    
    def render_molecules_efficiently(self, wave_data, molecules):
        """
        Render molecules using available methods.
        Delegates to PlotRenderer for molecule rendering.
        """
        self.plot_renderer.render_molecules_efficiently(wave_data, molecules)
        self.canvas.draw_idle()
    
    def update_plot_display(self):
        """
        Update the plot display.
        Delegates to PlotRenderer for display updates.
        """
        self.plot_renderer.update_plot_display()
    
    def force_plot_refresh(self):
        """
        Force a complete plot refresh.
        Delegates to PlotRenderer for comprehensive refresh.
        """
        self.plot_renderer.force_plot_refresh()

    def on_click(self, event):
        """Handle mouse click events on the plot."""
        self.interaction_handler.handle_click_event(event)
    
    def on_active_molecule_changed(self):
        """
        Called when the active molecule changes.
        Updates plot titles and refreshes displays with current selection if available.
        """
        # Update the population diagram title
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            if isinstance(self.islat.active_molecule, str):
                self.ax3.set_title(f'{self.islat.active_molecule} - Population diagram not available')
            else:
                self.ax3.set_title(f'{self.islat.active_molecule.displaylabel} Population diagram')
        
        # Clear active lines since they belong to the previous molecule
        self.clear_active_lines()
        
        # If we have a current selection, refresh the line inspection and population diagram
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
        else:
            # Just update the population diagram without active lines
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()

    def on_molecule_parameter_changed(self, molecule_name, parameter_name, old_value, new_value):
        """
        Called when any molecule parameter changes.
        Refreshes displays if the changed molecule is the active one.
        """
        # Check if the changed molecule is the active one
        if (hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule and 
            hasattr(self.islat.active_molecule, 'name') and
            self.islat.active_molecule.name == molecule_name):
            
            # If we have a current selection, refresh the line inspection and population diagram
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
                self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
            else:
                # Just update the population diagram without active lines
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.canvas.draw_idle()

    def on_molecule_deleted(self, molecule_name):
        """
        Handle molecule deletion by clearing relevant plot elements and updating displays.
        
        Parameters
        ----------
        molecule_name : str
            Name of the deleted molecule
        """
        # Clear model lines first
        self.clear_model_lines()
        
        # Clear active lines if they belong to the deleted molecule
        if (hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule and 
            hasattr(self.islat.active_molecule, 'name') and
            self.islat.active_molecule.name == molecule_name):
            self.clear_active_lines()
        
        # Update all plots to reflect the change
        self.update_all_plots()
    
    def on_molecule_visibility_changed(self, molecule_name, is_visible):
        """
        Handle molecule visibility changes.
        
        Parameters
        ----------
        molecule_name : str
            Name of the molecule whose visibility changed
        is_visible : bool
            New visibility state
        """
        # Update model plot to reflect visibility changes
        self.update_model_plot()
        
        # If the active molecule's visibility changed, update line inspection
        if (hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule and 
            hasattr(self.islat.active_molecule, 'name') and
            self.islat.active_molecule.name == molecule_name and
            hasattr(self, 'current_selection') and self.current_selection):
            
            xmin, xmax = self.current_selection
            self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
    
    def batch_update_molecule_colors(self, molecule_color_map):
        """
        Update multiple molecule colors.
        
        Parameters
        ----------
        molecule_color_map : dict
            Dictionary mapping molecule names to colors
        """
        self.plot_renderer.batch_update_molecule_colors(molecule_color_map)
    
    # Convenience methods that delegate to specialized modules
    
    def compute_fit_line(self, xmin=None, xmax=None, deblend=False):
        """
        Compute fit line using FittingEngine with data access.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range. Uses current_selection if not provided.
        deblend : bool
            Whether to perform multi-component deblending
            
        Returns
        -------
        tuple or None
            (fit_result, fitted_wave, fitted_flux) or None if fitting fails
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                return None
        
        # Use vectorized mask for efficient data selection
        fit_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        x_fit = self.islat.wave_data[fit_mask]
        y_fit = self.islat.flux_data[fit_mask]
        
        if len(x_fit) < 5:
            return None
        
        try:
            fit_result, fitted_wave, fitted_flux = self.fitting_engine.fit_gaussian_line(
                x_fit, y_fit, xmin=xmin, xmax=xmax, deblend=deblend
            )
            self.fit_result = fit_result, fitted_wave, fitted_flux
            return self.fit_result
        except Exception as e:
            print(f"Error in fitting: {str(e)}")
            return None
    
    def find_single_lines(self, xmin=None, xmax=None):
        """
        Find single lines using LineAnalyzer with data processing.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range. Uses current_selection if not provided.
            
        Returns
        -------
        list
            List of detected lines with properties
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                return []
        
        # Efficient data selection using vectorized operations
        range_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        range_wave = self.islat.wave_data[range_mask]
        range_flux = self.islat.flux_data[range_mask]
        
        if len(range_wave) < 10:
            return []
        
        try:
            detected_lines = self.line_analyzer.detect_lines_automatic(
                range_wave, range_flux, detection_type='both'
            )
            
            # Create optimized line data structure
            self.single_lines_list = []
            ylim = self.ax1.get_ylim()
            
            for line in detected_lines:
                line_info = {
                    "wavelength": line['wavelength'], 
                    "ylim": ylim,
                    "strength": line['line_strength'],
                    "type": line['type']
                }
                self.single_lines_list.append(line_info)
            
            return self.single_lines_list
        except Exception as e:
            print(f"Error in line detection: {str(e)}")
            return []
    
    def plot_single_lines(self):
        """Plot single lines on main plot with rendering."""
        self.update_model_plot()
        if not hasattr(self, 'single_lines_list') or not self.single_lines_list:
            return
            
        # Extract wavelengths for batch plotting
        wavelengths = [line['wavelength'] for line in self.single_lines_list]
        # Delegate to PlotRenderer for plotting
        self.plot_vertical_lines(wavelengths)
    
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines using PlotRenderer with delegation."""
        self.plot_renderer.plot_saved_lines(saved_lines)

    def apply_theme(self, theme=None):
        """Apply theme to the plot and update colors"""
        if theme:
            self.theme = theme
        self._apply_plot_theming()
        # Refresh the plots to apply colors
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        # Also update any existing plots to use colors
        self._refresh_existing_plots()
    
    def _refresh_existing_plots(self):
        """Refresh existing plots to use theme colors"""
        try:
            # This method can be called to refresh plots after theme changes
            if hasattr(self, 'update_all_plots'):
                self.update_all_plots()
        except:
            pass

    @property 
    def model_lines_sync(self):
        """
        Synchronized access to model lines from PlotRenderer.
        Ensures legacy code has access to current model lines.
        """
        if hasattr(self, 'plot_renderer') and self.plot_renderer:
            # Keep legacy attribute in sync
            self.model_lines = self.plot_renderer.model_lines.copy()
        return self.model_lines
    
    def sync_model_lines(self):
        """
        Synchronize legacy model_lines attribute with PlotRenderer.
        Call this after operations that modify model lines.
        """
        if hasattr(self, 'plot_renderer') and self.plot_renderer:
            self.model_lines = self.plot_renderer.model_lines.copy()
    
    def update_model_plot_with_sync(self):
        """
        Update the model plot and ensure legacy attributes are synchronized.
        This method ensures backwards compatibility while using optimized rendering.
        """
        self.update_model_plot()
        self.sync_model_lines()
    
    def refresh_all_plots_with_sync(self):
        """
        Refresh all plots and synchronize legacy attributes.
        Use this method when you need to ensure complete consistency.
        """
        self.update_all_plots()
        self.sync_model_lines()
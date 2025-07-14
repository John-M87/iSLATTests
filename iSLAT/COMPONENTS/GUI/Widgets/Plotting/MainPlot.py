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
    Modern plotting class for iSLAT spectroscopy tool.
    
    This class coordinates between specialized modules to provide comprehensive
    plotting functionality including spectrum visualization, line analysis,
    population diagrams, and interactive features. Uses optimized data structures
    and APIs from the refactored molecular data model.
    
    Architecture:
    - PlotRenderer: Handles matplotlib rendering and visual updates
    - DataProcessor: Manages data processing and caching operations  
    - InteractionHandler: Processes mouse/keyboard interactions
    - FittingEngine: Handles line fitting operations
    - LineAnalyzer: Provides line detection and analysis capabilities
    
    Data Sources:
    - Uses MoleculeLine objects for line data access
    - Leverages Spectrum and Intensity classes for efficient computation
    - Integrates with modern MoleculeDict for parameter management
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

        # Set up interaction handler callbacks
        self.interaction_handler.set_span_select_callback(self.onselect)
        self.interaction_handler.set_click_callback(self.on_click)
        
        self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        self.selected_wave = None
        self.selected_flux = None
        self.fit_result = None

        self.model_lines = []

        # Initial data and model computation using new data structures
        self.summed_flux = np.array([])
        
        # Register callbacks for parameter and molecule changes
        self._register_update_callbacks()
        
        # Use modern update coordination if available
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
        Updates all plots in the GUI using modern data structures.
        This method leverages the new molecular data model for efficient updates.
        """
        self.update_model_plot()
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        self.plot_spectrum_around_line()

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.
        Uses optimized data processing through the new molecular data structures.
        """
        #print("Debug: Updating model plot...")
        
        # Calculate summed flux using optimized MoleculeDict methods if available
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

        # Get lines in range using the new MoleculeLine API
        try:
            line_data = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
            if not line_data:
                # Clear active lines and update population diagram even if no lines in range
                self.active_lines.clear()
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.canvas.draw_idle()
                return
        except Exception as e:
            print(f"Warning: Could not get line data: {e}")
            # Clear active lines and update population diagram even if no lines in range
            self.active_lines.clear()
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()
            return

        # Clear previous active_lines before plotting
        self.active_lines.clear()
        self.plot_line_inspection(xmin, xmax, line_data, highlight_strongest=highlight_strongest)
        self.plot_population_diagram(line_data)
        self.canvas.mpl_connect('pick_event', self.on_pick_line)
    
    def on_pick_line(self, event):
        artist = event.artist
        # Find which entry in self.active_lines was picked
        for idx, (line, scatter, value) in enumerate(self.active_lines):
            picked = False
            if artist is line:
                picked = True
            elif artist is scatter:
                picked = True
            # Reset all to green first
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('green')
            # If picked, highlight both line and scatter
            if picked:
                if line is not None:
                    line.set_color('orange')
                if scatter is not None:
                    scatter.set_facecolor('orange')
                if 'text_obj' in value and value['text_obj'] is not None:
                    value['text_obj'].set_color('orange')
                
                # Display line information using the helper method
                self._display_line_info(value)

        self.canvas.draw_idle()

    def get_active_line_values(self, line_data, max_y=None):
        """
        Prepares a list of dictionaries with line properties for plotting.
        Uses only the new MoleculeLine-based data structure.
        
        Parameters
        ----------
        line_data : list
            List of (MoleculeLine, intensity, tau) tuples from the new approach
        max_y : float, optional
            Maximum y value for scaling line heights
            
        Returns
        -------
        list
            List of dictionaries with line properties
        """
        values = []
        
        # Verify we have the correct data format
        if not isinstance(line_data, list) or not line_data or not isinstance(line_data[0], tuple):
            print("Warning: get_active_line_values expects list of (MoleculeLine, intensity, tau) tuples")
            return values
        
        # Extract intensities for normalization
        intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(intensities) if intensities else 1.0
        
        for line, intensity, tau_val in line_data:
            # Compute lineheight for later plotting
            lineheight = None
            if max_y is not None and max_intensity != 0:
                lineheight = (intensity / max_intensity) * max_y
            
            # Compute rd_yax for population diagram
            rd_yax = None
            if all(x is not None for x in [intensity, line.a_stein, line.g_up, line.lam]):
                area = np.pi * (self.islat.active_molecule.radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
                dist = self.islat.active_molecule.distance * c.PARSEC_CM
                beam_s = area / dist ** 2
                F = intensity * beam_s
                freq = c.SPEED_OF_LIGHT_MICRONS / line.lam
                rd_yax = np.log(4 * np.pi * F / (line.a_stein * c.PLANCK_CONSTANT * freq * line.g_up))
            
            values.append({
                'lam': line.lam,
                'lineheight': lineheight,
                'e': line.e_up,
                'a': line.a_stein,
                'g': line.g_up,
                'rd_yax': rd_yax,
                'inten': intensity,
                'up_lev': line.lev_up if line.lev_up else 'N/A',
                'low_lev': line.lev_low if line.lev_low else 'N/A',
                'tau': tau_val if tau_val is not None else 'N/A'
            })
        
        return values

    def find_strongest_line(self):
        """
        Finds and returns the [vline, sc, value] triplet in self.active_lines whose vline has the largest y value.
        Updates self.strongest_active_line with this triplet.
        """
        highest_y_value = -float('inf')
        strongest_triplet = None

        for vline, sc, value in self.active_lines:
            # vline is a LineCollection from vlines, get its segments
            segments = vline.get_segments() if vline is not None else []
            if not segments:
                continue
            # Each segment is [[x0, y0], [x1, y1]], vertical line so y0 and y1
            y_values = [pt[1] for seg in segments for pt in seg]
            max_y = max(y_values) if y_values else -float('inf')
            if max_y > highest_y_value:
                highest_y_value = max_y
                strongest_triplet = [vline, sc, value]

        self.strongest_active_line = strongest_triplet
        return strongest_triplet

    def find_strongest_line_from_data(self):
        """
        Find strongest line directly from line data using the new MoleculeLine API.
        Returns a dictionary with line information ready for display.
        """
        if not hasattr(self, 'current_selection') or self.current_selection is None:
            return None
            
        xmin, xmax = self.current_selection
        
        # Use the new MoleculeLine approach
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
        Highlights the strongest line (by height) in orange, others in green.
        Also automatically displays the strongest line information in the data field.
        """
        strongest = self.find_strongest_line()
        for line, scatter, value in self.active_lines:
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('green')
        
        if strongest is not None:
            line, scatter, value = strongest
            if line is not None:
                line.set_color('orange')
            if scatter is not None:
                scatter.set_facecolor('orange')
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('orange')
            
            # Automatically display strongest line information in data field
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
        
        # Get line data using the new molecular line API
        if line_data is None:
            try:
                line_data = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
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

        # Clear and repopulate self.active_lines
        self.active_lines.clear()
        values = self.get_active_line_values(line_data, max_y=max_y)
        
        # Plot vertical lines for each molecular line
        for v in values:
            if v['lineheight'] is not None and v['lineheight'] > 0:
                vline = self.ax2.vlines(v['lam'], 0, v['lineheight'],
                                        color=self.theme.get("active_scatter_line_color", "green"), linestyle='dashed', linewidth=1, picker=True)
                text = self.ax2.text(v['lam'], v['lineheight'],
                              f"{v['e']:.0f},{v['a']:.3f}", fontsize='x-small', color=self.theme.get("active_scatter_line_color", "green"), rotation=45)
                # Store text object with line data for color changes
                v['text_obj'] = text
                # Add placeholder for scatter, will be filled in plot_population_diagram
                self.active_lines.append([vline, None, v])

        self.canvas.draw_idle()

        # Highlight the strongest line
        if highlight_strongest:
            self.highlight_strongest_line()

    def plot_population_diagram(self, line_data):
        """
        Plot population diagram for the currently active lines in the selected region.
        Uses only the new MoleculeLine-based data structure.
        
        Parameters
        ----------
        line_data : list
            List of (MoleculeLine, intensity, tau) tuples
        """
        # First update the base population diagram with current molecule parameters
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        
        # Ensure we have valid line data
        if not line_data or not isinstance(line_data, list):
            return
        
        # Recalculate values using current molecule parameters
        values = self.get_active_line_values(line_data)

        # Clear existing active line scatter points and rebuild
        for line, scatter, value in self.active_lines:
            if scatter is not None:
                scatter.remove()
        
        # Update the scatter part of self.active_lines
        # Ensure self.active_lines has the same length as values
        while len(self.active_lines) < len(values):
            self.active_lines.append([None, None, values[len(self.active_lines)]])
        
        # Add scatter points with updated parameters
        for idx, v in enumerate(values):
            if idx < len(self.active_lines) and v['rd_yax'] is not None:
                sc = self.ax3.scatter(v['e'], v['rd_yax'], s=30, color=self.theme.get("scatter_main_color", 'green'), edgecolors='black', picker=True)
                # Update the scatter object and value in active_lines
                self.active_lines[idx][1] = sc
                self.active_lines[idx][2] = v
        
        self.canvas.draw_idle()
        self.highlight_strongest_line()

    def update_line_inspection_plot(self, xmin=None, xmax=None):
        """
        Update the line inspection plot showing data and active molecule model in the selected range.
        Uses modern data access patterns for efficiency.
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

        # Plot the active molecule model using modern spectrum API
        active_molecule = self.islat.active_molecule
        if active_molecule is not None and hasattr(active_molecule, 'spectrum'):
            # Use the spectrum object's optimized data access
            model_wave = active_molecule.spectrum.lamgrid
            model_mask = (model_wave >= xmin) & (model_wave <= xmax)
            
            if np.any(model_mask):
                model_wave_range = model_wave[model_mask]
                model_flux_range = active_molecule.spectrum.flux_jy[model_mask]
                
                if len(model_wave_range) > 0 and len(model_flux_range) > 0:
                    label = getattr(active_molecule, 'displaylabel', active_molecule.name)
                    self.ax2.plot(model_wave_range, model_flux_range, 
                                 color=active_molecule.color, linestyle="--", 
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
        Calculate flux integral in the selected wavelength range using modern approach.
        
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
        self.active_lines.clear()
        
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

    # Convenience methods that delegate to specialized modules
    
    def compute_fit_line(self, xmin=None, xmax=None, deblend=False):
        """
        Compute fit line using FittingEngine with optimized data access.
        
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
        Find single lines using LineAnalyzer with optimized data processing.
        
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
        """Plot single lines on main plot with efficient rendering."""
        self.update_model_plot()
        if not hasattr(self, 'single_lines_list') or not self.single_lines_list:
            return
            
        # Batch plot all vertical lines for efficiency
        wavelengths = [line['wavelength'] for line in self.single_lines_list]
        ylims = [line['ylim'] for line in self.single_lines_list]
        
        if wavelengths:
            # Use vectorized vlines for better performance
            for wavelength, ylim in zip(wavelengths, ylims):
                self.ax1.vlines(
                    wavelength, ylim[0], ylim[1],
                    linestyles='dashed',
                    color=self.theme.get("single_line_color", "blue"),
                )
        self.canvas.draw_idle()
    
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines using PlotRenderer with optimized delegation."""
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
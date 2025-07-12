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
    def __init__(self, parent_frame, wave_data, flux_data, theme, islat_class_ref):
        #self.wave_data = wave_data
        #self.flux_data = flux_data
        self.theme = theme
        self.islat = islat_class_ref

        self.active_lines = []  # List of (line, scatter) tuples for green lines/scatter

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

        # Initialize the new modular classes
        self.plot_renderer = PlotRenderer(self)
        self.data_processor = DataProcessor(self)
        self.interaction_handler = InteractionHandler(self)
        self.fitting_engine = FittingEngine(self.islat)
        self.line_analyzer = LineAnalyzer(self.islat)

        # Set up interaction handler callbacks
        self.interaction_handler.set_span_select_callback(self.onselect)
        self.interaction_handler.set_click_callback(self.on_click)
        
        #self.make_span_selector()
        self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        self.selected_wave = None
        self.selected_flux = None
        self.fit_result = None

        self.model_lines = []

        # Initial setup without redundant calculations
        self.summed_flux = np.array([])
        
        # Register callbacks for parameter and molecule changes
        self._register_update_callbacks()
        
        # Use coordinator if available, otherwise fallback to direct calls
        if hasattr(self.islat, '_update_coordinator') and self.islat._update_coordinator:
            self.islat.request_update('model_spectrum')
            self.islat.request_update('plots')
        else:
            self.compute_sum_flux_visible()
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
        #try:
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
            
        #except Exception as e:
        #    print(f"Warning: Could not set initial zoom range: {e}")

    def make_span_selector(self):
        """
        Creates a SpanSelector for the main plot to select a region for line inspection.
        Delegated to InteractionHandler for cleaner separation of concerns.
        """
        self.span = self.interaction_handler.create_span_selector(self.ax1, self.theme["selection_color"])

    def clear_model_lines(self):
        """
        Remove previously plotted model lines.
        Delegated to PlotRenderer for cleaner separation of concerns.
        """
        self.plot_renderer.clear_model_lines()
        self.canvas.draw_idle()

    def update_all_plots(self):
        """
        Updates all plots in the GUI.
        This function is called when the user changes parameters or loads new data.
        """
        self.update_model_plot()
        self.update_population_diagram()
        self.plot_spectrum_around_line()

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.
        Delegated to PlotRenderer for cleaner separation of concerns.
        """
        # Calculate summed flux for visible molecules
        summed_flux = self.compute_sum_flux_visible()
        
        # Get all molecules (let PlotRenderer handle visibility filtering)
        #all_molecules = list(self.islat.molecules_dict.values())
        
        # Delegate to PlotRenderer
        self.plot_renderer.render_main_spectrum_plot(
            self.islat.wave_data,
            self.islat.flux_data,
            molecules = list(self.islat.molecules_dict.values()),  # Pass all molecules, PlotRenderer will filter by visibility
            summed_flux=summed_flux,
            error_data=getattr(self.islat, 'err_data', None)
        )
        
        # Recreate span selector and redraw
        self.make_span_selector()
        self.canvas.draw_idle()

    def update_population_diagram(self):
        """
        Updates the population diagram plot.
        Delegated to PlotRenderer for cleaner separation of concerns.
        """
        # Delegate to PlotRenderer
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        self.canvas.draw_idle()
    
    def plot_saved_lines(self, saved_lines):
        """
        Plots saved lines on the main plot.
        Expects saved_lines to be a list of dictionaries with 'wavelength', 'xmin', and 'xmax' keys.
        """
        for line in saved_lines:
            wave = line['wavelength']
            self.ax1.vlines(wave, self.ax1.get_ylim()[0], self.ax1.get_ylim()[1], linestyles='dashed', color='red', label=f"Saved Line at {wave:.4f} μm")
            if 'xmin' in line and 'xmax' in line:
                self.ax1.vlines(line['xmin'], self.ax1.get_ylim()[0], self.ax1.get_ylim()[1], color='coral', alpha=0.5, label=f"Range Start at {line['xmin']:.4f} μm")
                self.ax1.vlines(line['xmax'], self.ax1.get_ylim()[0], self.ax1.get_ylim()[1], color='coral', alpha=0.5, label=f"Range End at {line['xmax']:.4f} μm")

        self.ax1.legend()
        self.canvas.draw_idle()

    def compute_sum_flux_all(self):
        """
        Computes the sum of all model fluxes using DataProcessor.
        """
        return self.data_processor.compute_summed_flux(
            self.islat.wave_data, 
            self.islat.molecules_dict.values(), 
            visible_only=False
        )

    def compute_sum_flux_visible(self):
        """
        Optimized method to compute sum of visible molecule fluxes using DataProcessor.
        """
        return self.data_processor.compute_summed_flux(
            self.islat.wave_data, 
            self.islat.molecules_dict.values(), 
            visible_only=True
        )

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
            self.update_population_diagram()
            self.canvas.draw_idle()
            return

        line_data = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
        if line_data.empty:
            # Clear active lines and update population diagram even if no lines in range
            self.active_lines.clear()
            self.update_population_diagram()
            self.canvas.draw_idle()
            return

        # Clear previous active_lines before plotting new ones
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
        Does not create or store matplotlib objects.
        """
        lam = line_data['lam']
        intensities = line_data['intens']
        e_up = line_data['e_up']
        a_stein = line_data['a_stein']
        g_up = line_data['g_up']
        lev_up = line_data['lev_up'] if 'lev_up' in line_data else None
        lev_low = line_data['lev_low'] if 'lev_low' in line_data else None
        tau = line_data['tau'] if 'tau' in line_data else None
        max_intensity = intensities.max()
        values = []
        for i, (lam_val, inten, e, a, g) in enumerate(zip(lam, intensities, e_up, a_stein, g_up)):
            # Compute lineheight for later plotting
            lineheight = None
            if max_y is not None and max_intensity != 0:
                lineheight = (inten / max_intensity) * max_y
            # Compute rd_yax for population diagram
            rd_yax = None
            if all(x is not None for x in [inten, a, g, lam_val]):
                area = np.pi * (self.islat.active_molecule.radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
                dist = self.islat.active_molecule.distance * c.PARSEC_CM
                beam_s = area / dist ** 2
                F = inten * beam_s
                freq = c.SPEED_OF_LIGHT_MICRONS / lam_val
                rd_yax = np.log(4 * np.pi * F / (a * c.PLANCK_CONSTANT * freq * g))
            
            # Get additional fields with safe indexing
            up_lev = lev_up.iloc[i] if lev_up is not None else 'N/A'
            low_lev = lev_low.iloc[i] if lev_low is not None else 'N/A'
            tau_val = tau.iloc[i] if tau is not None else 'N/A'
            
            values.append({
                'lam': lam_val, 
                'lineheight': lineheight, 
                'e': e, 
                'a': a, 
                'g': g, 
                'rd_yax': rd_yax, 
                'inten': inten,
                'up_lev': up_lev,
                'low_lev': low_lev,
                'tau': tau_val
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
        Alternative method to find strongest line directly from line data,
        which returns a dictionary with line information ready for display.
        """
        if not hasattr(self, 'current_selection') or self.current_selection is None:
            return None
            
        xmin, xmax = self.current_selection
        
        # Get line data for the selected range
        line_data = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
        if line_data.empty:
            return None
            
        # Find the line with maximum intensity
        max_idx = line_data['intens'].idxmax()
        strongest_line_data = line_data.loc[max_idx]
        
        # Create a dictionary with the line information
        line_info = {
            'lam': strongest_line_data['lam'],
            'e': strongest_line_data['e_up'], 
            'a': strongest_line_data['a_stein'],
            'g': strongest_line_data['g_up'],
            'inten': strongest_line_data['intens'],
            'up_lev': strongest_line_data.get('lev_up', 'N/A'),
            'low_lev': strongest_line_data.get('lev_low', 'N/A'),
            'tau': strongest_line_data.get('tau', 'N/A'),
            'wavelength': strongest_line_data['lam'],  # For compatibility
            'intensity': strongest_line_data['intens'],  # For compatibility
            'flux': strongest_line_data['intens']  # For compatibility
        }
        
        return line_info

    def flux_integral(self, wave_data, flux_data, err_data, xmin, xmax):
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
        Highlights the strongest line (by height) in orange by default, others in green.
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
            line_flux, line_err = self.flux_integral(
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
        
        if line_data is None or line_data.empty:
            line_data = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
            if line_data.empty:
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
        
        # Plot vertical lines for each molecular line - match original style
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

        # Highlight the strongest line by default
        if highlight_strongest:
            self.highlight_strongest_line()

    def plot_population_diagram(self, line_data):
        # First update the base population diagram with current molecule parameters
        self.update_population_diagram()
        
        # Ensure we have valid line data
        if line_data is None or line_data.empty:
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
        
        # Add new scatter points with updated parameters
        for idx, v in enumerate(values):
            if idx < len(self.active_lines) and v['rd_yax'] is not None:
                sc = self.ax3.scatter(v['e'], v['rd_yax'], s=30, color=self.theme.get("scatter_main_color", 'green'), edgecolors='black', picker=True)
                # Update the scatter object and value in active_lines
                self.active_lines[idx][1] = sc
                self.active_lines[idx][2] = v
        
        self.canvas.draw_idle()
        self.highlight_strongest_line()

    def update_line_inspection_plot(self, xmin=None, xmax=None):
        self.ax2.clear()

        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None

        if xmin is None or xmax is None or (xmax - xmin) < 0.0001:
            self.canvas.draw_idle()
            return

        # Plot data in selected range - match original style
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.ax2.plot(self.islat.wave_data[mask], self.islat.flux_data[mask], color=self.theme["foreground"], linewidth=1, label="Observed")

        # Plot the active molecule in the line inspection plot - match original style
        active_molecule = self.islat.active_molecule
        if active_molecule is not None:
            wavegrid = active_molecule.spectrum.lamgrid
            mol_mask = (wavegrid >= xmin) & (wavegrid <= xmax)
            data = wavegrid[mol_mask]
            flux = active_molecule.spectrum.flux_jy[mol_mask]
            if len(data) > 0 and len(flux) > 0:
                label = getattr(active_molecule, 'displaylabel', active_molecule.name)
                self.ax2.plot(data, flux, color=active_molecule.color, linestyle="--", linewidth=1, label=label)
            max_y = np.nanmax(self.islat.flux_data[mask]) if np.any(mask) else 1.0
        else:
            max_y = np.nanmax(self.islat.flux_data[mask]) if np.any(mask) else 1.0

        # Plot the fit line using the compute_fit_line function
        if self.fit_result is not None:
            gauss_fit, fitted_wave, fitted_flux = self.fit_result
            if gauss_fit is not None:
                # Create x_fit array for plotting (use intersection of fitted_wave and current range)
                x_fit_mask = (fitted_wave >= xmin) & (fitted_wave <= xmax)
                x_fit = fitted_wave[x_fit_mask]
                
                if len(x_fit) > 0:
                    # Plot the total fit line
                    total_flux = gauss_fit.eval(x=x_fit)
                    self.ax2.plot(x_fit, total_flux, color=self.theme.get("total_fit_line_color", "red"), linestyle='-', linewidth=1, label="Total Fit Line")
                    max_y = max(max_y, np.nanmax(total_flux))

                    # Plot individual component lines if it's a multi-component fit
                    if self.fitting_engine.is_multi_component_fit():
                        components = self.fitting_engine.evaluate_fit_components(x_fit)
                        component_prefixes = self.fitting_engine.get_component_prefixes()
                        
                        for i, prefix in enumerate(component_prefixes):
                            if prefix in components:
                                component_flux = components[prefix]
                                self.ax2.plot(x_fit, component_flux, linestyle='--', linewidth=1, label=f"Component {i+1}")

            self.fit_result = None

        # Set limits and labels to match original
        self.ax2.set_xlim(xmin, xmax)
        self.ax2.set_ylim(0, max_y * 1.1)
        self.ax2.legend()
        self.ax2.set_title("Line inspection plot")
        self.ax2.set_xlabel("Wavelength (μm)")
        self.ax2.set_ylabel("Flux (Jy)")

        self.canvas.draw_idle()

    def find_single_lines(self, xmin=None, xmax=None):
        """
        Finds isolated (single) molecular lines using LineAnalyzer.
        Delegates line detection to the LineAnalyzer class.
        """
        # Use current selection if not provided
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                print("No selection made for finding single lines.")
                return

        # Get data in selected range
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        range_wave = self.islat.wave_data[mask]
        range_flux = self.islat.flux_data[mask]

        if len(range_wave) < 10:
            print("Not enough data points for line detection.")
            return

        try:
            # Use LineAnalyzer for automatic line detection
            detected_lines = self.line_analyzer.detect_lines_automatic(
                range_wave, range_flux, detection_type='both'
            )
            
            # Convert to format expected by plotting functions
            self.single_lines_list = []
            for line in detected_lines:
                vline = {
                    "wavelength": line['wavelength'], 
                    "ylim": self.ax1.get_ylim(),
                    "strength": line['line_strength'],
                    "type": line['type']
                }
                self.single_lines_list.append(vline)

            # Feedback to user via GUI data field if available
            counter = len(detected_lines)
            if hasattr(self.islat, "GUI") and hasattr(self.islat.GUI, "data_field"):
                if counter == 0:
                    self.islat.GUI.data_field.insert_text('No lines detected in the current wavelength range.')
                else:
                    emission_count = len([l for l in detected_lines if l['type'] == 'emission'])
                    absorption_count = len([l for l in detected_lines if l['type'] == 'absorption'])
                    self.islat.GUI.data_field.insert_text(
                        f'Detected {counter} lines ({emission_count} emission, {absorption_count} absorption) '
                        f'in the current wavelength range.'
                    )

        except Exception as e:
            print(f"Error in line detection: {str(e)}")
            self.single_lines_list = []

        self.canvas.draw_idle()
        return self.single_lines_list

    def plot_single_lines(self):
        """
        Plots single (isolated) lines on the main plot.
        This function is called after find_single_lines.
        """
        self.update_model_plot()
        if not hasattr(self, 'single_lines_list') or not self.single_lines_list:
            print("No single lines to plot.")
            return
        for vline in self.single_lines_list:
            self.ax1.vlines(
                vline['wavelength'],
                vline['ylim'][0],
                vline['ylim'][1],
                linestyles='dashed',
                color=self.theme.get("single_line_color", "blue"),
            )
        self.canvas.draw_idle()

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
            Wavelength array
        flux : array
            Flux array
        err : array
            Error array
        lam_min : float
            Minimum wavelength
        lam_max : float
            Maximum wavelength
            
        Returns
        -------
        tuple
            (line_flux_meas, line_err_meas) in erg/s/cm^2
        """
        # Calculate flux integral
        integral_range = np.where(np.logical_and(lam > lam_min, lam < lam_max))
        line_flux_meas = np.trapz(flux[integral_range[::-1]], x=c.SPEED_OF_LIGHT_KMS / lam[integral_range[::-1]])
        line_flux_meas = -line_flux_meas * 1e-23  # to get (erg s-1 cm-2); it's using frequency array, so need the - in front of it
        
        if err is not None:
            line_err_meas = np.trapz(err[integral_range[::-1]], x=c.SPEED_OF_LIGHT_MICRONS / lam[integral_range[::-1]])
            line_err_meas = -line_err_meas * 1e-23  # to get (erg s-1 cm-2); it's using frequency array, so need the - in front of it
        else:
            line_err_meas = 0.0
            
        return line_flux_meas, line_err_meas

    def on_click(self, event):
        """
        Handle mouse click events on the plot.
        Delegated to InteractionHandler for processing.
        """
        self.interaction_handler.handle_click_event(event)
    
    def on_active_molecule_changed(self): # a lot of this should be moved to the plot renderer
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
            self.update_population_diagram()
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
                self.update_population_diagram()
                self.canvas.draw_idle()

    # Convenience methods to expose new modular functionality
    
    def perform_line_analysis(self, xmin=None, xmax=None):
        """
        Perform comprehensive line analysis on selected region.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range for analysis. If None, uses current selection.
            
        Returns
        -------
        analysis_results : dict
            Comprehensive analysis results
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                print("No selection made for line analysis.")
                return {}
        
        # Get data in range
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        range_wave = self.islat.wave_data[mask]
        range_flux = self.islat.flux_data[mask]
        
        if len(range_wave) < 10:
            print("Not enough data points for analysis.")
            return {}
        
        # Detect lines
        detected_lines = self.line_analyzer.detect_lines_automatic(range_wave, range_flux)
        
        # Measure line properties for each detected line
        analysis_results = {
            'detected_lines': detected_lines,
            'line_measurements': {},
            'summary': self.line_analyzer.get_analysis_summary()
        }
        
        for line in detected_lines:
            measurements = self.line_analyzer.measure_line_properties(
                range_wave, range_flux, line['wavelength']
            )
            analysis_results['line_measurements'][line['wavelength']] = measurements
        
        return analysis_results
    
    def fit_voigt_profile(self, xmin=None, xmax=None):
        """
        Fit a Voigt profile to the selected spectral region.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range for fitting
            
        Returns
        -------
        fit_result : tuple
            Voigt fit result, fitted wavelength, fitted flux
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                print("No selection made for Voigt fitting.")
                return None
        
        # Get data in range
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        range_wave = self.islat.wave_data[mask]
        range_flux = self.islat.flux_data[mask]
        
        if len(range_wave) < 5:
            print("Not enough data points for Voigt fitting.")
            return None
        
        try:
            return self.fitting_engine.fit_voigt_profile(range_wave, range_flux)
        except Exception as e:
            print(f"Error in Voigt fitting: {str(e)}")
            return None
    
    def calculate_equivalent_widths(self, line_centers):
        """
        Calculate equivalent widths for a list of line centers.
        
        Parameters
        ----------
        line_centers : list
            List of wavelengths to calculate equivalent widths for
            
        Returns
        -------
        ew_results : dict
            Dictionary mapping line centers to equivalent width measurements
        """
        ew_results = {}
        
        for center in line_centers:
            ew, continuum = self.line_analyzer.calculate_equivalent_width(
                self.islat.wave_data, self.islat.flux_data, center
            )
            ew_results[center] = {
                'equivalent_width': ew,
                'continuum_flux': continuum
            }
        
        return ew_results
    
    def export_analysis_results(self, filename=None):
        """
        Export current analysis results to file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, auto-generated.
        """
        self.line_analyzer.export_line_analysis(filename)
        
        # Also export fitting results if available
        if hasattr(self, 'fit_result') and self.fit_result:
            self.fitting_engine.save_fit_results(filename)
    
    def get_fit_statistics(self):
        """
        Get statistics from the last fitting operation.
        
        Returns
        -------
        stats : dict
            Fitting statistics
        """
        return self.fitting_engine.get_fit_statistics()
    
    def get_line_parameters(self):
        """
        Extract line parameters from the last fit.
        
        Returns
        -------
        params : dict
            Line parameters (center, width, amplitude, etc.)
        """
        return self.fitting_engine.extract_line_parameters()
    
    def identify_detected_lines(self, tolerance=0.01):
        """
        Identify detected lines using available databases.
        
        Parameters
        ----------
        tolerance : float, optional
            Wavelength tolerance for line matching (microns)
            
        Returns
        -------
        identified_lines : list
            List of identified lines with species information
        """
        return self.line_analyzer.identify_lines(tolerance)
    
    def set_line_detection_parameters(self, min_snr=None, min_width=None, max_width=None):
        """
        Configure line detection parameters.
        
        Parameters
        ----------
        min_snr : float, optional
            Minimum signal-to-noise ratio
        min_width : float, optional
            Minimum line width (microns)
        max_width : float, optional
            Maximum line width (microns)
        """
        self.line_analyzer.set_detection_parameters(min_snr, min_width, max_width)
    
    def process_spectrum_data(self, molecules=None, visible_only=True):
        """
        Process spectrum data using DataProcessor.
        
        Parameters
        ----------
        molecules : list, optional
            List of molecules to process. If None, uses all molecules.
        visible_only : bool, optional
            Only process visible molecules
            
        Returns
        -------
        processed_data : dict
            Processed spectrum data
        """
        if molecules is None:
            molecules = self.islat.molecules_dict.values()
        
        return self.data_processor.process_molecule_spectra(molecules, visible_only)

    def apply_theme(self, theme=None):
        """Public method to apply theme to the plot and update colors"""
        if theme:
            self.theme = theme
        self._apply_plot_theming()
        # Refresh the plots to apply new colors
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        # Also update any existing plots to use new colors
        self._refresh_existing_plots()
    
    def _refresh_existing_plots(self):
        """Refresh existing plots to use new theme colors"""
        try:
            # This method can be called to refresh plots after theme changes
            if hasattr(self, 'update_all_plots'):
                self.update_all_plots()
        except:
            pass
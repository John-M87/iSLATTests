import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import numpy as np
from lmfit.models import GaussianModel
from iSLAT.ir_model import Spectrum
from iSLAT.iSLATDefaultInputParms import dist, au, pc, ccum, hh, specsep

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

        self.make_span_selector()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent_frame)
        self.toolbar.pack(side="top", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        self.selected_wave = None
        self.selected_flux = None
        self.fit_result = None

        self.model_lines = []

        self.compute_sum_flux_visible()
        self.islat.update_model_spectrum()
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
                self.islat.display_range = list(self.ax1.get_xlim())
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

    def make_span_selector(self):
        """
        Creates a SpanSelector for the main plot to select a region for line inspection.
        """
        self.span = SpanSelector(
            self.ax1, self.onselect, "horizontal",
            useblit=True, interactive=True,
            props=dict(alpha=0.5, facecolor=self.theme["selection_color"])
        )

    def clear_model_lines(self):
        # remove previously plotted lines
        for line in self.model_lines:
            line.remove()
        self.model_lines.clear()
        self.canvas.draw_idle()

    def plot_model_lines(self):
        """
        Plots all model lines for the molecules in the islat.molecules_dict.
        Assumes the islat.molecules_dict has the MolData instances for calc.
        """
        self.clear_model_lines()
        for mol_name, molecule_obj in self.islat.molecules_dict.items():
            if molecule_obj.is_visible:
                self.add_model_line(
                    mol_name,
                    temp=molecule_obj.temp,
                    radius=molecule_obj.radius,
                    density=molecule_obj.n_mol_init,
                )

    def update_all_plots(self):
        """
        Updates all plots in the GUI.
        This function is called when the user changes parameters or loads new data.
        """
        self.update_model_plot()
        self.update_population_diagram()
        #self.update_line_inspection_plot()
        self.plot_spectrum_around_line()
        '''if hasattr(self, 'update_model_plot'):
            self.update_model_plot()
        if hasattr(self, 'update_population_diagram'):
            self.update_population_diagram()
        if hasattr(self, 'update_line_inspection_plot'):
            self.update_line_inspection_plot()'''
        #self.islat.update_model_spectrum()
        #self.canvas.draw_idle()

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.
        Optimized for performance and better visual organization.
        """
        # Store current view limits before clearing
        current_xlim = self.ax1.get_xlim() if hasattr(self.ax1, 'get_xlim') else None
        current_ylim = self.ax1.get_ylim() if hasattr(self.ax1, 'get_ylim') else None
        
        # Clear the main plot
        self.ax1.clear()
        
        # Early return if no data available
        if not hasattr(self.islat, 'wave_data') or self.islat.wave_data is None:
            self.ax1.set_title("No spectrum data loaded")
            self.canvas.draw_idle()
            return
        
        # Plot observed spectrum with error bars if available (full data, no masking for zooming)
        self._plot_observed_spectrum(self.islat.wave_data, self.islat.flux_data, None)
        
        # Calculate and plot model spectra (full data, no masking for zooming)
        visible_molecules = [mol for mol in self.islat.molecules_dict.values() if mol.is_visible]
        if visible_molecules:
            summed_flux = self._plot_model_spectra(self.islat.wave_data, visible_molecules)
            self._plot_summed_spectrum(self.islat.wave_data, summed_flux)
        
        # Configure plot appearance
        self._configure_plot_appearance()
        
        # Restore view limits if they existed, otherwise set display range
        if current_xlim is not None and current_ylim is not None:
            # Check if limits are reasonable (not default matplotlib limits)
            if current_xlim != (0.0, 1.0) and current_ylim != (0.0, 1.0):
                self.ax1.set_xlim(current_xlim)
                self.ax1.set_ylim(current_ylim)
            else:
                self.match_display_range()
        else:
            self.match_display_range()
        
        # Recreate span selector and redraw
        self.make_span_selector()
        self.canvas.draw_idle()

    def add_model_line(self, mol_name, temp, radius, density, color = None):
        """
        Adds a model spectrum line for given molecule parameters to the main plot.
        Assumes the islat.molecules dict has the MolData instances for calc.
        """
        molecule_obj = self.islat.molecules_dict[mol_name]
        model_flux = molecule_obj.spectrum.flux_jy

        if color is None:
            color = molecule_obj.color
        
        # Use display label instead of mol_name for the plot legend
        display_label = getattr(molecule_obj, 'displaylabel', mol_name)
        line, = self.ax1.plot(molecule_obj.spectrum.lamgrid, model_flux, linestyle='-', color=color, alpha=0.7, label=display_label)

        self.model_lines.append(line)
        self.ax1.legend()
        self.canvas.draw_idle()

    def plot_data_line(self, wave, flux, label=None, color=None):
        """
        Plots a data line on the main plot.
        """
        if label is None:
            label = "Data Line"
        if color is None:
            color = self.theme["foreground"]
        
        print("Plotting data line with wavelength and flux:")
        print("Wavelength:", wave)
        print("Flux:", flux)
        line, = self.ax1.plot(wave, flux, linestyle='-', color=color, alpha=0.7, label=label)
        self.model_lines.append(line)
        self.ax1.legend()
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
        Computes the sum of all model fluxes (regardless of visibility) and updates the summed_flux array,
        ensuring all fluxes are interpolated onto the common wave_data grid.
        """
        summed_flux = np.zeros_like(self.islat.wave_data)
        for mol in self.islat.molecules_dict.values():
            # Interpolate molecule's flux onto the common grid
            flux_on_common_grid = np.interp(
                self.islat.wave_data,
                mol.spectrum.lamgrid,
                mol.spectrum.flux_jy,
                left=0.0, right=0.0
            )
            summed_flux += flux_on_common_grid
        self.summed_flux = summed_flux
        return summed_flux

    def compute_sum_flux_visible(self):
        """
        Optimized method to compute sum of visible molecule fluxes.
        Uses vectorized operations where possible.
        """
        if not hasattr(self.islat, 'wave_data') or self.islat.wave_data is None:
            return np.array([])
        
        summed_flux = np.zeros_like(self.islat.wave_data)
        
        for mol in self.islat.molecules_dict.values():
            if mol.is_visible and hasattr(mol, 'spectrum'):
                # Vectorized interpolation
                mol_flux = np.interp(
                    self.islat.wave_data,
                    mol.spectrum.lamgrid,
                    mol.spectrum.flux_jy,
                    left=0.0, 
                    right=0.0
                )
                summed_flux += mol_flux
        
        self.summed_flux = summed_flux
        return summed_flux

    def plot_sum_line(self, wave, flux, label=None, color=None, compute = True):
        """
        Plots the sum line on the main plot.
        """
        if compute:
            flux = self.compute_sum_flux_visible()
        if label is None:
            label = "Sum Line"
        if color is None:
            color = self.theme["highlight"]
        
        line, = self.ax1.plot(wave, flux, linestyle='--', color=color, alpha=0.7, label=label)
        self.model_lines.append(line)
        self.ax1.legend()
        self.canvas.draw_idle()

    def compute_fit_line(self, xmin=None, xmax=None, deblend=False):
        """
        Computes a fit line (or lines) for the selected region using LMFIT.
        Automatically determines line centers for single or multi-fit.
        If not enough data for all Gaussians, reduces number of components.
        Returns fit results but does not plot.
        """
        # Use selected region if not provided
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                print("No selection made for fitting.")
                return None

        # Get data in selected range
        fit_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        x_fit = self.islat.wave_data[fit_mask]
        y_fit = self.islat.flux_data[fit_mask]
        err_fit = getattr(self.islat, "err_data", None)
        if err_fit is not None:
            err_fit = err_fit[fit_mask]
        else:
            err_fit = np.ones_like(y_fit)

        if len(x_fit) < 5:
            print("Not enough data points for fitting.")
            return None

        # Automatically determine line centers from active molecule's intensity table in range
        line_table = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
        line_centers = np.array(line_table['lam'])
        # Remove duplicate/very close lines
        if len(line_centers) > 1:
            line_centers = np.sort(line_centers)
            min_sep = 1e-4  # μm, adjust as needed
            filtered_centers = [line_centers[0]]
            for lc in line_centers[1:]:
                if np.all(np.abs(lc - np.array(filtered_centers)) > min_sep):
                    filtered_centers.append(lc)
            line_centers = np.array(filtered_centers)

        # Sort line centers by intensity (descending), so least important are last
        if len(line_centers) > 1:
            intensities = np.array(line_table['intens'])
            # Ensure both arrays are the same length
            min_len = min(len(line_centers), len(intensities))
            line_centers = line_centers[:min_len]
            intensities = intensities[:min_len]
            sort_idx = np.argsort(-intensities)
            line_centers = line_centers[sort_idx]
            intensities = intensities[sort_idx]
        else:
            intensities = np.array(line_table['intens']) if len(line_centers) > 0 else np.array([])

        def extract_fit_results(gauss_fit, prefixes):
            fit_results = []
            for prefix in prefixes:
                p = gauss_fit.params
                c = p[prefix + "center"].value
                fwhm = p[prefix + "fwhm"].value / c * ccum
                fwhm_err = (p[prefix + "fwhm"].stderr / c * ccum) if p[prefix + "fwhm"].stderr is not None else np.nan
                sigma_freq = ccum / (c ** 2) * p[prefix + "sigma"].value
                sigma_freq_err = (ccum / (c ** 2) * p[prefix + "sigma"].stderr) if p[prefix + "sigma"].stderr is not None else np.nan
                gauss_area = p[prefix + "height"].value * sigma_freq * np.sqrt(2 * np.pi) * 1.e-23
                if p[prefix + "height"].stderr is not None:
                    gauss_area_err = np.abs(gauss_area * np.sqrt(
                        (p[prefix + "height"].stderr / p[prefix + "height"].value) ** 2 +
                        (sigma_freq_err / sigma_freq) ** 2))
                else:
                    gauss_area_err = np.nan
                fit_results.append({
                    'center': c,
                    'center_err': p[prefix + "center"].stderr,
                    'fwhm': fwhm,
                    'fwhm_err': fwhm_err,
                    'gauss_area': gauss_area,
                    'gauss_area_err': gauss_area_err
                })
            return fit_results

        # Multi-Gaussian fit if deblend requested and multiple centers found
        if deblend and len(line_centers) > 1:
            max_gaussians = len(line_centers)
            # Try reducing number of Gaussians if not enough data
            while max_gaussians > 0:
                num_params = 3 * max_gaussians  # center, sigma, amplitude per Gaussian
                if len(x_fit) >= num_params * 2:
                    break
                max_gaussians -= 1
            use_centers = line_centers[:max_gaussians]
            # use_intens is not used, so we remove it to avoid the unused variable warning
                #return None
            # Use only the most important (strongest) lines
            use_centers = line_centers[:max_gaussians]
            use_intens = intensities[:max_gaussians]
            models = []
            params = None
            prefixes = []
            for i, center in enumerate(use_centers):
                prefix = f"g{i+1}_"
                prefixes.append(prefix)
                model = GaussianModel(prefix=prefix)
                if params is None:
                    params = model.make_params()
                else:
                    params.update(model.make_params())
                # Set initial values and bounds
                params[prefix + "center"].set(value=center, min=center-0.01, max=center+0.01)
                params[prefix + "sigma"].set(value=0.001, min=0.0001, max=0.01)
                params[prefix + "amplitude"].set(value=(y_fit.max()-y_fit.min())*0.1, min=0)
                models.append(model)
            mod = models[0]
            for m in models[1:]:
                mod += m
            fitmethod = 'leastsq'
            gauss_fit = mod.fit(y_fit, params, x=x_fit, weights=1/err_fit, method=fitmethod, nan_policy='omit')
            fit_results = extract_fit_results(gauss_fit, prefixes)
            self.fit_result = gauss_fit, fit_results, x_fit
            return self.fit_result
        else:
            # Single Gaussian fit, center at strongest line or middle of region
            if len(line_centers) > 0:
                center_guess = line_centers[0]
            else:
                center_guess = np.mean([xmin, xmax])
            model = GaussianModel()
            params = model.guess(y_fit, x=x_fit)
            params['center'].set(value=center_guess, min=x_fit.min(), max=x_fit.max())
            gauss_fit = model.fit(y_fit, params, x=x_fit, weights=1/err_fit, nan_policy='omit')
            fit_results = extract_fit_results(gauss_fit, [""])
            self.fit_result = gauss_fit, fit_results, x_fit
            return self.fit_result

    def onselect(self, xmin, xmax):
        self.current_selection = (xmin, xmax)
        #self.update_line_inspection_plot(xmin, xmax)
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

        # Fit and update line
        #self.fit_result = self.islat.fit_selected_line(xmin, xmax)
        #self.update_line_inspection_plot(xmin, xmax)
        #self.update_population_diagram()
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
            self.canvas.draw_idle()
            return

        line_data = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
        if line_data.empty:
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
                area = np.pi * (self.islat.active_molecule.radius * au * 1e2) ** 2
                Dist = self.islat.active_molecule.distance * pc
                beam_s = area / Dist ** 2
                F = inten * beam_s
                freq = ccum / lam_val
                rd_yax = np.log(4 * np.pi * F / (a * hh * freq * g))
            
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
            "Strongest line:\n"
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
                                        color='green', linestyle='dashed', linewidth=1, picker=True)
                text = self.ax2.text(v['lam'], v['lineheight'],
                              f"{v['e']:.0f},{v['a']:.3f}", fontsize='x-small', color='green', rotation=45)
                # Store text object with line data for color changes
                v['text_obj'] = text
                # Add placeholder for scatter, will be filled in plot_population_diagram
                self.active_lines.append([vline, None, v])

        self.canvas.draw_idle()

        # Highlight the strongest line by default
        if highlight_strongest:
            self.highlight_strongest_line()

    def plot_population_diagram(self, line_data):
        self.update_population_diagram()
        values = self.get_active_line_values(line_data)
        # Update the scatter part of self.active_lines
        # Ensure self.active_lines has the same length as values
        while len(self.active_lines) < len(values):
            self.active_lines.append([None, None, values[len(self.active_lines)]])
        for idx, v in enumerate(values):
            if v['rd_yax'] is not None:
                sc = self.ax3.scatter(v['e'], v['rd_yax'], s=30, color='green', edgecolors='black', picker=True)
                # Update the second value in each triplet in self.active_lines to the new scatter object (sc)
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
            gauss_fit, fit_results, x_fit = self.fit_result
            if gauss_fit is not None:
                # Plot the total fit line
                self.ax2.plot(x_fit, gauss_fit.eval(x=x_fit), color="red", linestyle='-', linewidth=1, label="Total Fit Line")
                max_y = max(max_y, np.nanmax(gauss_fit.eval(x=x_fit)))

                # Plot individual component lines if it's a deblended fit
                if len(fit_results) > 1:  # Check if multiple components exist
                    for i, fit_result in enumerate(fit_results):
                        prefix = f'g{i+1}_'
                        component_flux = gauss_fit.eval_components(x=x_fit)[prefix]
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

    def update_population_diagram(self):
        self.ax3.clear()
        self.ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
        self.ax3.set_xlabel(r'$E_{u}$ (K)')
        
        # Handle special cases where active_molecule is not a Molecule object
        active_mol = self.islat.active_molecule
        if isinstance(active_mol, str) or active_mol is None:
            # If active_molecule is "SUM", "ALL", or None, show a message and return
            if active_mol in ["SUM", "ALL"]:
                self.ax3.set_title(f'{active_mol} - Population diagram not available', fontsize='medium')
                self.ax3.text(0.5, 0.5, f'Population diagram not available for {active_mol}', 
                             transform=self.ax3.transAxes, ha='center', va='center')
            else:
                self.ax3.set_title('No molecule selected', fontsize='medium')
                self.ax3.text(0.5, 0.5, 'No molecule selected', 
                             transform=self.ax3.transAxes, ha='center', va='center')
            self.canvas.draw_idle()
            return
        
        # Check if it's a valid Molecule object with required attributes
        if not (hasattr(active_mol, 'displaylabel') and hasattr(active_mol, 'intensity')):
            self.ax3.set_title('Invalid molecule selected', fontsize='medium')
            self.ax3.text(0.5, 0.5, 'Invalid molecule selected', 
                         transform=self.ax3.transAxes, ha='center', va='center')
            self.canvas.draw_idle()
            return

        self.ax3.set_title(f'{active_mol.displaylabel} Population diagram', fontsize='medium')

        molecule_obj = active_mol
        int_pars = molecule_obj.intensity.get_table
        int_pars.index = range(len(int_pars.index))

        # Parsing the components of the lines in int_pars
        wl = int_pars['lam']
        intens_mod = int_pars['intens']
        Astein_mod = int_pars['a_stein']
        gu = int_pars['g_up']
        eu = int_pars['e_up']

        # Calculating the y-axis for the population diagram for each line in int_pars
        # Use actual molecule parameters instead of hardcoded global constants
        area = np.pi * (molecule_obj.radius * au * 1e2) ** 2  # In cm^2
        Dist = molecule_obj.distance * pc  # Use molecule's actual distance
        beam_s = area / Dist ** 2
        F = intens_mod * beam_s
        freq = ccum / wl
        rd_yax = np.log(4 * np.pi * F / (Astein_mod * hh * freq * gu))
        threshold = np.nanmax(F) / 100

        self.ax3.set_ylim(np.nanmin(rd_yax[F > threshold]), np.nanmax(rd_yax) + 0.5)
        self.ax3.set_xlim(np.nanmin(eu) - 50, np.nanmax(eu[F > threshold]))

        # Populating the population diagram graph with the lines
        self.ax3.scatter(eu, rd_yax, s=0.5, color='#838B8B')
        self.canvas.draw_idle()

    def find_single_lines(self, xmin=None, xmax=None):
        """
        Finds isolated (single) molecular lines in the selected range and stores them for plotting.
        This mimics the logic of the provided single_finder() function.
        """
        # Use current selection if not provided
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                print("No selection made for finding single lines.")
                return

        # Get user-defined separation and intensity threshold
        specsep_val = self.islat.user_settings.get("specsep", specsep)
        line_threshold = self.islat.user_settings.get("line_threshold", 0.03)

        # Get all lines in the selected range
        int_pars_line = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
        int_pars_line.index = range(len(int_pars_line.index))

        lamb_cnts = int_pars_line['lam']
        intensities = int_pars_line['intens']

        # Find max intensity in range
        max_intens = intensities.max() if len(intensities) > 0 else 0
        max_threshold = max_intens * line_threshold

        self.single_lines_list = []
        counter = 0

        for j in int_pars_line.index:
            include = True
            j_lam = lamb_cnts[j]
            sub_xmin = j_lam - specsep_val
            sub_xmax = j_lam + specsep_val
            j_intens = intensities[j]
            loc_threshold = j_intens * 0.1

            if j_intens >= max_threshold:
                # Check all lines within specsep_val of j_lam
                chk_range = int_pars_line[(int_pars_line['lam'] > sub_xmin) & (int_pars_line['lam'] < sub_xmax)]
                range_intens = chk_range['intens']
                for k in chk_range.index:
                    k_intens = range_intens[k]
                    if k_intens >= loc_threshold and k_intens != j_intens:
                        include = False
                        break
                if include:
                    # Store for plotting
                    vline = {"wavelength": j_lam, "ylim": self.ax1.get_ylim()}
                    self.single_lines_list.append(vline)
                    counter += 1

        # Feedback to user via GUI data field if available
        if hasattr(self.islat, "GUI") and hasattr(self.islat.GUI, "data_field"):
            if counter == 0:
                self.islat.GUI.data_field.insert_text('No single lines found in the current wavelength range.')
            else:
                self.islat.GUI.data_field.insert_text(
                    f'There are {counter} single lines found in the current wavelength range.'
                )

        self.canvas.draw_idle()

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
                color='blue'
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

    def _plot_observed_spectrum(self, wave_data, flux_data, visible_mask):
        """Helper method to plot the observed spectrum data."""
        if hasattr(self.islat, "err_data") and self.islat.err_data is not None:
            self.ax1.errorbar(
                wave_data,
                flux_data,
                yerr=self.islat.err_data,
                fmt='-',
                color=self.theme["foreground"],
                ecolor='lightgray',
                elinewidth=0.8,
                capsize=2,
                zorder=5,  # Plot on top
                label="Data",
                alpha=0.8
            )
        else:
            self.ax1.plot(
                wave_data,
                flux_data,
                color=self.theme["foreground"],
                linewidth=1.2,
                zorder=5,  # Plot on top
                label="Data",
                alpha=0.8
            )

    def _plot_model_spectra(self, wave_data, visible_molecules):
        """
        Helper method to plot individual molecule spectra and return summed flux.
        Returns the summed flux on the full wavelength grid.
        """
        summed_flux = np.zeros_like(wave_data)
        
        # Sort molecules by peak intensity for better visual layering
        mol_intensities = []
        for mol in visible_molecules:
            mol_flux_interp = np.interp(
                wave_data,
                mol.spectrum.lamgrid,
                mol.spectrum.flux_jy,
                left=0.0, right=0.0
            )
            peak_intensity = np.max(mol_flux_interp) if len(mol_flux_interp) > 0 else 0
            mol_intensities.append((peak_intensity, mol, mol_flux_interp))
        
        # Sort by peak intensity (lowest first for better visibility)
        mol_intensities.sort(key=lambda x: x[0])
        
        for peak_intensity, mol, mol_flux_interp in mol_intensities:
            if peak_intensity > 0:  # Only plot if there's actual flux
                # Plot molecule spectrum
                self.ax1.plot(
                    wave_data,
                    mol_flux_interp,
                    linestyle='--',
                    color=mol.color,
                    alpha=0.8,
                    linewidth=1.5,
                    label=mol.displaylabel,
                    zorder=3
                )
                
                # Add to summed flux
                summed_flux += mol_flux_interp
        
        return summed_flux

    def _plot_summed_spectrum(self, wave_data, summed_flux):
        """Helper method to plot the summed model spectrum."""
        if np.any(summed_flux > 0):
            # Plot summed spectrum as filled area
            self.ax1.fill_between(
                wave_data,
                0,
                summed_flux,
                color='lightgray',
                alpha=1.0,
                label="Sum",
                zorder=2
            )
            
            # Don't plot additional line for summed spectrum to match original behavior

    def _configure_plot_appearance(self):
        """Helper method to configure plot labels, grid, and styling."""
        self.ax1.set_xlabel('Wavelength (μm)', fontsize=12)
        self.ax1.set_ylabel('Flux density (Jy)', fontsize=12)
        self.ax1.set_title("Spectrum Overview", fontsize=14, pad=20)
        
        # Add grid for better readability
        self.ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Configure legend to appear on top
        legend = self.ax1.legend(
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True,
            ncol=1,
            fontsize=10
        )
        if legend:
            legend.get_frame().set_alpha(0.9)
            legend.set_zorder(10)  # Set z-order after creation
        
        # Set background color if available in theme
        if 'plot_background' in self.theme:
            self.ax1.set_facecolor(self.theme['plot_background'])

    def _optimize_y_limits(self, wave_data, flux_data):
        """Helper method to set optimal y-axis limits based on current view range."""
        if len(flux_data) == 0:
            return
        
        # Get current x-limits to determine visible data
        xlim = self.ax1.get_xlim()
        mask = (wave_data >= xlim[0]) & (wave_data <= xlim[1])
        
        if not np.any(mask):
            # If no data in current view, use full range
            flux_visible = flux_data
        else:
            flux_visible = flux_data[mask]
        
        # Calculate percentiles for robust limit setting
        flux_min = np.nanpercentile(flux_visible, 5)  # 5th percentile
        flux_max = np.nanpercentile(flux_visible, 95)  # 95th percentile
        
        # Add margin
        flux_range = flux_max - flux_min
        margin = flux_range * 0.1 if flux_range > 0 else abs(flux_max) * 0.1
        
        y_min = max(0, flux_min - margin)  # Don't go below zero for flux
        y_max = flux_max + margin
        
        # Ensure reasonable limits
        if y_max <= y_min:
            y_max = y_min + 1e-10
        
        self.ax1.set_ylim(y_min, y_max)

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
        line_flux_meas = np.trapz(flux[integral_range[::-1]], x=ccum / lam[integral_range[::-1]])
        line_flux_meas = -line_flux_meas * 1e-23  # to get (erg s-1 cm-2); it's using frequency array, so need the - in front of it
        
        if err is not None:
            line_err_meas = np.trapz(err[integral_range[::-1]], x=ccum / lam[integral_range[::-1]])
            line_err_meas = -line_err_meas * 1e-23  # to get (erg s-1 cm-2); it's using frequency array, so need the - in front of it
        else:
            line_err_meas = 0.0
            
        return line_flux_meas, line_err_meas
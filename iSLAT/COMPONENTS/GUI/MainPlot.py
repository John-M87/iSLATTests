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

        self.compute_sum_flux()
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
        self.ax1.clear()
        self.match_display_range()
        self.plot_model_lines()
        self.make_span_selector()
        self.compute_sum_flux()
        # plot the data line
        self.ax1.plot(self.islat.wave_data, self.islat.flux_data, color=self.theme["foreground"], label="Data")

        # plot each molecule if it is turned on
        for mol_name, molecule_obj in self.islat.molecules_dict.items():
            if molecule_obj.is_visible:
                #model_flux = molecule_obj.get_flux(self.wave_data)
                self.ax1.fill_between(
                    molecule_obj.spectrum.lamgrid,
                    molecule_obj.spectrum.flux_jy,
                    color=molecule_obj.color, 
                    alpha=0.3, 
                    lw=0
                )

        self.ax1.legend()
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
        line, = self.ax1.plot(molecule_obj.spectrum.lamgrid, model_flux, linestyle='-', color=color, alpha=0.7, label=f"{mol_name}")

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

    def compute_sum_flux(self):
        """
        Computes the sum of all model fluxes and updates the sum line.
        """
        summed_flux = np.zeros_like(self.islat.wave_data)
        for mol in self.islat.molecules_dict.values():
            if mol.is_visible:
                mol_flux = mol.get_flux(self.islat.wave_data)
                summed_flux += mol_flux
        self.summed_flux = summed_flux

    def plot_sum_line(self, wave, flux, label=None, color=None, compute = True):
        """
        Plots the sum line on the main plot.
        """
        if compute:
            flux = self.compute_sum_flux()
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

    def plot_spectrum_around_line(self, xmin=None, xmax=None):
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
        self.plot_line_inspection(xmin, xmax, line_data)
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
            # If picked, highlight both line and scatter
            if picked:
                if line is not None:
                    line.set_color('orange')
                if scatter is not None:
                    scatter.set_facecolor('orange')
                # Display line information on the data field
                lam = value.get('lam', None)
                e_up = value.get('e', None)
                a_stein = value.get('a', None)
                g_up = value.get('g', None)
                inten = value.get('inten', None)
                max_up_lev = value.get('up_lev', 'N/A')
                max_low_lev = value.get('low_lev', 'N/A')
                max_lamb_cnts = f"{lam:.4f}" if lam is not None else 'N/A'
                max_einstein = f"{a_stein:.3f}" if a_stein is not None else 'N/A'
                max_e_up = f"{e_up:.0f}" if e_up is not None else 'N/A'
                max_tau = value.get('tau', 'N/A')
                if isinstance(max_tau, float):
                    max_tau = f"{max_tau:.3f}"
                line_flux = [inten] if inten is not None else ['N/A']
                flux_str = f"{line_flux[0]:.3e}" if isinstance(line_flux[0], (float, int)) else 'N/A'

                info_str = (
                    "Selected line:\n"
                    f"Upper level = {max_up_lev}\n"
                    f"Lower level = {max_low_lev}\n"
                    f"Wavelength (μm) = {max_lamb_cnts}\n"
                    f"Einstein-A coeff. (1/s) = {max_einstein}\n"
                    f"Upper level energy (K) = {max_e_up}\n"
                    f"Opacity = {max_tau}\n"
                    f"Flux in sel. range (erg/s/cm2) = {flux_str}\n"
                )
                self.islat.GUI.data_field.insert_text(info_str)

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
        max_intensity = intensities.max()
        values = []
        for lam_val, inten, e, a, g in zip(lam, intensities, e_up, a_stein, g_up):
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
            values.append({'lam': lam_val, 'lineheight': lineheight, 'e': e, 'a': a, 'g': g, 'rd_yax': rd_yax, 'inten': inten})
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

    def highlight_strongest_line(self):
        """
        Highlights the strongest line (by height) in orange by default, others in green.
        """
        strongest = self.find_strongest_line()
        for line, scatter, value in self.active_lines:
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
        if strongest is not None:
            line, scatter, value = strongest
            if line is not None:
                line.set_color('orange')
            if scatter is not None:
                scatter.set_facecolor('orange')
        self.canvas.draw_idle()

    def plot_line_inspection(self, xmin, xmax, line_data):
        data_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        data_region_y = self.islat.flux_data[data_mask]
        max_y = np.nanmax(data_region_y)

        self.ax2.clear()
        self.update_line_inspection_plot(xmin=xmin, xmax=xmax)

        # Clear and repopulate self.active_lines
        self.active_lines.clear()
        values = self.get_active_line_values(line_data, max_y=max_y)
        for v in values:
            vline = self.ax2.vlines(v['lam'], 0, v['lineheight'] if v['lineheight'] is not None else 0,
                                    color='green', linestyle='dashed', picker=True)
            self.ax2.text(v['lam'], v['lineheight'] if v['lineheight'] is not None else 0,
                          f"{v['e']:.0f},{v['a']:.3f}", fontsize='x-small', color='green', rotation=45)
            # Add placeholder for scatter, will be filled in plot_population_diagram
            self.active_lines.append([vline, None, v])

        self.ax2.set_xlim(xmin, xmax)
        self.ax2.set_ylim(0, max_y * 1.1)
        self.canvas.draw_idle()

        # Highlight the strongest line by default
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
        self.ax3.clear()

        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None

        if xmin is None or xmax is None or (xmax - xmin) < 0.0001:
            self.canvas.draw_idle()
            return

        # Plot data in zoom
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.ax2.plot(self.islat.wave_data[mask], self.islat.flux_data[mask], color="black", label="Observed")

        # plot the active molecule in the line inspection plot
        active_molecule = self.islat.active_molecule
        if active_molecule is not None:
            wavegrid = active_molecule.spectrum.lamgrid
            mol_mask = (wavegrid >= xmin) & (wavegrid <= xmax)
            data = wavegrid[mol_mask]
            flux = active_molecule.spectrum.flux_jy[mol_mask]
            self.ax2.plot(data, flux, color=active_molecule.color, linestyle="--", label=active_molecule.name)
            #max_y = np.nanmax([np.nanmax(self.flux_data[mask]), np.nanmax(flux)])
            max_y = np.nanmax(self.islat.flux_data[mask])
        else:
            max_y = np.nanmax(self.islat.flux_data[mask])

        # Plot the fit line using the compute_fit_line function
        if self.fit_result is not None:
            gauss_fit, fit_results, x_fit = self.fit_result
            if gauss_fit is not None:
                # Plot the total fit line
                self.ax2.plot(x_fit, gauss_fit.eval(x=x_fit), color="red", linestyle='-', label="Total Fit Line")
                max_y = max(max_y, np.nanmax(gauss_fit.eval(x=x_fit)))

                # Plot individual component lines if it's a deblended fit
                if len(fit_results) > 1:  # Check if multiple components exist
                    for i, fit_result in enumerate(fit_results):
                        prefix = f'g{i+1}_'
                        component_flux = gauss_fit.eval_components(x=x_fit)[prefix]
                        self.ax2.plot(x_fit, component_flux, linestyle='--', label=f"Component {i+1}")

            self.fit_result = None

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
        self.ax3.set_title(f'{self.islat.active_molecule.displaylabel} Population diagram', fontsize='medium')

        molecule_obj = self.islat.active_molecule
        #molecule_obj = self.islat.molecules_dict["H2O"]
        int_pars = molecule_obj.intensity.get_table
        int_pars.index = range(len(int_pars.index))

        # Parsing the components of the lines in int_pars
        wl = int_pars['lam']
        intens_mod = int_pars['intens']
        Astein_mod = int_pars['a_stein']
        gu = int_pars['g_up']
        eu = int_pars['e_up']

        # Calculating the y-axis for the population diagram for each line in int_pars
        area = np.pi * (molecule_obj.radius * au * 1e2) ** 2  # In cm^2
        Dist = dist * pc
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
        Finds single lines in the selected range and updates the line inspection plot.
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                print("No selection made for finding single lines.")
                return

        # Resetting the text feed box
        #self.islat.data_field.delete('1.0', "end")

        # Getting all the lines in the range of xmin and xmax
        #int_pars_line = self.islat.active_molecule.intensity.get_table
        #int_pars_line = int_pars_line[(int_pars_line['lam'] > xmin) & (int_pars_line['lam'] < xmax)]
        
        int_pars_line = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
        int_pars_line.index = range(len(int_pars_line.index))

        # Parsing the wavelengths and intensities of the lines in int_pars_line
        lam = int_pars_line['lam']
        intensities = int_pars_line['intens']

        # Determining a max threshold for lines we may want to consider
        max_intens = intensities.max()
        #max_threshold = max_intens * self.islat.line_threshold  # Filter out weak lines
        max_threshold = max_intens * self.islat.user_settings.get("line_threshold", 0.03)

        #specsep = self.islat.specsep
        #counter = 0
        self.single_lines_list = []  # List to store single lines found

        # Main function to find single lines
        for j in int_pars_line.index:
            include = True  # Boolean for determining if the line is single
            j_lam = lam[int_pars_line.index[j]]  # Wavelength of line of interest
            sub_xmin = j_lam - specsep
            sub_xmax = j_lam + specsep
            j_intens = intensities[int_pars_line.index[j]]  # Intensity of line of interest
            loc_threshold = j_intens * 0.1  # Local threshold for determining if the line is single

            if j_intens >= max_threshold:  # Filter out weak lines
                chk_range = int_pars_line[(int_pars_line['lam'] > sub_xmin) & (int_pars_line['lam'] < sub_xmax)]
                range_intens = chk_range['intens']  # Intensities of lines within specsep range

                for k in range(len(range_intens)):
                    k_intens = range_intens.iloc[k]
                    if k_intens >= loc_threshold and k_intens != j_intens:  # Exclude non-single lines
                        include = False

            if include:  # If the line is single, store it as a vline object
                vline = {"wavelength": j_lam, "ylim": self.ax1.get_ylim()}
                self.single_lines_list.append(vline)
            
            '''if include:  # If the line is single, plot it
                self.single_lines_list
                #self.ax1.vlines(j_lam, self.ax1.get_ylim()[0], self.ax1.get_ylim()[1], linestyles='dashed', color='blue')
                #counter += 1'''

        '''# Print the number of isolated lines found in the range
        if counter == 0:
            print("No single lines found in the current wavelength range.")
        else:
            print(f"There are {counter} single lines found in the current wavelength range.")'''

        self.canvas.draw_idle()

    def plot_single_lines(self):
        """
        Plots single lines on the main plot.
        This function is called after finding single lines.
        """
        self.update_model_plot()
        if not hasattr(self, 'single_lines_list') or not self.single_lines_list:
            print("No single lines to plot.")
            return
        for vline in self.single_lines_list:
            # Plot each single line as a vertical dashed line
            self.ax1.vlines(vline['wavelength'], vline['ylim'][0], vline['ylim'][1],
                            linestyles='dashed', color='blue')

        self.canvas.draw_idle()

    def toggle_legend(self):
        leg = self.ax1.get_legend()
        if leg is not None:
            vis = not leg.get_visible()
            leg.set_visible(vis)
            self.canvas.draw_idle()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import tkinter as tk

class iSLATPlot:
    def __init__(self, master, wave_data, flux_data, theme, islat):
        self.master = master
        self.wave_data = wave_data
        self.flux_data = flux_data
        self.theme = theme
        self.islat = islat
        self.legend_on = True

        # Create figure and GridSpec
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

        # Full spectrum
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_ylabel("Flux (arb. units)")
        self.ax1.set_xlabel("Wavelength (µm)")

        # Line inspection
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_ylabel("Flux")
        self.ax2.set_xlabel("Wavelength (µm)")

        # Population diagram
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax3.set_ylabel("ln(Nu/gu)")
        self.ax3.set_xlabel("Eu (K)")

        # Plot the initial spectrum
        self.line, = self.ax1.plot(self.wave_data, self.flux_data, color=self.theme["foreground"], label="Spectrum")
        self.ax1.legend()

        # SpanSelector for interactive selection
        self.span = SpanSelector(self.ax1, self.onselect, 'horizontal', useblit=True,
                                 props=dict(alpha=0.5, facecolor='green'))

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas.draw()

    def onselect(self, xmin, xmax):
        # Perform fit
        print(f"Selected range: {xmin:.3f}-{xmax:.3f}")
        self.islat.xp1, self.islat.xp2 = xmin, xmax
        self.islat.fit_selected_line(xmin, xmax)

        # Update line inspection
        x_fit = self.islat.selected_x
        y_fit = self.islat.selected_y
        y_model = self.islat.y_model

        self.ax2.clear()
        self.ax2.plot(x_fit, y_fit, 'b.', label='Data')
        self.ax2.plot(x_fit, y_model, 'r-', label='Fit')
        self.ax2.legend()

        # Update population diagram
        energies, pops = self.islat.generate_population_diagram()
        self.ax3.clear()
        if energies is not None and len(energies) > 0:
            self.ax3.errorbar(energies, pops, fmt='o')
        else:
            print("No points for population diagram.")
        self.ax3.set_ylabel("ln(Nu/gu)")
        self.ax3.set_xlabel("Eu (K)")

        self.canvas.draw()

    def toggle_legend(self):
        self.legend_on = not self.legend_on
        if self.legend_on:
            self.ax1.legend()
        else:
            leg = self.ax1.get_legend()
            if leg:
                leg.remove()
        self.canvas.draw()
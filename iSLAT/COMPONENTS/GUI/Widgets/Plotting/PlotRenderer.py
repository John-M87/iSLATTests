import tkinter as tk
from .MainSpectralPlot import MainSpectralPlot
from .LineInspectionPlot import LineInspectionPlot
from .PopulationDiagram import PopulationDiagram


class PlotRenderer:
    """Bridge class that orchestrates the three plot components"""

    def __init__(self, parent_widget: tk.Widget, theme):
        self.parent_widget = parent_widget
        self.theme = theme

        # Create main container frame for the plot cluster
        self.plot_cluster_frame = tk.Frame(self.parent_widget)
        self.plot_cluster_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create individual subplot containers
        self.subplot_frames = {}
        
        # Main spectrum plot (subplot 1)
        self.subplot_frames['main'] = tk.Frame(self.plot_cluster_frame)
        self.subplot_frames['main'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_plot = MainSpectralPlot(self.subplot_frames['main'], theme)
        
        # Line inspection plot (subplot 2)
        self.subplot_frames['line'] = tk.Frame(self.plot_cluster_frame)
        self.subplot_frames['line'].pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.line_plot = LineInspectionPlot(self.subplot_frames['line'], theme)
        
        # Population diagram (subplot 3)
        self.subplot_frames['population'] = tk.Frame(self.plot_cluster_frame)
        self.subplot_frames['population'].pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.population_plot = PopulationDiagram(self.subplot_frames['population'], theme)
        
        # Store all canvases for easy access and backward compatibility
        self.canvases = [self.main_plot.canvas, self.line_plot.canvas, self.population_plot.canvas]
        self.canvas = self.main_plot.canvas  # Keep backward compatibility
        
        # Backward compatibility properties
        self.ax1 = self.main_plot.axis
        self.ax2 = self.line_plot.axis
        self.ax3 = self.population_plot.axis
        self.canvas1 = self.main_plot.canvas
        self.canvas2 = self.line_plot.canvas
        self.canvas3 = self.population_plot.canvas
        self.fig1 = self.main_plot.figure
        self.fig2 = self.line_plot.figure
        self.fig3 = self.population_plot.figure
        
        # Visual state (delegate to main plot)
        self.model_lines = self.main_plot.model_lines
        self.active_lines = self.main_plot.active_lines
        
    def update_all_plots(self, data):
        """Update all plots with new data"""
        # Clear previous plots
        self.clear_all_plots()
        
        # Render main spectrum plot
        if 'wave_data' in data and 'flux_data' in data:
            self.main_plot.render_spectrum_plot(
                data['wave_data'], 
                data['flux_data'], 
                data.get('molecules', []), 
                summed_flux=data.get('summed_flux'), 
                error_data=data.get('error_data')
            )
        
        # Render line inspection plot if available
        if 'line_wave' in data and 'line_flux' in data:
            self.line_plot.render_line_inspection(data['line_wave'], data['line_flux'], line_label=data.get('line_label'))
        
        # Render population diagram if active molecule is set
        if 'active_molecule' in data:
            self.population_plot.render_population_diagram(data['active_molecule'], wave_range=data.get('wave_range'))
        
        # Update canvas to reflect changes
        self.update_plot_display()

    def clear_all_plots(self):
        """Clear all plots and reset visual state"""
        self.main_plot.clear_plot()
        self.line_plot.clear_plot()
        self.population_plot.clear_plot()
        
    def clear_model_lines(self):
        """Clear only the model spectrum lines from the main plot"""
        self.main_plot.clear_model_lines()
        
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines on the main spectrum"""
        self.main_plot.plot_saved_lines(saved_lines)
    
    def highlight_line_selection(self, xmin, xmax):
        """Highlight a selected wavelength range"""
        self.main_plot.highlight_line_selection(xmin, xmax)
    
    def update_plot_display(self):
        """Update the plot display"""
        self.main_plot.update_display()
        self.line_plot.update_display()
        self.population_plot.update_display()
    
    def force_plot_refresh(self):
        """Force a complete plot refresh"""
        self.main_plot.force_refresh()
        self.line_plot.force_refresh()
        self.population_plot.force_refresh()
    
    def plot_vertical_lines(self, wavelengths, heights=None, colors=None, labels=None):
        """Plot vertical lines at specified wavelengths"""
        self.line_plot.plot_vertical_lines(wavelengths, heights, colors, labels)
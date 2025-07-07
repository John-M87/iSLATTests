import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk


class LineInspectionPlot:
    """Handles the line inspection plot for detailed line analysis"""
    
    def __init__(self, parent_frame, theme):
        self.parent_frame = parent_frame
        self.theme = theme
        
        # Create matplotlib figure and axis
        self.figure = plt.Figure(figsize=(5, 3.5))
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def clear_plot(self):
        """Clear the line inspection plot"""
        self.axis.clear()
        
    def render_line_inspection(self, line_wave, line_flux, line_label=None):
        """Render the line inspection subplot"""
        self.clear_plot()
        
        if line_wave is not None and line_flux is not None:
            # Plot data in selected range
            self.axis.plot(line_wave, line_flux, 
                         color=self.theme.get("foreground", "black"), 
                         linewidth=1, 
                         label="Observed")
            
            self.axis.set_xlabel("Wavelength (μm)")
            self.axis.set_ylabel("Flux (Jy)")
            self.axis.set_title("Line inspection plot")
            
            # Show legend if there are labeled items
            handles, labels = self.axis.get_legend_handles_labels()
            if handles:
                self.axis.legend()
    
    def plot_vertical_lines(self, wavelengths, heights=None, colors=None, labels=None):
        """Plot vertical lines at specified wavelengths"""
        if heights is None:
            # Get current y-limits for line height
            ylim = self.axis.get_ylim()
            height = ylim[1] - ylim[0]
            heights = [height] * len(wavelengths)
        
        if colors is None:
            colors = ['green'] * len(wavelengths)
        
        if labels is None:
            labels = [None] * len(wavelengths)
        
        for i, (wave, height, color, label) in enumerate(zip(wavelengths, heights, colors, labels)):
            # Plot vertical line from bottom to specified height
            self.axis.axvline(wave, color=color, alpha=0.7, linewidth=1, 
                           linestyle='-', picker=True, label=label)
            
            # Add scatter point at top of line for picking
            self.axis.scatter([wave], [height], color=color, s=20, 
                           alpha=0.8, picker=True, zorder=5)
    
    def update_display(self):
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_refresh(self):
        """Force a complete plot refresh"""
        self.canvas.draw()

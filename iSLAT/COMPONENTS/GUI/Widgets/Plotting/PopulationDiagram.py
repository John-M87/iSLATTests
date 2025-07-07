import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
from iSLAT.Constants import (
    ASTRONOMICAL_UNIT_M, 
    PARSEC_CM_ALT, 
    SPEED_OF_LIGHT_MICRONS, 
    PLANCK_CONSTANT, 
    DEFAULT_DISTANCE
)

class PopulationDiagram:
    """Handles the population diagram plot for molecular excitation analysis"""
    
    def __init__(self, parent_frame, theme):
        self.parent_frame = parent_frame
        self.theme = theme
        
        # Create matplotlib figure and axis
        self.figure = plt.Figure(figsize=(10, 3.5))
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def clear_plot(self):
        """Clear the population diagram plot"""
        self.axis.clear()
        
    def render_population_diagram(self, molecule, wave_range=None):
        """Render the population diagram for the active molecule"""
        self.clear_plot()
        
        if molecule is None:
            self.axis.set_title("No molecule selected")
            return
            
        try:
            # Get the intensity table
            int_pars = molecule.intensity.get_table
            int_pars.index = range(len(int_pars.index))

            # Parsing the components of the lines in int_pars
            wl = int_pars['lam']
            intens_mod = int_pars['intens']
            Astein_mod = int_pars['a_stein']
            gu = int_pars['g_up']
            eu = int_pars['e_up']

            # Calculating the y-axis for the population diagram
            area = np.pi * (molecule.radius * ASTRONOMICAL_UNIT_M * 1e2) ** 2  # In cm^2
            Dist = DEFAULT_DISTANCE * PARSEC_CM_ALT
            beam_s = area / Dist ** 2
            F = intens_mod * beam_s
            freq = SPEED_OF_LIGHT_MICRONS / wl
            rd_yax = np.log(4 * np.pi * F / (Astein_mod * PLANCK_CONSTANT * freq * gu))
            threshold = np.nanmax(F) / 100

            # Set limits
            self.axis.set_ylim(np.nanmin(rd_yax[F > threshold]), np.nanmax(rd_yax) + 0.5)
            self.axis.set_xlim(np.nanmin(eu) - 50, np.nanmax(eu[F > threshold]))

            # Populating the population diagram graph with the lines
            self.axis.scatter(eu, rd_yax, s=0.5, color='#838B8B')

            # Set labels
            self.axis.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
            self.axis.set_xlabel(r'$E_{u}$ (K)')
            self.axis.set_title(f'{molecule.displaylabel} Population diagram', fontsize='medium')
            
        except Exception as e:
            print(f"Error rendering population diagram: {e}")
            self.axis.set_title(f"{molecule.displaylabel} - Error in calculation")
    
    def update_display(self):
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_refresh(self):
        """Force a complete plot refresh"""
        self.canvas.draw()

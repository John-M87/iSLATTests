import tkinter as tk
from tkinter import filedialog
#from .MainPlot import iSLATPlot
##from .Data_field import DataField
#from .MoleculeWindow import MoleculeWindow
#from .ControlPanel import ControlPanel
#from .TopOptions import TopOptions
#from .BottomOptions import BottomOptions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from iSLAT.COMPONENTS.GUI.Widgets.Plotting.PlotRenderer import PlotRenderer
from iSLAT.COMPONENTS.GUI.Widgets.BottomOptions import BottomOptions
from iSLAT.COMPONENTS.GUI.Widgets.TopOptions import TopOptions
from iSLAT.COMPONENTS.GUI.Widgets.MoleculeWindow import MoleculeWindow
from iSLAT.COMPONENTS.GUI.Widgets.Data_field import DataField

class iSLATGUI:
    """
    iSLATGUI class to handle the graphical user interface of the iSLAT application.
    This class is responsible for initializing and managing the GUI components.
    """

    def __init__(self, theme):
        self.theme = theme
        '''self.main = tk.Toplevel()
        self.plot_group = PlotRenderer(theme=theme)

        self.windows = {
            "main": self.main,
            "plot_group": self.plot_group,
        }'''

    def create_windows(self):
        self.main = tk.Tk()
        self.plot_group = PlotRenderer(parent_widget=self.main, theme=self.theme)

        self.windows = {
            "main": self.main,
            "plot_group": self.plot_group,
        }
        self.main.title("iSLAT Version 5.00.00")
        self.main.mainloop()

    def open_spectrum_file_selector(self, filetypes=[("All Files", "*.*")], default_path=None, title="Choose Spectrum Data File"):
        # Open a file dialog to select a spectrum file
        file_path = filedialog.askopenfilename(filetypes=filetypes, initialdir=default_path, title=title)
        return file_path

    def update_main_plots(self, data):
        """
        Update the main plots with new data.
        This method should be called whenever the data changes.
        """
        if "plot_group" in self.windows:
            self.plot_group.update_all_plots(data)

    def init_gui(self):
        # Initialize GUI components here
        print("Initializing GUI...")

    def start_main_loop(self):
        # Start the main loop of the GUI
        print("Starting main loop...")
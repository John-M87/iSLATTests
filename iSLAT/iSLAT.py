iSLAT_version = 'v5.00.00'
print(' ')
print('Loading iSLAT ' + iSLAT_version + ': Please Wait ...')

#import iSLAT.COMPONENTS.FileHandling as ifh
#from iSLAT.COMPONENTS.GUI import iSLATGUI
#from iSLAT.COMPONENTS.DataTypes import *

#from iSLAT import *
#import iSLAT.COMPONENTS.FileHandling as ifh
#import tkinter as tk
#from iSLAT.COMPONENTS.DataProcessing import DataProcessor as dp
#from iSLAT.COMPONENTS.DataManager import DataManager
from iSLAT.COMPONENTS.DataManager import DataManager
from iSLAT.COMPONENTS.GUI.GUI import iSLATGUI

class iSLAT:
    """
    This is the main class of the application.
    It binds the GUI and the data processing components together through callbacks and function passing.
    """

    def __init__(self, GUI : iSLATGUI = None, data_manager : DataManager = None):
        if data_manager is None:
            data_manager = DataManager()
        if GUI is None:
            GUI = iSLATGUI(theme=data_manager.file_handler.read_user_settings())
        self.GUI = GUI
        self.data_manager = data_manager
        main_directory = self.data_manager.file_handler.get_main_directory()

    def start_data_processing(self, spectrum_data_file_path: str = None):
        """Start the data processing pipeline."""
        if spectrum_data_file_path is not None:
            file_path = spectrum_data_file_path
        else:
            file_path = self.GUI.open_spectrum_file_selector()

        self.data_manager.load_spectrum_file(file_path)
        #self.data_manager.process_loaded_files()
        #self.data_manager.plot_main_cluster(plots = self.GUI.plot_main_cluster)
        #self.GUI.update_data_view(self.data_manager.get_data())
        self.data_manager.process_loaded_files()
        self.GUI.create_windows()
        self.GUI.update_main_plots(self.data_manager.get_data())
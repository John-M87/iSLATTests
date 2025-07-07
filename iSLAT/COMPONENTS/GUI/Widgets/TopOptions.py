import tkinter as tk
from tkinter import ttk, messagebox
#from .GUIFunctions import create_button

class TopOptions:
    def __init__(self, master, islat, theme, data_field=None):
        self.master = master
        self.islat = islat
        self.theme = theme
        self.data_field = data_field
        #self.theme = self.master.theme

        # Create the frame for top options
        self.frame = tk.Frame(master, borderwidth=2, relief="groove")

        # Create buttons for top options
        create_button(self.frame, self.theme, "HITRAN Query", self.hitran_query, 0, 0)
        create_button(self.frame, self.theme, "Spectra Browser", self.spectra_browser, 0, 1)
        create_button(self.frame, self.theme, "Default Molecules", self.default_molecules, 1, 0)
        create_button(self.frame, self.theme, "Add Molecule", self.add_molecule, 1, 1)
        create_button(self.frame, self.theme, "Save Parameters", self.save_parameters, 0, 2)
        create_button(self.frame, self.theme, "Load Parameters", self.load_parameters, 1, 2)
        create_button(self.frame, self.theme, "Export Models", self.export_models, 0, 3)
        create_button(self.frame, self.theme, "Toggle Legend", self.toggle_legend, 1, 3)
    
    def hitran_query(self):
        """
        Open the HITRAN molecule selector window.
        This replicates the functionality from the original iSLAT HITRAN query feature.
        """
        try:
            from iSLAT.COMPONENTS.chart_window import MoleculeSelector
            # Use the root window from the islat class for the MoleculeSelector
            root_window = getattr(self.islat, 'root', self.master)
            MoleculeSelector(root_window, self.data_field)
        except Exception as e:
            print(f"Error opening HITRAN query: {e}")
            if self.data_field:
                self.data_field.insert_text(f"Error opening HITRAN query: {e}", console_print=True)
    
    def spectra_browser(self):
        print("Open spectra browser")

    def default_molecules(self):
        self.islat.load_default_molecules()

    def add_molecule(self):
        self.islat.add_molecule_from_hitran()

    def save_parameters(self):
        """
        Save current molecule parameters to CSV file.
        This replicates the functionality from the original iSLAT save parameters feature.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Sure you want to save? This will overwrite any previous save for this spectrum file."
        )
        if confirmed == "no":
            return
        
        # Get the loaded spectrum name for filename
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        #if spectrum_name == 'default':
        #    spectrum_name = "unknown"
        
        try:
            # Import the save function
            from iSLAT.COMPONENTS.FileHandling import write_molecules_to_csv, write_molecules_list_csv
            
            # Save the current molecule parameters
            saved_file = write_molecules_to_csv(
                self.islat.molecules_dict, 
                loaded_spectrum_name=spectrum_name
            )
            
            # Also save to the general molecules list for session persistence
            write_molecules_list_csv(self.islat.molecules_dict)
            
            if saved_file:
                # Update the data field to show success message
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Molecule parameters saved to: {saved_file}',
                        clear_first=True
                    )
                print(f"Molecule parameters saved successfully to: {saved_file}")
            else:
                print("Failed to save molecule parameters")
                
        except Exception as e:
            print(f"Error saving parameters: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error saving parameters: {str(e)}',
                    clear_first=True
                )

    def load_parameters(self):
        """
        Load molecule parameters from CSV file.
        This replicates the functionality from the original iSLAT load parameters feature.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Sure you want to load parameters? Make sure to save any unsaved changes!"
        )
        if confirmed == "no":
            return
        
        # Get the loaded spectrum name for filename
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        #if spectrum_name == 'default':
        #    spectrum_name = "unknown"
        
        # Check if save file exists
        from iSLAT.COMPONENTS.FileHandling import save_folder_path, molsave_file_name
        import os
        
        spectrum_base_name = os.path.splitext(spectrum_name)[0] if spectrum_name != "unknown" else "default"
        save_file = os.path.join(save_folder_path, f"{spectrum_base_name}-{molsave_file_name}")
        
        if not os.path.exists(save_file):
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    'No save file found for this spectrum.',
                    clear_first=True
                )
            print(f"No save file found at: {save_file}")
            return
        
        try:
            # Show loading message
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    'Loading saved parameters, this may take a moment...',
                    clear_first=True
                )
            
            # Clear existing molecules
            self.islat.molecules_dict.clear()
            
            # Read the saved molecule data
            import csv
            loaded_molecules = []
            
            with open(save_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Create molecule from saved data with correct field mapping
                    mol_data = {
                        'Molecule Name': row.get('Molecule Name', ''),
                        'File Path': row.get('File Path', ''),
                        'Molecule Label': row.get('Molecule Label', ''),
                        'Temp': float(row.get('Temp', 600)),
                        'Rad': float(row.get('Rad', 0.5)),
                        'N_Mol': float(row.get('N_Mol', 1e17)),
                        'Color': row.get('Color', '#FF0000'),
                        'Vis': row.get('Vis', 'True').lower() == 'true',
                        'Dist': float(row.get('Dist', 140)),
                        'StellarRV': float(row.get('StellarRV', 0)),
                        'FWHM': float(row.get('FWHM', 200)),
                        'Broad': float(row.get('Broad', 2.5))
                    }
                    loaded_molecules.append(mol_data)
            
            # Initialize molecules from loaded data
            self.islat.init_molecules(loaded_molecules)
            
            # Update GUI components
            if hasattr(self.islat, 'GUI'):
                if hasattr(self.islat.GUI, 'molecule_table'):
                    self.islat.GUI.molecule_table.update_table()
                if hasattr(self.islat.GUI, 'control_panel'):
                    self.islat.GUI.control_panel.reload_molecule_dropdown()
                if hasattr(self.islat.GUI, 'plot'):
                    self.islat.GUI.plot.update_all_plots()
                if hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Successfully loaded parameters from: {save_file}',
                        clear_first=True
                    )
            
            print(f"Successfully loaded {len(loaded_molecules)} molecules from: {save_file}")
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error loading parameters: {str(e)}',
                    clear_first=True
                )

    def export_models(self):
        print("Export models to file")

    def toggle_legend(self):
        #print("Toggled legend on plot")
        self.islat.GUI.plot.toggle_legend()
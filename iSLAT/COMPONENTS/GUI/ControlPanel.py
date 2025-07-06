import tkinter as tk
from tkinter import ttk

class ControlPanel:
    def __init__(self, master, islat):
        self.master = master
        self.islat = islat

        # Create the control panel frame
        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side="left", fill="y")

        # Create and place entry fields
        self.create_plot_start(0, 0)
        self.create_plot_range(0, 2)
        self.create_wavelength_range(1, 0, 1, 2)
        self.create_global_parameters(2, 0)
        self.create_molecule_dropdown(5, 0)
        
        self.reload_molecule_dropdown()

    def create_global_parameters(self, start_row, start_col):
        """Create entry fields for global parameters that affect all molecules"""
        
        # Create bound entry fields using the same pattern as MoleculeWindow
        def create_bound_global_entry(parent, molecules_dict, attr, label_text, row, col, width=8):
            """Create an Entry widget bound to a molecules_dict global attribute with two-way synchronization."""
            tk.Label(parent, text=label_text).grid(row=row, column=col, padx=5, pady=5)
            
            var = tk.StringVar()
            
            # Initialize with current value
            if hasattr(molecules_dict, attr):
                current_val = getattr(molecules_dict, attr)
                var.set(str(current_val))
            else:
                # Use default values if molecules_dict is not available
                import iSLAT.iSLATDefaultInputParms as default_parms
                if attr == 'global_dist':
                    var.set(str(default_parms.dist))
                elif attr == 'global_star_rv':
                    var.set(str(default_parms.star_rv))
                elif attr == 'global_fwhm':
                    var.set(str(default_parms.fwhm))
                elif attr == 'global_intrinsic_line_width':
                    var.set(str(default_parms.intrinsic_line_width))
            
            entry = tk.Entry(parent, textvariable=var, bg='lightgray', width=width)
            entry.grid(row=row, column=col + 1, padx=5, pady=5)
            
            # Bind changes to update molecules_dict object - exactly like MoleculeWindow
            def on_change(*args):
                val = var.get()
                try:
                    val = float(val)
                    if hasattr(molecules_dict, attr):
                        print(f"ControlPanel: Setting {attr} to {val}")
                        setattr(molecules_dict, attr, val)
                        # The setters in MoleculeDict handle all the updates automatically
                        # Now trigger plot updates like MoleculeWindow does
                        self.update_plots()
                except ValueError:
                    pass  # Ignore invalid input
            
            var.trace_add("write", on_change)
            
            # Also bind Return and FocusOut for immediate updates
            entry.bind("<Return>", lambda e: on_change())
            entry.bind("<FocusOut>", lambda e: on_change())
            
            return entry, var
        
        # Only create global parameter entries if molecules_dict is available
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            molecules_dict = self.islat.molecules_dict
            
            # Distance
            self.distance_entry, self.distance_var = create_bound_global_entry(
                self.frame, molecules_dict, "global_dist", "Distance:", start_row, start_col)
            
            # Stellar RV
            self.star_rv_entry, self.star_rv_var = create_bound_global_entry(
                self.frame, molecules_dict, "global_star_rv", "Stellar RV:", start_row, start_col + 2)
            
            # FWHM
            self.fwhm_entry, self.fwhm_var = create_bound_global_entry(
                self.frame, molecules_dict, "global_fwhm", "FWHM:", start_row + 1, start_col)
            
            # Broadening
            self.broadening_entry, self.broadening_var = create_bound_global_entry(
                self.frame, molecules_dict, "global_intrinsic_line_width", "Broadening:", start_row + 1, start_col + 2)
        else:
            # Fallback labels if molecules_dict is not available
            tk.Label(self.frame, text="Global parameters not available").grid(row=start_row, column=start_col, columnspan=4, padx=5, pady=5)

    def refresh_from_molecules_dict(self):
        """Refresh all fields from the current molecules_dict values"""
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            # Update global parameter fields if they exist
            if hasattr(self, 'distance_var'):
                self.distance_var.set(str(self.islat.molecules_dict.global_dist))
            if hasattr(self, 'star_rv_var'):
                self.star_rv_var.set(str(self.islat.molecules_dict.global_star_rv))
            if hasattr(self, 'fwhm_var'):
                self.fwhm_var.set(str(self.islat.molecules_dict.global_fwhm))
            if hasattr(self, 'broadening_var'):
                self.broadening_var.set(str(self.islat.molecules_dict.global_intrinsic_line_width))
            
            # Update wavelength range fields if they exist
            if hasattr(self, 'min_wavelength_var'):
                self.min_wavelength_var.set(str(self.islat.molecules_dict.global_wavelength_range[0]))
            if hasattr(self, 'max_wavelength_var'):
                self.max_wavelength_var.set(str(self.islat.molecules_dict.global_wavelength_range[1]))
                
            # Reload molecule dropdown
            self.reload_molecule_dropdown()

    def create_molecule_dropdown(self, row, column):
        """Create dropdown for molecule selection"""
        tk.Label(self.frame, text="Molecule:").grid(row=row, column=column, padx=5, pady=5)

        # Initialize dropdown options
        dropdown_options = ["SUM", "ALL"]
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            for mol_name, mol_obj in self.islat.molecules_dict.items():
                display_label = getattr(mol_obj, 'displaylabel', mol_name)
                dropdown_options.insert(-2, display_label)  # Insert before SUM and ALL
            
        self.molecule_var = tk.StringVar(self.frame)
        self.molecule_var.set(dropdown_options[0])  # Default to the first option

        self.dropdown = ttk.Combobox(self.frame, textvariable=self.molecule_var, values=dropdown_options)
        self.dropdown.grid(row=row, column=column + 1, padx=5, pady=5)

        # Bind selection event
        self.dropdown.bind("<<ComboboxSelected>>", self.on_molecule_selected)

    def on_molecule_selected(self, event=None):
        """Handle molecule selection from dropdown"""
        selected_label = self.molecule_var.get()
        
        # Handle special cases
        if selected_label in ["SUM", "ALL"]:
            self.islat.active_molecule = selected_label
            self.update_plots()  # Update plots when molecule selection changes
            return
        
        # Find the molecule name that corresponds to this display label
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            for mol_name, mol_obj in self.islat.molecules_dict.items():
                display_label = getattr(mol_obj, 'displaylabel', mol_name)
                if display_label == selected_label:
                    self.islat.active_molecule = mol_name
                    self.update_plots()  # Update plots when molecule selection changes
                    return
        
        # If no match found, default to SUM
        self.islat.active_molecule = "SUM"
        self.update_plots()  # Update plots even for default selection

    def reload_molecule_dropdown(self):
        """Reload the molecule dropdown when molecules are added/removed"""
        if not hasattr(self, 'dropdown'):
            return
            
        # Build new dropdown options
        dropdown_options = []
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            for mol_name, mol_obj in self.islat.molecules_dict.items():
                display_label = getattr(mol_obj, 'displaylabel', mol_name)
                dropdown_options.append(display_label)
        dropdown_options += ["SUM", "ALL"]
        
        # Update dropdown values
        self.dropdown['values'] = dropdown_options
        
        # Handle case where current selection is no longer valid
        current_value = self.molecule_var.get()
        if current_value not in dropdown_options:
            # Default to first available option
            if dropdown_options:
                self.molecule_var.set(dropdown_options[0])
                if dropdown_options[0] in ["SUM", "ALL"]:
                    self.islat.active_molecule = dropdown_options[0]
                else:
                    # Find the molecule name for this display label
                    for mol_name, mol_obj in self.islat.molecules_dict.items():
                        display_label = getattr(mol_obj, 'displaylabel', mol_name)
                        if display_label == dropdown_options[0]:
                            self.islat.active_molecule = mol_name
                            break

    def create_plot_start(self, row, column):
        """Create plot start entry field"""
        tk.Label(self.frame, text="Plot start:").grid(row=row, column=column, padx=5, pady=5)

        self.plot_start_var = tk.StringVar()
        if hasattr(self.islat, 'display_range'):
            self.plot_start_var.set(str(self.islat.display_range[0]))
        else:
            self.plot_start_var.set("4.5")
            
        self.plot_start_entry = tk.Entry(self.frame, textvariable=self.plot_start_var, bg='lightgray', width=8)
        self.plot_start_entry.grid(row=row, column=column + 1, padx=5, pady=5)

        # Bind change events
        self.plot_start_var.trace_add('write', lambda *args: self.update_display_range())
        self.plot_start_entry.bind("<Return>", lambda e: self.update_display_range())
        self.plot_start_entry.bind("<FocusOut>", lambda e: self.update_display_range())

    def create_plot_range(self, row, column):
        """Create plot range entry field"""
        tk.Label(self.frame, text="Plot range:").grid(row=row, column=column, padx=5, pady=5)

        self.plot_range_var = tk.StringVar()
        if hasattr(self.islat, 'display_range'):
            self.plot_range_var.set(str(self.islat.display_range[1] - self.islat.display_range[0]))
        else:
            self.plot_range_var.set("1.0")
            
        self.plot_range_entry = tk.Entry(self.frame, textvariable=self.plot_range_var, bg='lightgray', width=8)
        self.plot_range_entry.grid(row=row, column=column + 1, padx=5, pady=5)

        # Bind change events
        self.plot_range_var.trace_add('write', lambda *args: self.update_display_range())
        self.plot_range_entry.bind("<Return>", lambda e: self.update_display_range())
        self.plot_range_entry.bind("<FocusOut>", lambda e: self.update_display_range())

    def update_display_range(self):
        """Update the display range when plot start or range changes"""
        try:
            start = float(self.plot_start_var.get())
            range_val = float(self.plot_range_var.get())
            new_range = (start, start + range_val)
            
            if hasattr(self.islat, 'display_range'):
                self.islat.display_range = new_range
                
            # Try multiple ways to find the plot object for display range updates
            plot_obj = None
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
                plot_obj = self.islat.GUI.plot
            elif hasattr(self.islat, 'main_plot'):
                plot_obj = self.islat.main_plot
            elif hasattr(self.islat, 'plot'):
                plot_obj = self.islat.plot
                
            # Update plots when display range changes
            if plot_obj:
                if hasattr(plot_obj, 'match_display_range'):
                    plot_obj.match_display_range()
                # Also trigger a full plot update to ensure everything is refreshed
                if hasattr(plot_obj, 'update_all_plots'):
                    plot_obj.update_all_plots()
        except ValueError:
            pass  # Ignore invalid input

    def create_wavelength_range(self, row_min, col_min, row_max, col_max):
        """Create wavelength range entry fields"""
        
        # Create bound entry fields for wavelength range using the same pattern
        def create_bound_wavelength_entry(parent, molecules_dict, attr_name, label_text, row, col, index, width=8):
            """Create an Entry widget bound to wavelength range with two-way synchronization."""
            tk.Label(parent, text=label_text).grid(row=row, column=col, padx=5, pady=5)
            
            var = tk.StringVar()
            
            # Initialize with current value
            if hasattr(molecules_dict, 'global_wavelength_range'):
                current_val = molecules_dict.global_wavelength_range[index]
                var.set(str(current_val))
            else:
                # Use default values if molecules_dict is not available
                import iSLAT.iSLATDefaultInputParms as default_parms
                var.set(str(default_parms.wavelength_range[index]))
            
            entry = tk.Entry(parent, textvariable=var, bg='lightgray', width=width)
            entry.grid(row=row, column=col + 1, padx=5, pady=5)
            
            # Bind changes to update molecules_dict object
            def on_change(*args):
                try:
                    min_val = float(self.min_wavelength_var.get())
                    max_val = float(self.max_wavelength_var.get())
                    
                    if min_val < max_val and hasattr(molecules_dict, 'global_wavelength_range'):
                        print(f"ControlPanel: Setting wavelength range to ({min_val}, {max_val})")
                        # Set the tuple directly - the setter handles all updates
                        molecules_dict.global_wavelength_range = (min_val, max_val)
                        # Also update islat for backward compatibility
                        self.islat.wavelength_range = (min_val, max_val)
                        # Trigger plot updates
                        self.update_plots()
                except ValueError:
                    pass  # Ignore invalid input
            
            var.trace_add("write", on_change)
            
            # Also bind Return and FocusOut for immediate updates
            entry.bind("<Return>", lambda e: on_change())
            entry.bind("<FocusOut>", lambda e: on_change())
            
            return entry, var
        
        # Only create wavelength range entries if molecules_dict is available
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            molecules_dict = self.islat.molecules_dict
            
            # Min wavelength
            self.min_wavelength_entry, self.min_wavelength_var = create_bound_wavelength_entry(
                self.frame, molecules_dict, "global_wavelength_range", "Min. Wave:", row_min, col_min, 0)
            
            # Max wavelength
            self.max_wavelength_entry, self.max_wavelength_var = create_bound_wavelength_entry(
                self.frame, molecules_dict, "global_wavelength_range", "Max. Wave:", row_max, col_max, 1)
        else:
            # Fallback labels if molecules_dict is not available
            tk.Label(self.frame, text="Min. Wave:").grid(row=row_min, column=col_min, padx=5, pady=5)
            tk.Label(self.frame, text="N/A").grid(row=row_min, column=col_min + 1, padx=5, pady=5)
            tk.Label(self.frame, text="Max. Wave:").grid(row=row_max, column=col_max, padx=5, pady=5)
            tk.Label(self.frame, text="N/A").grid(row=row_max, column=col_max + 1, padx=5, pady=5)

    def refresh_from_molecules_dict(self):
        """Refresh all fields from the current molecules_dict values"""
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            # Update global parameter fields if they exist
            if hasattr(self, 'distance_var'):
                self.distance_var.set(str(self.islat.molecules_dict.global_dist))
            if hasattr(self, 'star_rv_var'):
                self.star_rv_var.set(str(self.islat.molecules_dict.global_star_rv))
            if hasattr(self, 'fwhm_var'):
                self.fwhm_var.set(str(self.islat.molecules_dict.global_fwhm))
            if hasattr(self, 'broadening_var'):
                self.broadening_var.set(str(self.islat.molecules_dict.global_intrinsic_line_width))
            
            # Update wavelength range fields if they exist
            if hasattr(self, 'min_wavelength_var'):
                self.min_wavelength_var.set(str(self.islat.molecules_dict.global_wavelength_range[0]))
            if hasattr(self, 'max_wavelength_var'):
                self.max_wavelength_var.set(str(self.islat.molecules_dict.global_wavelength_range[1]))
                
            # Reload molecule dropdown
            self.reload_molecule_dropdown()

    def update_plots(self):
        """Update model spectrum and all plots, similar to MoleculeWindow.update_lines()"""
        try:
            print("ControlPanel.update_plots() called")
            
            # Update the model spectrum
            if hasattr(self.islat, 'update_model_spectrum'):
                self.islat.update_model_spectrum()
                print("Model spectrum updated")
            
            # Try multiple ways to find the plot object
            plot_obj = None
            
            # First try: GUI.plot (should be the correct way)
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
                plot_obj = self.islat.GUI.plot
                print("Found plot via islat.GUI.plot")
            
            # Second try: main_plot (alternative reference)
            elif hasattr(self.islat, 'main_plot'):
                plot_obj = self.islat.main_plot
                print("Found plot via islat.main_plot")
                
            # Third try: Check if islat has a plot attribute directly
            elif hasattr(self.islat, 'plot'):
                plot_obj = self.islat.plot
                print("Found plot via islat.plot")
            
            # Update all plots if we found a plot object
            if plot_obj and hasattr(plot_obj, 'update_all_plots'):
                plot_obj.update_all_plots()
                print("Plot update_all_plots() called successfully")
            else:
                print("Plot object not found or doesn't have update_all_plots method")
                if plot_obj:
                    print(f"Plot object methods: {[m for m in dir(plot_obj) if not m.startswith('_')]}")
                
        except Exception as e:
            print(f"Error updating plots: {e}")
            import traceback
            traceback.print_exc()
            pass  # Don't crash the GUI on plot update errors
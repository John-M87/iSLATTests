import tkinter as tk
from tkinter import ttk

class ControlPanel:
    def __init__(self, master, islat):
        self.master = master
        self.islat = islat
        self._debounce_after_id = None  # For debouncing updates

        # Create the control panel frame
        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side="left", fill="y")

        # Initialize all UI components
        self._create_all_components()
        
        # Register for global parameter change notifications
        self._register_callbacks()

    def _register_callbacks(self):
        """Register callbacks for global parameter changes"""
        try:
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                # Register callback for when global parameters change
                self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
            else:
                pass  # molecules_dict not available for callback registration
        except Exception as e:
            print(f"ControlPanel: Error registering global parameter callback: {e}")
        
        try:
            # Register callback for individual molecule parameter changes
            from iSLAT.COMPONENTS.Molecule import Molecule
            Molecule.add_molecule_parameter_change_callback(self._on_molecule_parameter_change)
        except Exception as e:
            print(f"ControlPanel: Error registering molecule parameter callback: {e}")

    def _on_molecule_parameter_change(self, molecule_name, parameter_name, old_value, new_value):
        """Handle individual molecule parameter changes"""
        # Update population diagram specifically when molecule parameters change
        self._trigger_population_diagram_update()

    def _on_global_parameter_change(self, parameter_name, old_value, new_value):
        """Handle global parameter changes from MoleculeDict"""
        # Use coordinator for updates
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('model_spectrum')
            self.islat.request_update('plots')
        else:
            # Fallback to direct update
            plot_obj = self._get_plot_object()
            if plot_obj:
                # Update all plots including population diagram
                if hasattr(plot_obj, 'update_all_plots'):
                    plot_obj.update_all_plots()
            # Specifically update population diagram if it exists
            if hasattr(plot_obj, 'update_population_diagram'):
                plot_obj.update_population_diagram()

    def _create_all_components(self):
        """Create all control panel components in order"""
        self._create_display_controls(0, 0)
        self._create_wavelength_controls(1, 0)  
        self._create_global_parameter_controls(2, 0)
        self._create_molecule_selector(5, 0)
        self.reload_molecule_dropdown()

    def _debounced_update(self, callback):
        """Debounce updates to prevent excessive calls"""
        if self._debounce_after_id:
            self.master.after_cancel(self._debounce_after_id)
        self._debounce_after_id = self.master.after(100, callback)

    def _create_simple_entry(self, label_text, initial_value, row, col, on_change_callback, width=8):
        """Create a simple entry field with label and change callback"""
        tk.Label(self.frame, text=label_text).grid(row=row, column=col, padx=5, pady=5)
        
        var = tk.StringVar()
        var.set(str(initial_value))
        
        entry = tk.Entry(self.frame, textvariable=var, bg='lightgray', width=width)
        entry.grid(row=row, column=col + 1, padx=5, pady=5)
        
        def on_change(*args):
            self._debounced_update(lambda: on_change_callback(var.get()))
        
        var.trace_add("write", on_change)
        return entry, var

    def _create_bound_parameter_entry(self, label_text, param_name, row, col, width=8):
        """Create an entry bound to a global parameter in molecules_dict"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return None, None
            
        molecules_dict = self.islat.molecules_dict
        current_value = getattr(molecules_dict, param_name, 0)
        
        def update_parameter(value_str):
            try:
                value = float(value_str)
                old_value = getattr(molecules_dict, param_name)
                if abs(old_value - value) > 1e-10:  # Only update if actually different
                    setattr(molecules_dict, param_name, value)
                    # Explicitly trigger population diagram update for molecule parameters
                    self._trigger_population_diagram_update()
            except (ValueError, AttributeError):
                pass
        
        return self._create_simple_entry(label_text, current_value, row, col, update_parameter, width)

    def _create_display_controls(self, start_row, start_col):
        """Create plot start and range controls for display view"""
        # Plot start
        initial_start = getattr(self.islat, 'display_range', [4.5, 5.5])[0]
        self.plot_start_entry, self.plot_start_var = self._create_simple_entry(
            "Plot start:", initial_start, start_row, start_col, self._update_display_range)
        
        # Plot range  
        display_range = getattr(self.islat, 'display_range', [4.5, 5.5])
        initial_range = display_range[1] - display_range[0]
        self.plot_range_entry, self.plot_range_var = self._create_simple_entry(
            "Plot range:", initial_range, start_row, start_col + 2, self._update_display_range)

    def _create_wavelength_controls(self, start_row, start_col):
        """Create wavelength range controls for model calculation range"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        molecules_dict = self.islat.molecules_dict
        min_wave, max_wave = molecules_dict.global_wavelength_range
        
        self.min_wavelength_entry, self.min_wavelength_var = self._create_simple_entry(
            "Min. Wave:", min_wave, start_row, start_col, self._update_wavelength_range)
        self.max_wavelength_entry, self.max_wavelength_var = self._create_simple_entry(
            "Max. Wave:", max_wave, start_row, start_col + 2, self._update_wavelength_range)

    def _create_global_parameter_controls(self, start_row, start_col):
        """Create global parameter entry fields using MoleculeDict properties"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            tk.Label(self.frame, text="Global parameters not available").grid(
                row=start_row, column=start_col, columnspan=4, padx=5, pady=5)
            return

        # Define parameters with their labels - all info comes from MoleculeDict
        parameters = [
            ("Distance:", "global_dist", start_row, start_col),
            ("Stellar RV:", "global_star_rv", start_row, start_col + 2),
            ("FWHM:", "global_fwhm", start_row + 1, start_col),
            ("Broadening:", "global_intrinsic_line_width", start_row + 1, start_col + 2)
        ]
        
        # Store references for later updates
        self._parameter_entries = {}
        
        for label, param_name, row, col in parameters:
            entry, var = self._create_bound_parameter_entry(label, param_name, row, col)
            if entry and var:
                self._parameter_entries[param_name] = (entry, var)

    def _create_molecule_selector(self, row, column):
        """Create molecule dropdown selector"""
        tk.Label(self.frame, text="Molecule:").grid(row=row, column=column, padx=5, pady=5)

        self.molecule_var = tk.StringVar(self.frame)
        self.dropdown = ttk.Combobox(self.frame, textvariable=self.molecule_var)
        self.dropdown.grid(row=row, column=column + 1, padx=5, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self._on_molecule_selected)

    def _update_display_range(self, value_str=None):
        """Update display range from either start or range change"""
        try:
            start = float(self.plot_start_var.get())
            range_val = float(self.plot_range_var.get())
            self._set_display_range(start, start + range_val)
        except (ValueError, AttributeError):
            pass

    def _set_display_range(self, start, end):
        """Set the display range and update plots"""
        if hasattr(self.islat, 'display_range'):
            self.islat.display_range = (start, end)
        
        plot_obj = self._get_plot_object()
        if plot_obj:
            if hasattr(plot_obj, 'match_display_range'):
                plot_obj.match_display_range()
            if hasattr(plot_obj, 'update_all_plots'):
                plot_obj.update_all_plots()

    def _update_wavelength_range(self, value_str=None):
        """Update wavelength range for model calculations (not display)"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        try:
            min_val = float(self.min_wavelength_var.get())
            max_val = float(self.max_wavelength_var.get())
            
            if min_val < max_val:
                molecules_dict = self.islat.molecules_dict
                molecules_dict.global_wavelength_range = (min_val, max_val)
                # Update the iSLAT wavelength range if it exists
                if hasattr(self.islat, 'wavelength_range'):
                    self.islat.wavelength_range = (min_val, max_val)
        except (ValueError, AttributeError):
            pass

    def _on_molecule_selected(self, event=None):
        """Handle molecule selection - only updates active molecule"""
        selected_label = self.molecule_var.get()
        
        # Handle special cases first
        if selected_label in ["SUM", "ALL"]:
            # For SUM/ALL, set active_molecule to a special string that won't break population diagram
            self.islat._active_molecule = selected_label
            # Update population diagram for special cases
            self._trigger_population_diagram_update()
            return
            
        # Handle regular molecules
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            # Find molecule by display label
            for mol_name, mol_obj in self.islat.molecules_dict.items():
                display_label = getattr(mol_obj, 'displaylabel', mol_name)
                if display_label == selected_label:
                    self.islat.active_molecule = mol_name  # Use the setter for proper handling
                    # Update population diagram for the new molecule
                    self._trigger_population_diagram_update()
                    return
            
        # Default fallback if molecule not found
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            first_mol = next(iter(self.islat.molecules_dict.keys()), None)
            if first_mol:
                self.islat.active_molecule = first_mol
                self._trigger_population_diagram_update()

    def _reload_molecule_dropdown(self):
        """Reload molecule dropdown options"""
        if not hasattr(self, 'dropdown'):
            return
            
        options = ["SUM", "ALL"]
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            molecule_options = [
                getattr(mol_obj, 'displaylabel', mol_name) 
                for mol_name, mol_obj in self.islat.molecules_dict.items()
            ]
            options = molecule_options + options
        
        self.dropdown['values'] = options
        
        # Set default value if current selection is invalid
        current_value = self.molecule_var.get()
        if current_value not in options and options:
            self.molecule_var.set(options[0])
            self._on_molecule_selected()

    def _get_plot_object(self):
        """Get the plot object for updates"""
        for attr_path in ['GUI.plot', 'main_plot', 'plot']:
            obj = self.islat
            for part in attr_path.split('.'):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    obj = None
                    break
            if obj:
                return obj
        return None

    def _trigger_population_diagram_update(self):
        """Trigger an explicit population diagram update"""
        plot_obj = self._get_plot_object()
        if plot_obj and hasattr(plot_obj, 'update_population_diagram'):
            # Always call update_population_diagram - it will handle invalid cases internally
            plot_obj.update_population_diagram()

    def refresh_from_molecules_dict(self):
        """Refresh all fields from current molecules_dict values"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        molecules_dict = self.islat.molecules_dict
        
        # Re-register callbacks in case molecules_dict was created after ControlPanel
        try:
            molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
        except:
            pass  # Callback might already be registered
        
        # Update parameter entries stored in the dictionary
        if hasattr(self, '_parameter_entries'):
            for param_name, (entry, var) in self._parameter_entries.items():
                if hasattr(molecules_dict, param_name):
                    var.set(str(getattr(molecules_dict, param_name)))
        
        # Update wavelength range fields
        if (hasattr(self, 'min_wavelength_var') and hasattr(self, 'max_wavelength_var') 
            and hasattr(molecules_dict, 'global_wavelength_range')):
            min_val, max_val = molecules_dict.global_wavelength_range
            self.min_wavelength_var.set(str(min_val))
            self.max_wavelength_var.set(str(max_val))
        
        self._reload_molecule_dropdown()

    # Public interface methods for backward compatibility
    def reload_molecule_dropdown(self):
        """Public method for reloading molecule dropdown"""
        self._reload_molecule_dropdown()

    def update_plots(self):
        """Public method for triggering plot updates using coordinator"""
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('plots')
        else:
            # Fallback to direct update
            plot_obj = self._get_plot_object()
            if plot_obj and hasattr(plot_obj, 'update_all_plots'):
                plot_obj.update_all_plots()
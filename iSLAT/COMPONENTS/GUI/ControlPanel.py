import tkinter as tk
from tkinter import ttk
from iSLAT.COMPONENTS.Molecule import Molecule

class ControlPanel:
    def __init__(self, master, islat):
        self.master = master
        self.islat = islat

        # Create the control panel frame
        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side="left", fill="y")

        # Register for molecule parameter change notifications
        Molecule.add_molecule_parameter_change_callback(self._on_individual_molecule_parameter_change)
        
        # Register for global parameter change notifications from MoleculeDict
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)

        # Create and place entry fields
        self.create_plot_start(0, 0)
        self.create_plot_range(0, 2)
        self.create_wavelength_range(1, 0, 1, 2)
        
        # These now bind to MoleculeDict global attributes (with safety checks)
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            self.create_global_entry("Distance:", 2, 0, "dist", 
                                     lambda: self.islat.molecules_dict.global_dist,
                                     lambda v: setattr(self.islat.molecules_dict, 'global_dist', v))
            self.create_global_entry("Stellar RV:", 2, 2, "star_rv", 
                                    lambda: self.islat.molecules_dict.global_star_rv,
                                    lambda v: setattr(self.islat.molecules_dict, 'global_star_rv', v))
            self.create_global_entry("FWHM:", 3, 0, "fwhm", 
                                    lambda: self.islat.molecules_dict.global_fwhm,
                                    lambda v: setattr(self.islat.molecules_dict, 'global_fwhm', v))
            self.create_global_entry("Broadening:", 3, 2, "intrinsic_line_width", 
                                    lambda: self.islat.molecules_dict.global_intrinsic_line_width,
                                    lambda v: setattr(self.islat.molecules_dict, 'global_intrinsic_line_width', v))
        else:
            print("Warning: molecules_dict not available during ControlPanel initialization")

        self.create_molecule_dropdown(4, 0)
        
        self.reload_molecule_dropdown()

    def _on_global_parameter_change(self, parameter_name, old_value, new_value):
        """Called when a MoleculeDict global parameter changes"""
        # Avoid infinite loops when updating from our own callbacks
        if hasattr(self, f'_updating_{parameter_name}') and getattr(self, f'_updating_{parameter_name}'):
            return
        if hasattr(self, '_updating_wavelength_range') and self._updating_wavelength_range and parameter_name == 'wavelength_range':
            return
            
        # Update the corresponding entry field if it exists
        if hasattr(self, f"_{parameter_name}_var"):
            var = getattr(self, f"_{parameter_name}_var")
            var.set(str(new_value))
        
        # Handle special case for wavelength range
        if parameter_name == "wavelength_range":
            if hasattr(self, 'min_wavelength_var'):
                self.min_wavelength_var.set(str(new_value[0]))
            if hasattr(self, 'max_wavelength_var'):
                self.max_wavelength_var.set(str(new_value[1]))
            # Also update islat for backward compatibility
            self.islat.wavelength_range = new_value
        
        # Trigger plot updates when parameters change that affect plotting
        if parameter_name in ['dist', 'star_rv', 'fwhm', 'intrinsic_line_width', 'wavelength_range']:
            print(f"ControlPanel: Parameter {parameter_name} changed from {old_value} to {new_value}")
            self.update_plots()

    def _on_individual_molecule_parameter_change(self, molecule_name, parameter_name, old_value, new_value):
        """Called when an individual molecule's parameter changes (e.g., from MoleculeWindow)"""
        print(f"ControlPanel: Molecule {molecule_name} parameter {parameter_name} changed from {old_value} to {new_value}")
        
        # Trigger plot updates when molecule parameters change
        if parameter_name in ['temp', 'radius', 'n_mol']:
            self.update_plots()

    def create_global_entry(self, label_text, row, column, attribute_name, getter_func, setter_func):
        """
        Creates a labeled entry field that binds to MoleculeDict global attributes.
        
        Args:
            label_text (str): The label for the entry.
            row (int): The row in the grid.
            column (int): The column in the grid.
            attribute_name (str): The attribute name for internal reference.
            getter_func (callable): Function to get the current value.
            setter_func (callable): Function to set the new value.
        """
        label = tk.Label(self.frame, text=label_text)
        label.grid(row=row, column=column, padx=5, pady=5)

        # Get initial value
        initial_value = getter_func()

        var = tk.StringVar(value=str(initial_value))
        entry = tk.Entry(self.frame, textvariable=var, bg='lightgray', width=8)
        entry.grid(row=row, column=column + 1, padx=5, pady=5)

        def on_change(*args):
            # Avoid infinite loops when updating from parameter change callbacks
            if hasattr(self, f'_updating_{attribute_name}') and getattr(self, f'_updating_{attribute_name}'):
                return
                
            value = var.get()
            try:
                setattr(self, f'_updating_{attribute_name}', True)
                value_to_set = float(value)
                old_value = getter_func()
                setter_func(value_to_set)
                print(f"ControlPanel: Updated {attribute_name} from {old_value} to {value_to_set}")
                
                # Force plot reload for parameters that affect plotting
                if attribute_name in ['dist', 'star_rv', 'fwhm', 'intrinsic_line_width']:
                    self.update_plots()
                    
            except ValueError:
                # If conversion fails, revert to current value
                print(f"ControlPanel: Invalid value '{value}' for {attribute_name}, reverting to {getter_func()}")
                var.set(str(getter_func()))
            except Exception as e:
                # Handle any other errors during setting
                print(f"ControlPanel: Error setting {attribute_name} to {value}: {e}")
                var.set(str(getter_func()))
            finally:
                setattr(self, f'_updating_{attribute_name}', False)

        var.trace_add('write', on_change)

        # Store references for updates from global parameter changes
        setattr(self, f"_{attribute_name}_entry", entry)
        setattr(self, f"_{attribute_name}_var", var)

    def __del__(self):
        """Cleanup when control panel is destroyed"""
        try:
            Molecule.remove_molecule_parameter_change_callback(self._on_individual_molecule_parameter_change)
            if hasattr(self.islat, 'molecules_dict'):
                self.islat.molecules_dict.remove_global_parameter_change_callback(self._on_global_parameter_change)
        except:
            pass  # Ignore errors during cleanup

    def create_molecule_dropdown(self, row, column):
        label = tk.Label(self.frame, text="Molecule:")
        label.grid(row=row, column=column, padx=5, pady=5)

        # Initialize dropdown options with safety check
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            dropdown_options = list(self.islat.molecules_dict.keys()) + ["SUM", "ALL"]
        else:
            dropdown_options = ["SUM", "ALL"]
            
        self.molecule_var = tk.StringVar(self.frame)
        self.molecule_var.set(dropdown_options[0])  # Default to the first option

        self.dropdown = ttk.Combobox(self.frame, textvariable=self.molecule_var, values=dropdown_options)
        self.dropdown.grid(row=row, column=column + 1, padx=5, pady=5)

        # Update self.islat.active_molecule when a new molecule is selected
        self.dropdown.bind("<<ComboboxSelected>>", self.on_molecule_selected)

    def on_molecule_selected(self, event=None):
        """Handle molecule selection from dropdown, mapping display label to molecule name."""
        selected_label = self.molecule_var.get()
        
        # Handle special cases
        if selected_label in ["SUM", "ALL"]:
            old_active = getattr(self.islat, 'active_molecule', None)
            setattr(self.islat, 'active_molecule', selected_label)
            if old_active != selected_label:
                print(f"ControlPanel: Active molecule changed to {selected_label}")
                self.update_plots()
            return
        
        # Check if molecules_dict is available
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        
        # Find the molecule name that corresponds to this display label
        for mol_name, mol_obj in self.islat.molecules_dict.items():
            display_label = getattr(mol_obj, 'displaylabel', mol_name)
            if display_label == selected_label:
                old_active = getattr(self.islat, 'active_molecule', None)
                setattr(self.islat, 'active_molecule', mol_name)
                if old_active != mol_name:
                    print(f"ControlPanel: Active molecule changed from {old_active} to {mol_name}")
                    self.update_plots()
                return
        
        # If no match found, default to the first molecule or SUM
        if len(self.islat.molecules_dict) > 0:
            first_mol_name = list(self.islat.molecules_dict.keys())[0]
            old_active = getattr(self.islat, 'active_molecule', None)
            setattr(self.islat, 'active_molecule', first_mol_name)
            if old_active != first_mol_name:
                print(f"ControlPanel: Active molecule defaulted to {first_mol_name}")
                self.update_plots()
        else:
            old_active = getattr(self.islat, 'active_molecule', None)
            setattr(self.islat, 'active_molecule', "SUM")
            if old_active != "SUM":
                print(f"ControlPanel: Active molecule defaulted to SUM")
                self.update_plots()

    def reload_molecule_dropdown(self):
        # Check if molecules_dict is available
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        # Use display labels instead of molecule names
        dropdown_options = []
        for mol_name, mol_obj in self.islat.molecules_dict.items():
            display_label = getattr(mol_obj, 'displaylabel', mol_name)
            dropdown_options.append(display_label)
        dropdown_options += ["SUM", "ALL"]
        
        if hasattr(self, 'dropdown') and self.dropdown:
            self.dropdown['values'] = dropdown_options
            
            # Handle case where current selection is no longer valid
            current_value = self.molecule_var.get()
            if current_value not in dropdown_options:
                # If there are molecules available, default to the first one
                if len(self.islat.molecules_dict) > 0:
                    first_mol_name = list(self.islat.molecules_dict.keys())[0]
                    first_mol = self.islat.molecules_dict[first_mol_name]
                    first_display_label = getattr(first_mol, 'displaylabel', first_mol_name)
                    self.molecule_var.set(first_display_label)
                    setattr(self.islat, 'active_molecule', first_mol_name)  # Use molecule name, not display label
                # If no molecules, default to "SUM"
                else:
                    self.molecule_var.set("SUM")
                    setattr(self.islat, 'active_molecule', "SUM")

    def create_plot_start(self, row, column):
        label = tk.Label(self.frame, text="Plot start:")
        label.grid(row=row, column=column, padx=5, pady=5)

        self.ax1_starting_x_var = tk.DoubleVar(value=self.islat.display_range[0])
        entry = tk.Entry(self.frame, textvariable=self.ax1_starting_x_var, bg='lightgray', width=8)
        entry.grid(row=row, column=column + 1, padx=5, pady=5)

        def on_change(*args):
            try:
                start = float(self.ax1_starting_x_var.get())
                rng = float(self.ax1_range_x_var.get())
                old_range = self.islat.display_range
                self.islat.display_range = (start, start + rng)
                print(f"ControlPanel: Updated display range from {old_range} to {(start, start + rng)}")
                # Update plots when display range changes
                if hasattr(self.islat, 'main_plot') and self.islat.main_plot is not None:
                    if hasattr(self.islat.main_plot, 'update_plot_limits'):
                        self.islat.main_plot.update_plot_limits()
                    elif hasattr(self.islat.main_plot, 'update_all_plots'):
                        self.islat.main_plot.update_all_plots()
            except Exception:
                pass

        self.ax1_starting_x_var.trace_add('write', on_change)

        # Update entry if islat.display_range changes externally
        def update_entry():
            self.ax1_starting_x_var.set(self.islat.display_range[0])
        self.islat.update_plot_start = update_entry

    def create_plot_range(self, row, column):
        label = tk.Label(self.frame, text="Plot range:")
        label.grid(row=row, column=column, padx=5, pady=5)

        self.ax1_range_x_var = tk.DoubleVar(value=self.islat.display_range[1] - self.islat.display_range[0])
        entry = tk.Entry(self.frame, textvariable=self.ax1_range_x_var, bg='lightgray', width=8)
        entry.grid(row=row, column=column + 1, padx=5, pady=5)

        def on_change(*args):
            try:
                start = float(self.ax1_starting_x_var.get())
                rng = float(self.ax1_range_x_var.get())
                old_range = self.islat.display_range
                self.islat.display_range = (start, start + rng)
                print(f"ControlPanel: Updated display range from {old_range} to {(start, start + rng)}")
                # Update plots when display range changes
                if hasattr(self.islat, 'main_plot') and self.islat.main_plot is not None:
                    if hasattr(self.islat.main_plot, 'update_plot_limits'):
                        self.islat.main_plot.update_plot_limits()
                    elif hasattr(self.islat.main_plot, 'update_all_plots'):
                        self.islat.main_plot.update_all_plots()
            except Exception:
                pass

        self.ax1_range_x_var.trace_add('write', on_change)

        # Update entry if islat.display_range changes externally
        def update_entry():
            self.ax1_range_x_var.set(self.islat.display_range[1] - self.islat.display_range[0])
        self.islat.update_plot_range = update_entry

    def update_xp1_rng(self):
        try:
            xp1_value = float(self.ax1_starting_x.get())
            rng_value = float(self.ax1_range_x.get())
            self.islat.display_range = (xp1_value, xp1_value + rng_value)  # Update the display range with both values
            print(f"Updated display_range to start: {xp1_value}, range: {rng_value}")
        except ValueError:
            print("Invalid input for xp1 or rng")

    def create_wavelength_range(self, minrow, mincolumn, maxrow, maxcolumn):
        label = tk.Label(self.frame, text="Min. Wave:")
        label.grid(row=minrow, column=mincolumn, padx=5, pady=5)
        
        # Initialize with default values or from molecules_dict if available
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            min_val = self.islat.molecules_dict.global_wavelength_range[0]
            max_val = self.islat.molecules_dict.global_wavelength_range[1]
        else:
            # Use default values if molecules_dict is not available
            import iSLAT.iSLATDefaultInputParms as default_parms
            min_val = default_parms.wavelength_range[0]
            max_val = default_parms.wavelength_range[1]
            
        self.min_wavelength_var = tk.StringVar(value=str(min_val))
        self.min_wavelength = tk.Entry(self.frame, textvariable=self.min_wavelength_var, bg='lightgray', width=8)
        self.min_wavelength.grid(row=minrow, column=mincolumn + 1, padx=5, pady=5)
        self.min_wavelength_var.trace_add('write', self.update_wavelength_range)

        label = tk.Label(self.frame, text="Max. Wave:")
        label.grid(row=maxrow, column=maxcolumn, padx=5, pady=5)
        
        self.max_wavelength_var = tk.StringVar(value=str(max_val))
        self.max_wavelength = tk.Entry(self.frame, textvariable=self.max_wavelength_var, bg='lightgray', width=8)
        self.max_wavelength.grid(row=maxrow, column=maxcolumn + 1, padx=5, pady=5)
        self.max_wavelength_var.trace_add('write', self.update_wavelength_range)
        
        # Store variables for wavelength range updates
        self._wavelength_range_min_var = self.min_wavelength_var
        self._wavelength_range_max_var = self.max_wavelength_var

    def update_wavelength_range(self, *args):
        # Avoid infinite loop by temporarily disabling callbacks
        if hasattr(self, '_updating_wavelength_range') and self._updating_wavelength_range:
            return
            
        # Check if molecules_dict is available
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            print("Warning: molecules_dict not available for wavelength range update")
            return
            
        try:
            self._updating_wavelength_range = True
            min_wave = float(self.min_wavelength_var.get())
            max_wave = float(self.max_wavelength_var.get())
            if min_wave < max_wave:
                old_range = self.islat.molecules_dict.global_wavelength_range
                self.islat.molecules_dict.global_wavelength_range = (min_wave, max_wave)
                # Also update islat for backward compatibility
                self.islat.wavelength_range = (min_wave, max_wave)
                print(f"ControlPanel: Updated wavelength range from {old_range} to {(min_wave, max_wave)}")
                # Force plot reload after wavelength range change
                self.update_plots()
            else:
                print("ControlPanel: Min wavelength must be less than max wavelength")
        except ValueError:
            # If conversion fails, revert to current values
            current_range = self.islat.molecules_dict.global_wavelength_range
            print(f"ControlPanel: Invalid wavelength values, reverting to {current_range}")
            self.min_wavelength_var.set(str(current_range[0]))
            self.max_wavelength_var.set(str(current_range[1]))
        except Exception as e:
            print(f"ControlPanel: Error updating wavelength range: {e}")
        finally:
            self._updating_wavelength_range = False

    def initialize_global_entries(self):
        """Initialize global parameter entries if molecules_dict becomes available after __init__"""
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            # Check if entries already exist
            if not hasattr(self, '_dist_entry'):
                self.create_global_entry("Distance:", 2, 0, "dist", 
                                         lambda: self.islat.molecules_dict.global_dist,
                                         lambda v: setattr(self.islat.molecules_dict, 'global_dist', v))
            if not hasattr(self, '_star_rv_entry'):
                self.create_global_entry("Stellar RV:", 2, 2, "star_rv", 
                                        lambda: self.islat.molecules_dict.global_star_rv,
                                        lambda v: setattr(self.islat.molecules_dict, 'global_star_rv', v))
            if not hasattr(self, '_fwhm_entry'):
                self.create_global_entry("FWHM:", 3, 0, "fwhm", 
                                        lambda: self.islat.molecules_dict.global_fwhm,
                                        lambda v: setattr(self.islat.molecules_dict, 'global_fwhm', v))
            if not hasattr(self, '_intrinsic_line_width_entry'):
                self.create_global_entry("Broadening:", 3, 2, "intrinsic_line_width", 
                                        lambda: self.islat.molecules_dict.global_intrinsic_line_width,
                                        lambda v: setattr(self.islat.molecules_dict, 'global_intrinsic_line_width', v))
                                        
            # Register for global parameter change notifications if not already done
            if hasattr(self.islat.molecules_dict, 'add_global_parameter_change_callback'):
                self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)

    def update_plots(self):
        """Update all plots after parameter changes - ensures plots are properly reloaded"""
        try:
            # First update the model spectrum (recalculates all molecule data)
            if hasattr(self.islat, 'update_model_spectrum'):
                self.islat.update_model_spectrum()
                
            # Then update all plots
            if hasattr(self.islat, 'main_plot') and self.islat.main_plot is not None:
                if hasattr(self.islat.main_plot, 'update_all_plots'):
                    self.islat.main_plot.update_all_plots()
                    print("ControlPanel: Reloaded all plots")
                else:
                    # Fallback to individual plot updates
                    if hasattr(self.islat.main_plot, 'update_model_plot'):
                        self.islat.main_plot.update_model_plot()
                    if hasattr(self.islat.main_plot, 'update_data_plot'):
                        self.islat.main_plot.update_data_plot()
                    if hasattr(self.islat.main_plot, 'update_residual_plot'):
                        self.islat.main_plot.update_residual_plot()
                    print("ControlPanel: Reloaded individual plots")
                    
            # Also update molecule window plots if available
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'molecule_window'):
                if hasattr(self.islat.GUI.molecule_window, 'plot'):
                    if hasattr(self.islat.GUI.molecule_window.plot, 'update_all_plots'):
                        self.islat.GUI.molecule_window.plot.update_all_plots()
                        print("ControlPanel: Reloaded molecule window plots")
                        
        except Exception as e:
            print(f"ControlPanel: Error updating plots: {e}")

    def force_plot_refresh(self):
        """Force a complete refresh of all plots - can be called externally"""
        print("ControlPanel: Forcing complete plot refresh...")
        self.update_plots()
        
        # Also trigger molecule flux updates
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            if hasattr(self.islat.molecules_dict, 'update_molecule_fluxes'):
                # Get current wave data if available
                wave_data = None
                if hasattr(self.islat, 'wave_data'):
                    wave_data = self.islat.wave_data
                elif hasattr(self.islat, 'main_plot') and hasattr(self.islat.main_plot, 'wave_data'):
                    wave_data = self.islat.main_plot.wave_data
                    
                if wave_data is not None:
                    self.islat.molecules_dict.update_molecule_fluxes(wave_data)
                    print("ControlPanel: Updated molecule fluxes")
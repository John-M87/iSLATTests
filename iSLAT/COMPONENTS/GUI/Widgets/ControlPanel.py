import tkinter as tk
from tkinter import ttk
from iSLAT.COMPONENTS.DataTypes.Molecule import Molecule

class ControlPanel:
    # Class-level dictionary defining editable molecule fields
    MOLECULE_FIELDS = {
        'distance': {
            'label': 'Distance:',
            'attribute': 'distance',
            'datatype': float,
            'format': '{:.2f}',
            'default': 140.0,
            'width': 12
        },
        'stellar_rv': {
            'label': 'Stellar RV:',
            'attribute': 'stellar_rv',
            'datatype': float,
            'format': '{:.2f}',
            'default': 0.0,
            'width': 12
        },
        'fwhm': {
            'label': 'FWHM:',
            'attribute': 'fwhm',
            'datatype': float,
            'format': '{:.2f}',
            'default': 0.1,
            'width': 12
        },
        'broad': {
            'label': 'Broadening:',
            'attribute': 'broad',
            'datatype': float,
            'format': '{:.2f}',
            'default': 0.1,
            'width': 12
        }
    }
    
    def __init__(self, master, islat):
        self.master = master
        self.islat = islat
        self._debounce_after_id = None  # For debouncing updates
        
        # Get theme from islat
        self.theme = getattr(islat, 'config', {}).get('theme', {})

        # Create the control panel frame
        self.frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.frame.pack(side="left", fill="y")
        
        # Apply theme to the main frame immediately
        self.frame.configure(
            bg=self.theme.get("background", "#181A1B"),
            highlightbackground=self.theme.get("foreground", "#F0F0F0")
        )

        # Initialize all UI components
        self._create_all_components()
        
        # Register for change notifications using the new callback system
        self._register_callbacks()
        
        # Apply theming after everything is created
        self.frame.after(50, lambda: self.apply_theme(self.theme))

    def _register_callbacks(self):
        """Register callbacks using the new iSLAT callback system"""
        try:
            # Register for active molecule changes using the iSLAT callback system
            if hasattr(self.islat, 'add_active_molecule_change_callback'):
                self.islat.add_active_molecule_change_callback(self._on_active_molecule_change)
            
            # Register for global parameter changes if molecules_dict is available
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
            
            # Register for individual molecule parameter changes
            Molecule.add_molecule_parameter_change_callback(self._on_molecule_parameter_change)
            
        except Exception as e:
            print(f"ControlPanel: Error registering callbacks: {e}")

    def _on_active_molecule_change(self, old_molecule, new_molecule):
        """Handle active molecule changes from the iSLAT callback system"""
        # Update the dropdown selection to match the new active molecule
        if hasattr(self, 'molecule_var') and hasattr(self, 'dropdown'):
            # Get the display label for the new molecule
            if hasattr(new_molecule, 'displaylabel'):
                display_label = new_molecule.displaylabel
            elif hasattr(new_molecule, 'name'):
                display_label = new_molecule.name
            elif isinstance(new_molecule, str):
                display_label = new_molecule  # For "SUM", "ALL", etc.
            else:
                display_label = str(new_molecule)
            
            # Update dropdown without triggering callback
            self.molecule_var.set(display_label)
        
        # Update molecule-specific parameter fields
        self._update_molecule_parameter_fields()
        
        # Trigger population diagram update for the new molecule
        self._trigger_population_diagram_update()

    def _on_molecule_parameter_change(self, molecule_name, parameter_name, old_value, new_value):
        """Handle individual molecule parameter changes"""
        # Use the iSLAT update coordinator for efficient updates
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('model_spectrum')
            self.islat.request_update('plots')
        else:
            # Fallback for direct updates
            self._trigger_population_diagram_update()

    def _on_global_parameter_change(self, parameter_name, old_value, new_value):
        """Handle global parameter changes from MoleculeDict"""
        # Use the iSLAT update coordinator for efficient updates
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('model_spectrum')
            self.islat.request_update('plots')
        else:
            # Fallback to direct update
            plot_obj = self._get_plot_object()
            if plot_obj:
                if hasattr(plot_obj, 'update_all_plots'):
                    plot_obj.update_all_plots()
            self._trigger_population_diagram_update()

    def _create_all_components(self):
        """Create all control panel components in order"""
        self._create_display_controls(0, 0)
        self._create_wavelength_controls(1, 0)  
        # Note: Global parameter controls removed - these are now per-molecule
        self._create_molecule_specific_controls(3, 0)  # Move up since global params removed
        self._create_molecule_selector(9, 0)  # Move down to accommodate more molecule params
        self.reload_molecule_dropdown()

    def _debounced_update(self, callback):
        """Debounce updates to prevent excessive calls using iSLAT's update coordinator"""
        if self._debounce_after_id:
            self.master.after_cancel(self._debounce_after_id)
        
        def execute_update():
            callback()
            # Use iSLAT's update coordinator for plot updates
            if hasattr(self.islat, 'request_update'):
                self.islat.request_update('model_spectrum')
                self.islat.request_update('plots')
        
        self._debounce_after_id = self.master.after(100, execute_update)

    def _create_simple_entry(self, label_text, initial_value, row, col, on_change_callback, width=8):
        """Create a simple entry field with label and change callback"""
        label = tk.Label(self.frame, text=label_text)
        label.grid(row=row, column=col, padx=5, pady=5)
        
        # Apply theme to the label
        label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )
        
        var = tk.StringVar()
        var.set(str(initial_value))
        
        entry = tk.Entry(self.frame, textvariable=var, width=width)
        entry.grid(row=row, column=col + 1, padx=5, pady=5)
        
        # Apply theme to the entry
        entry.configure(
            bg=self.theme.get("background_accent_color", "#23272A"),
            fg=self.theme.get("foreground", "#F0F0F0"),
            insertbackground=self.theme.get("foreground", "#F0F0F0"),
            selectbackground=self.theme.get("selection_color", "#00FF99"),
            selectforeground=self.theme.get("background", "#181A1B")
        )
        
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
                    # The global parameter change callback will handle updates automatically
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
            label = tk.Label(self.frame, text="Global parameters not available")
            label.grid(row=start_row, column=start_col, columnspan=4, padx=5, pady=5)
            # Apply theme to the label
            label.configure(
                bg=self.theme.get("background", "#181A1B"),
                fg=self.theme.get("foreground", "#F0F0F0")
            )
            return

        # Note: Removed global Distance, Stellar RV, FWHM, and Broadening
        # These are now handled as per-molecule parameters
        parameters = []
        
        # Store references for later updates
        self._parameter_entries = {}
        
        for label, param_name, row, col in parameters:
            entry, var = self._create_bound_parameter_entry(label, param_name, row, col)
            if entry and var:
                self._parameter_entries[param_name] = (entry, var)

    def _create_molecule_specific_controls(self, start_row, start_col):
        """Create controls for molecule-specific parameters that update with active molecule"""
        # Add a separator/label for molecule-specific section
        separator_label = tk.Label(self.frame, text="--- Active Molecule Parameters ---", font=('TkDefaultFont', 8, 'italic'))
        separator_label.grid(row=start_row, column=start_col, columnspan=4, padx=5, pady=5)
        separator_label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )
        
        # Store references for later updates
        self._molecule_parameter_entries = {}
        
        # Create fields based on the class-level dictionary
        row_offset = 1
        col_offset = 0
        
        for field_key, field_config in self.MOLECULE_FIELDS.items():
            # Calculate grid position (2 fields per row)
            row = start_row + row_offset + (col_offset // 2)
            col = start_col + (col_offset % 2) * 2
            
            entry, var = self._create_molecule_parameter_entry(
                field_config['label'], 
                field_config['attribute'], 
                row, 
                col, 
                field_config['width']
            )
            
            if entry and var:
                self._molecule_parameter_entries[field_config['attribute']] = (entry, var)
            
            col_offset += 1

    def _create_molecule_parameter_entry(self, label_text, param_name, row, col, width=12):
        """Create an entry bound to the active molecule's parameter"""
        
        def update_active_molecule_parameter(value_str):
            # Skip updates for special cases
            if value_str in ["N/A", ""]:
                return
                
            if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
                return
                
            # Get the active molecule object
            active_mol = None
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                    active_mol = self.islat.molecules_dict[self.islat.active_molecule]
                elif hasattr(self.islat.active_molecule, 'name'):
                    active_mol = self.islat.active_molecule
            
            if not active_mol:
                return
                
            try:
                # Get field configuration for proper data type handling
                field_config = None
                for field_key, config in self.MOLECULE_FIELDS.items():
                    if config['attribute'] == param_name:
                        field_config = config
                        break
                
                # Convert value based on field configuration
                if field_config:
                    value = field_config['datatype'](value_str)
                else:
                    # Fallback for unknown parameters
                    value = float(value_str) if param_name in ["distance", "stellar_rv", "fwhm", "broad"] else str(value_str)
                    
                # Get old value for comparison
                old_value = getattr(active_mol, param_name, None)
                
                # Only update if the value actually changed
                if field_config and field_config['datatype'] == float:
                    # Convert old_value to float for proper comparison
                    try:
                        old_float_value = float(old_value) if old_value is not None else None
                    except (ValueError, TypeError):
                        old_float_value = None
                    
                    if old_float_value is None or abs(old_float_value - value) > 1e-10:
                        setattr(active_mol, param_name, value)
                        # Note: Hash change will automatically trigger plot updates
                else:
                    if old_value != value:
                        setattr(active_mol, param_name, value)
                            
            except (ValueError, AttributeError) as e:
                print(f"Error updating {param_name}: {e}")
        
        # Get initial value from active molecule
        initial_value = self._get_active_molecule_parameter_value(param_name)
        
        return self._create_simple_entry(label_text, initial_value, row, col, update_active_molecule_parameter, width)

    def _get_active_molecule_parameter_value(self, param_name):
        """Get the current value of a parameter from the active molecule"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return ""
        
        # For special cases like SUM/ALL, return N/A
        if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in ["SUM", "ALL"]:
            return "N/A"
            
        # Get the active molecule object
        active_mol = None
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                active_mol = self.islat.molecules_dict[self.islat.active_molecule]
            elif hasattr(self.islat.active_molecule, 'name'):
                active_mol = self.islat.active_molecule
        
        if not active_mol:
            return ""
            
        try:
            value = getattr(active_mol, param_name, "")
            
            # Get field configuration for proper formatting
            field_config = None
            for field_key, config in self.MOLECULE_FIELDS.items():
                if config['attribute'] == param_name:
                    field_config = config
                    break
            
            # Format value based on field configuration
            if field_config and isinstance(value, (int, float)):
                return field_config['format'].format(value)
            # Fallback formatting for backward compatibility
            elif param_name in ["distance", "stellar_rv", "fwhm", "broad"] and isinstance(value, (int, float)):
                return f"{value:.2f}"
            
            return str(value)
        except:
            return ""

    def _create_molecule_selector(self, row, column):
        """Create molecule dropdown selector"""
        label = tk.Label(self.frame, text="Molecule:")
        label.grid(row=row, column=column, padx=5, pady=5)
        
        # Apply theme to the label
        label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )

        self.molecule_var = tk.StringVar(self.frame)
        self.dropdown = ttk.Combobox(self.frame, textvariable=self.molecule_var)
        self.dropdown.grid(row=row, column=column + 1, padx=5, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self._on_molecule_selected)
        
        # Apply theming to the control panel after all components are created
        self.frame.after(10, self._apply_theming)

    def _apply_theming(self):
        """Apply theme to all control panel widgets"""
        # Use the theme from self.theme
        if not self.theme:
            return
            
        # Configure TTK styles for combobox
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Use a theme that supports customization
            
            # Configure combobox
            style.configure("ControlPanel.TCombobox",
                          fieldbackground=self.theme.get("background_accent_color", "#23272A"),
                          background=self.theme.get("background_accent_color", "#23272A"),
                          foreground=self.theme.get("foreground", "#F0F0F0"),
                          bordercolor=self.theme.get("background_accent_color", "#23272A"),
                          selectbackground=self.theme.get("selection_color", "#00FF99"),
                          selectforeground=self.theme.get("background", "#181A1B"))
            
            style.map("ControlPanel.TCombobox",
                     fieldbackground=[('readonly', self.theme.get("background_accent_color", "#23272A"))],
                     selectbackground=[('readonly', self.theme.get("selection_color", "#00FF99"))])
            
            self.dropdown.configure(style="ControlPanel.TCombobox")
            
        except Exception as e:
            print(f"Could not apply TTK theming: {e}")
        
        # Apply theme to regular tk widgets recursively
        self._apply_theme_to_widget(self.frame, self.theme)
    
    def _apply_theme_to_widget(self, widget, theme):
        """Apply theme to tkinter widgets recursively"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'LabelFrame']:
                widget.configure(bg=theme.get("background", "#181A1B"))
                if widget_class == 'LabelFrame':
                    widget.configure(fg=theme.get("foreground", "#F0F0F0"))
            elif widget_class == 'Label':
                widget.configure(
                    bg=theme.get("background", "#181A1B"),
                    fg=theme.get("foreground", "#F0F0F0")
                )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=theme.get("background_accent_color", "#23272A"),
                    fg=theme.get("foreground", "#F0F0F0"),
                    insertbackground=theme.get("foreground", "#F0F0F0"),
                    selectbackground=theme.get("selection_color", "#00FF99"),
                    selectforeground=theme.get("background", "#181A1B")
                )
            elif widget_class == 'Button':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                widget.configure(
                    bg=btn_theme.get("background", theme.get("background_accent_color", "#23272A")),
                    fg=theme.get("foreground", "#F0F0F0"),
                    activebackground=btn_theme.get("active_background", theme.get("selection_color", "#00FF99")),
                    activeforeground=theme.get("foreground", "#F0F0F0")
                )
                
            # Recursively apply to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child, theme)
                
        except tk.TclError:
            pass

    def _update_display_range(self, value_str=None):
        """Update display range from either start or range change"""
        try:
            start = float(self.plot_start_var.get())
            range_val = float(self.plot_range_var.get())
            self._set_display_range(start, start + range_val)
        except (ValueError, AttributeError):
            pass

    def _set_display_range(self, start, end):
        """Set the display range and update plots using iSLAT's update system"""
        if hasattr(self.islat, 'display_range'):
            self.islat.display_range = (start, end)
        
        # Use iSLAT's update coordinator for consistent updates
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('plots')
        else:
            # Fallback to direct update
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
                # The global parameter change callback will handle updates automatically
        except (ValueError, AttributeError):
            pass

    def _update_molecule_parameter(self, value_str=None):
        """Update molecule-specific parameters for the active molecule"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        try:
            # Get the active molecule name
            active_molecule_name = getattr(self.islat, 'active_molecule', None)
            if not active_molecule_name:
                return
            
            molecules_dict = self.islat.molecules_dict
            molecule_obj = molecules_dict.get(active_molecule_name, None)
            if not molecule_obj:
                return
            
            # Update each molecule-specific parameter
            for param_name, (entry, var) in self._molecule_parameter_entries.items():
                if hasattr(molecule_obj, param_name):
                    value = getattr(molecule_obj, param_name)
                    var.set(str(value))
        
        except Exception as e:
            print(f"Error updating molecule parameters: {e}")

    def _on_molecule_selected(self, event=None):
        """Handle molecule selection - uses iSLAT's active_molecule property"""
        selected_label = self.molecule_var.get()
        
        # Use iSLAT's active_molecule setter which will trigger callbacks automatically
        try:
            if selected_label in ["SUM", "ALL"]:
                # For special cases, set directly (iSLAT handles these)
                self.islat.active_molecule = selected_label
                # Clear molecule-specific fields for special cases
                self._clear_molecule_parameter_fields()
            elif hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                # Find molecule by display label
                for mol_name, mol_obj in self.islat.molecules_dict.items():
                    display_label = getattr(mol_obj, 'displaylabel', mol_name)
                    if display_label == selected_label:
                        self.islat.active_molecule = mol_name  # This will trigger callbacks
                        return
                
                # Fallback if molecule not found
                first_mol = next(iter(self.islat.molecules_dict.keys()), None)
                if first_mol:
                    self.islat.active_molecule = first_mol
        except Exception as e:
            print(f"Error setting active molecule: {e}")

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
            # Invalidate cache before updating to ensure re-render
            if hasattr(plot_obj, 'plot_renderer') and hasattr(plot_obj.plot_renderer, 'invalidate_population_diagram_cache'):
                plot_obj.plot_renderer.invalidate_population_diagram_cache()
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
        
        # Note: Global parameter entries removed - these are now per-molecule parameters
        
        # Update wavelength range fields
        if (hasattr(self, 'min_wavelength_var') and hasattr(self, 'max_wavelength_var') 
            and hasattr(molecules_dict, 'global_wavelength_range')):
            min_val, max_val = molecules_dict.global_wavelength_range
            self.min_wavelength_var.set(str(min_val))
            self.max_wavelength_var.set(str(max_val))
        
        # Update molecule-specific parameter fields
        self._update_molecule_parameter_fields()
        
        self._reload_molecule_dropdown()
        
        # Ensure theming is applied after refresh
        self.apply_theme()

    # Public interface methods for backward compatibility
    def reload_molecule_dropdown(self):
        """Public method for reloading molecule dropdown"""
        self._reload_molecule_dropdown()
        # Update molecule parameter fields after reloading
        self._update_molecule_parameter_fields()
        # Ensure theming is applied after reload
        self.apply_theme()

    def update_plots(self):
        """Public method for triggering plot updates using iSLAT's update coordinator"""
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('plots')
        else:
            # Fallback to direct update
            plot_obj = self._get_plot_object()
            if plot_obj and hasattr(plot_obj, 'update_all_plots'):
                plot_obj.update_all_plots()
    
    def cleanup(self):
        """Clean up callbacks when control panel is destroyed"""
        try:
            # Remove callbacks to prevent memory leaks
            if hasattr(self.islat, 'remove_active_molecule_change_callback'):
                self.islat.remove_active_molecule_change_callback(self._on_active_molecule_change)
        except Exception as e:
            print(f"Error during ControlPanel cleanup: {e}")
    
    def apply_theme(self, theme=None):
        """Public method to apply theme to the control panel and all its widgets"""
        if theme:
            self.theme = theme
        
        try:
            # Apply TTK styling for Combobox and other TTK widgets
            style = ttk.Style()
            style.theme_use('clam')
            
            # Configure Combobox styling
            style.configure("TCombobox",
                          fieldbackground=self.theme.get("background_accent_color", "#23272A"),
                          background=self.theme.get("background_accent_color", "#23272A"),
                          foreground=self.theme.get("foreground", "#F0F0F0"),
                          bordercolor=self.theme.get("foreground", "#F0F0F0"),
                          arrowcolor=self.theme.get("foreground", "#F0F0F0"),
                          selectbackground=self.theme.get("selection_color", "#00FF99"),
                          selectforeground=self.theme.get("background", "#181A1B"))
            
            style.map("TCombobox",
                     fieldbackground=[('active', self.theme.get("background_accent_color", "#23272A")),
                                    ('focus', self.theme.get("background_accent_color", "#23272A"))],
                     background=[('active', self.theme.get("background_accent_color", "#23272A")),
                               ('focus', self.theme.get("background_accent_color", "#23272A"))],
                     foreground=[('active', self.theme.get("foreground", "#F0F0F0")),
                               ('focus', self.theme.get("foreground", "#F0F0F0"))])
                               
        except Exception as e:
            print(f"Could not apply TTK theming: {e}")
        
        # Apply theme to regular tk widgets recursively
        self._apply_theme_to_widget(self.frame, self.theme)

    def _update_molecule_parameter_fields(self):
        """Update all molecule-specific parameter fields with values from the active molecule"""
        if not hasattr(self, '_molecule_parameter_entries'):
            return
            
        for param_name, (entry, var) in self._molecule_parameter_entries.items():
            new_value = self._get_active_molecule_parameter_value(param_name)
            # Only update if the value actually changed to avoid triggering callbacks
            current_value = var.get()
            if current_value != new_value:
                var.set(new_value)

    def _clear_molecule_parameter_fields(self):
        """Clear molecule-specific parameter fields for special cases like SUM/ALL"""
        if not hasattr(self, '_molecule_parameter_entries'):
            return
            
        for param_name, (entry, var) in self._molecule_parameter_entries.items():
            var.set("N/A")
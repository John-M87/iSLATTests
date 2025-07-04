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
        #self.create_entry("Plot start:", 0, 0, "ax1_starting_x", self.update_xp1_rng)
        #self.create_entry("Plot range:", 0, 2, "ax1_range_x", self.update_xp1_rng)
        self.create_plot_start(0, 0)
        self.create_plot_range(0, 2)
        self.create_wavelength_range(1, 0, 1, 2)
        
        #self.create_entry("Min. Wave:", 1, 0, "min_wavelength", self.update_initvals)
        #self.create_entry("Max. Wave:", 1, 2, "max_wavelength", self.update_initvals)
        self.create_entry("Distance:", 2, 0, "distance")
        self.create_entry("Stellar RV:", 2, 2, "star_rv")
        self.create_entry("FWHM:", 3, 0, "fwhm")
        self.create_entry("Broadening:", 3, 2, "intrinsic_line_width")

        self.create_molecule_dropdown(4, 0)
        self.reload_molecule_dropdown()

    def create_molecule_dropdown(self, row, column):
        label = tk.Label(self.frame, text="Molecule:")
        label.grid(row=row, column=column, padx=5, pady=5)

        dropdown_options = list(self.islat.molecules_dict.keys()) + ["SUM", "ALL"]
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
            setattr(self.islat, 'active_molecule', selected_label)
            return
        
        # Find the molecule name that corresponds to this display label
        for mol_name, mol_obj in self.islat.molecules_dict.items():
            display_label = getattr(mol_obj, 'displaylabel', mol_name)
            if display_label == selected_label:
                setattr(self.islat, 'active_molecule', mol_name)
                return
        
        # If no match found, default to the first molecule or SUM
        if len(self.islat.molecules_dict) > 0:
            first_mol_name = list(self.islat.molecules_dict.keys())[0]
            setattr(self.islat, 'active_molecule', first_mol_name)
        else:
            setattr(self.islat, 'active_molecule', "SUM")

    def reload_molecule_dropdown(self):
        
        #print(self.islat.molecules_dict.keys())
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

    def create_entry(self, label_text, row, column, attribute_name, callback = None, bind_object=None):
        """
        Creates a labeled entry field in the control panel.

        Args:
            label_text (str): The label for the entry.
            row (int): The row in the grid.
            column (int): The column in the grid.
            attribute_name (str): The attribute name to bind to.
            callback (callable, optional): Function to call on value change.
            bind_object (object, optional): The object whose attribute is bound to the entry.
                                            Defaults to self.islat.active_molecule if not provided.
        """
        label = tk.Label(self.frame, text=label_text)
        label.grid(row=row, column=column, padx=5, pady=5)

        # Use bind_object if provided, else default to self.islat
        target = bind_object if bind_object is not None else self.islat.active_molecule

        # Get initial value from target if it exists, else empty string
        initial_value = getattr(target, attribute_name, "")

        var = tk.StringVar(value=str(initial_value))
        entry = tk.Entry(self.frame, textvariable=var, bg='lightgray', width=8)
        entry.grid(row=row, column=column + 1, padx=5, pady=5)

        def on_change(*args):
            value = var.get()
            try:
                value_to_set = float(value)
            except ValueError:
                value_to_set = value
            setattr(target, attribute_name, value_to_set)
            callback() if callback is not None else None

        var.trace_add('write', on_change)

        # Tie the entry field and variable to a property for possible external access
        setattr(self, f"_{attribute_name}_entry", entry)
        setattr(self, f"_{attribute_name}_var", var)
        setattr(self, attribute_name, property(
            lambda self: self._get_entry_value(attribute_name),
            lambda self, value: self._set_entry_value(attribute_name, value, callback)
        ))

    def _get_entry_value(self, attribute_name):
        entry = getattr(self, f"_{attribute_name}_entry")
        return entry.get()

    def _set_entry_value(self, attribute_name, value, callback):
        entry = getattr(self, f"_{attribute_name}_entry")
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
        callback()

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
                self.islat.display_range = (start, start + rng)
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
                self.islat.display_range = (start, start + rng)
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
        self.min_wavelength = tk.Entry(self.frame, bg='lightgray', width=8)
        self.min_wavelength.grid(row=minrow, column=mincolumn + 1, padx=5, pady=5)
        self.min_wavelength.insert(0, str(self.islat.wavelength_range[0]))
        self.min_wavelength.bind("<Return>", lambda _: self.update_wavelength_range())

        label = tk.Label(self.frame, text="Max. Wave:")
        label.grid(row=maxrow, column=maxcolumn, padx=5, pady=5)
        self.max_wavelength = tk.Entry(self.frame, bg='lightgray', width=8)
        self.max_wavelength.grid(row=maxrow, column=maxcolumn + 1, padx=5, pady=5)
        self.max_wavelength.insert(0, str(self.islat.wavelength_range[1]))
        self.max_wavelength.bind("<Return>", lambda _: self.update_wavelength_range())

    def update_wavelength_range(self):
        try:
            min_wave = float(self.min_wavelength.get())
            max_wave = float(self.max_wavelength.get())
            if min_wave < max_wave:
                self.islat.wavelength_range = (min_wave, max_wave)
                print(f"Updated wavelength range to: {self.islat.wavelength_range}")
            else:
                print("Min wavelength must be less than max wavelength")
        except ValueError:
            print("Invalid input for wavelength range")

    def update_initvals(self):
        # Placeholder for initialization values update logic
        print("Initialization values updated")
import tkinter as tk
from tkinter import ttk, colorchooser

class MoleculeWindow:
    def __init__(self, name, parent_frame, molecule_data, output_plot, config, islat):
        self.parent_frame = parent_frame
        self.name = name
        self.plot = output_plot
        self.config = config
        self.theme = config["theme"]
        self.islat = islat

        self.molecules_dict = self.islat.molecules_dict
        self.molecules = {}

        self.build_table()
        self.update_table()

    def build_table(self):
        self.frame = tk.LabelFrame(self.parent_frame, text="Molecules")
        #self.frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create a canvas and scrollbar for scrolling
        self.canvas = tk.Canvas(self.frame, height=300)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas (cross-platform)
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux(event):
            self.canvas.yview_scroll(-1, "units")
        
        def _on_mousewheel_linux_up(event):
            self.canvas.yview_scroll(1, "units")
            
        # Windows and Mac
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        # Linux
        self.canvas.bind("<Button-4>", _on_mousewheel_linux_up)
        self.canvas.bind("<Button-5>", _on_mousewheel_linux)

        headers = ["Molecule", "Temp", "Radius", "Density", "On", "Color", "Delete"]
        for col, text in enumerate(headers):
            tk.Label(self.scrollable_frame, text=text).grid(row=0, column=col, padx=2, pady=2)

    def update_table(self):
        # Clear existing widgets in the scrollable frame (except headers)
        for widget in self.scrollable_frame.winfo_children()[7:]:  # Skip the 7 header labels
            widget.destroy()
        
        # Clear the molecules dict
        self.molecules.clear()

        # Link Entry widgets and color directly to molecule object attributes
        def make_entry_callback(entry, attr, mol_obj):
            def callback(*args):
                val = entry.get()
                try:
                    if attr == "n_mol":
                        val = float(val)
                    elif attr == "temp":
                        val = float(val)
                    elif attr == "radius":
                        val = float(val)
                    elif attr == "displaylabel":
                        # For label changes, we also need to update any GUI dropdowns
                        old_label = getattr(mol_obj, attr, mol_obj.name)
                        if val != old_label:
                            self.update_control_panel_dropdown()
                    setattr(mol_obj, attr, val)
                    self.update_lines()
                except ValueError:
                    pass  # Ignore invalid input
            return callback

        for i, mol_name in enumerate(self.islat.molecules_dict.keys()):
            mol_obj = self.islat.molecules_dict[mol_name]

            # Make molecule name editable
            name_entry = tk.Entry(self.scrollable_frame, width=8)
            display_label = getattr(mol_obj, 'displaylabel', mol_name)
            name_entry.insert(0, display_label)
            name_entry.grid(row=i+1, column=0, padx=2, pady=1, sticky="w")
            
            # Add callback for name entry
            name_entry.bind("<FocusOut>", lambda e, entry=name_entry, m=mol_obj: make_entry_callback(entry, "displaylabel", m)())
            name_entry.bind("<Return>", lambda e, entry=name_entry, m=mol_obj: make_entry_callback(entry, "displaylabel", m)())

            temp_entry = tk.Entry(self.scrollable_frame, width=6)
            temp_entry.insert(0, f"{mol_obj.temp}")
            temp_entry.grid(row=i+1, column=1, padx=2, pady=1)
            temp_entry.bind("<FocusOut>", lambda e, entry=temp_entry, m=mol_obj: make_entry_callback(entry, "temp", m)())
            temp_entry.bind("<Return>", lambda e, entry=temp_entry, m=mol_obj: make_entry_callback(entry, "temp", m)())

            rad_entry = tk.Entry(self.scrollable_frame, width=6)
            rad_entry.insert(0, f"{mol_obj.radius}")
            rad_entry.grid(row=i+1, column=2, padx=2, pady=1)
            rad_entry.bind("<FocusOut>", lambda e, entry=rad_entry, m=mol_obj: make_entry_callback(entry, "radius", m)())
            rad_entry.bind("<Return>", lambda e, entry=rad_entry, m=mol_obj: make_entry_callback(entry, "radius", m)())

            dens_entry = tk.Entry(self.scrollable_frame, width=6)
            dens_entry.insert(0, f"{mol_obj.n_mol:.1e}")
            dens_entry.grid(row=i+1, column=3, padx=2, pady=1)
            dens_entry.bind("<FocusOut>", lambda e, entry=dens_entry, m=mol_obj: make_entry_callback(entry, "n_mol", m)())
            dens_entry.bind("<Return>", lambda e, entry=dens_entry, m=mol_obj: make_entry_callback(entry, "n_mol", m)())

            on_var = tk.BooleanVar(value=mol_obj.is_visible)
            def on_toggle(var=on_var, m=mol_obj):
                m.is_visible = var.get()
                self.update_lines()
            on_btn = tk.Checkbutton(self.scrollable_frame, variable=on_var, command=on_toggle)
            on_btn.grid(row=i+1, column=4, padx=2, pady=1)

            color = getattr(mol_obj, "color", self.theme["default_molecule_colors"][i % len(self.theme["default_molecule_colors"])])
            def pick_and_set_color(mol_name=mol_name, mol_obj=mol_obj, btn=None):
                color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
                if color_code:
                    mol_obj.color = color_code
                    btn.config(bg=color_code)
                    self.molecules[mol_name]["color"] = color_code
                    self.update_lines()
            color_btn = tk.Button(self.scrollable_frame, bg=color, width=4)
            color_btn.config(command=lambda m=mol_name, mo=mol_obj, b=color_btn: pick_and_set_color(m, mo, b))
            color_btn.grid(row=i+1, column=5, padx=2, pady=1)

            # Add delete button
            delete_btn = tk.Button(self.scrollable_frame, text="X", bg="red", fg="white", width=3,
                                 command=lambda m=mol_name: self.delete_molecule(m))
            delete_btn.grid(row=i+1, column=6, padx=2, pady=1)

            self.molecules[mol_name] = {
                "name_entry": name_entry,
                "temp_entry": temp_entry,
                "rad_entry": rad_entry,
                "dens_entry": dens_entry,
                "on_var": on_var,
                "color": color,
                "color_btn": color_btn,
                "delete_btn": delete_btn
            }

        self.update_lines()

    def update_control_panel_dropdown(self):
        """Update the control panel molecule dropdown when molecule labels change."""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            self.islat.GUI.control_panel.reload_molecule_dropdown()

    def delete_molecule(self, mol_name):
        """Delete a molecule from the GUI and the molecules dictionary."""
        if mol_name in self.islat.molecules_dict:
            # Remove from the main molecules dictionary
            del self.islat.molecules_dict[mol_name]
            
            # Clear any model lines for this molecule from the plot
            self.plot.clear_model_lines()
            
            # Update the control panel molecule dropdown if it exists
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
                self.islat.GUI.control_panel.reload_molecule_dropdown()
            
            # Refresh the table to reflect the changes
            self.update_table()
            
            # Update the plot
            self.islat.update_model_spectrum()
            self.plot.update_all_plots()

    def pick_color(self, mol_name):
        color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
        if color_code:
            self.molecules[mol_name]["color"] = color_code
            self.update_lines()

    def update_lines(self):
        self.plot.clear_model_lines()
        for mol, props in self.molecules.items():
            # Check if molecule still exists in the dictionary (not deleted)
            if mol not in self.islat.molecules_dict:
                continue
                
            if props["on_var"].get():
                temp = float(props["temp_entry"].get())
                rad = float(props["rad_entry"].get())
                dens = float(props["dens_entry"].get())
                color = props["color"]
                # actually update molecule parameters
                m_obj = self.islat.molecules_dict[mol]
                m_obj.temp = temp
                m_obj.radius = rad
                m_obj.n_mol = dens
                m_obj.color = color
                m_obj.is_visible = True
                #self.plot.add_model_line(mol, temp, rad, dens, color)
            elif not props["on_var"].get():
                if mol in self.islat.molecules_dict:  # Check existence before access
                    self.islat.molecules_dict[mol].is_visible = False
        self.islat.update_model_spectrum()
        self.plot.update_all_plots()
        #self.plot.update_model_plot()

        '''# If user has already selected a span, refresh the inspection plot with updated molecules
        if hasattr(self.plot, "current_selection") and self.plot.current_selection:
            xmin, xmax = self.plot.current_selection
            self.plot.update_line_inspection_plot(xmin, xmax)'''
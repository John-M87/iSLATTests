import tkinter as tk
from tkinter import colorchooser

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

        headers = ["Molecule", "Temp", "Radius", "Density", "On", "Color"]
        for col, text in enumerate(headers):
            tk.Label(self.frame, text=text).grid(row=0, column=col)

    def update_table(self):
        for i, mol_name in enumerate(self.molecules_dict.keys()):
            mol_data = self.molecules_dict[mol_name]

            lbl = tk.Label(self.frame, text=mol_name)
            lbl.grid(row=i+1, column=0)

            temp_entry = tk.Entry(self.frame, width=6)
            temp_entry.insert(0, f"{mol_data.temp}")
            temp_entry.grid(row=i+1, column=1)

            rad_entry = tk.Entry(self.frame, width=6)
            rad_entry.insert(0, f"{mol_data.radius}")
            rad_entry.grid(row=i+1, column=2)

            dens_entry = tk.Entry(self.frame, width=6)
            dens_entry.insert(0, f"{mol_data.n_mol:.1e}")
            dens_entry.grid(row=i+1, column=3)

            on_var = tk.BooleanVar(value=mol_data.is_visible)
            on_btn = tk.Checkbutton(self.frame, variable=on_var, command=self.update_lines)
            on_btn.grid(row=i+1, column=4)

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
                        setattr(mol_obj, attr, val)
                        self.update_lines()
                    except ValueError:
                        pass  # Ignore invalid input
                return callback

            mol_obj = self.islat.molecules_dict[mol_name]

            temp_entry = tk.Entry(self.frame, width=6)
            temp_entry.insert(0, f"{mol_obj.temp}")
            temp_entry.grid(row=i+1, column=1)
            temp_entry.bind("<FocusOut>", lambda e, entry=temp_entry, m=mol_obj: make_entry_callback(entry, "temp", m)())
            temp_entry.bind("<Return>", lambda e, entry=temp_entry, m=mol_obj: make_entry_callback(entry, "temp", m)())

            rad_entry = tk.Entry(self.frame, width=6)
            rad_entry.insert(0, f"{mol_obj.radius}")
            rad_entry.grid(row=i+1, column=2)
            rad_entry.bind("<FocusOut>", lambda e, entry=rad_entry, m=mol_obj: make_entry_callback(entry, "radius", m)())
            rad_entry.bind("<Return>", lambda e, entry=rad_entry, m=mol_obj: make_entry_callback(entry, "radius", m)())

            dens_entry = tk.Entry(self.frame, width=6)
            dens_entry.insert(0, f"{mol_obj.n_mol:.1e}")
            dens_entry.grid(row=i+1, column=3)
            dens_entry.bind("<FocusOut>", lambda e, entry=dens_entry, m=mol_obj: make_entry_callback(entry, "n_mol", m)())
            dens_entry.bind("<Return>", lambda e, entry=dens_entry, m=mol_obj: make_entry_callback(entry, "n_mol", m)())

            on_var = tk.BooleanVar(value=mol_obj.is_visible)
            def on_toggle(var=on_var, m=mol_obj):
                m.is_visible = var.get()
                self.update_lines()
            on_btn = tk.Checkbutton(self.frame, variable=on_var, command=on_toggle)
            on_btn.grid(row=i+1, column=4)

            color = getattr(mol_obj, "color", self.theme["default_molecule_colors"][i % len(self.theme["default_molecule_colors"])])
            def pick_and_set_color(mol_name=mol_name, mol_obj=mol_obj, btn=None):
                color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
                if color_code:
                    mol_obj.color = color_code
                    btn.config(bg=color_code)
                    self.molecules[mol_name]["color"] = color_code
                    self.update_lines()
            color_btn = tk.Button(self.frame, bg=color, width=4)
            color_btn.config(command=lambda m=mol_name, mo=mol_obj, b=color_btn: pick_and_set_color(m, mo, b))
            color_btn.grid(row=i+1, column=5)

            self.molecules[mol_name] = {
                "temp_entry": temp_entry,
                "rad_entry": rad_entry,
                "dens_entry": dens_entry,
                "on_var": on_var,
                "color": color,
                "color_btn": color_btn
            }

        self.update_lines()

    def pick_color(self, mol_name):
        color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
        if color_code:
            self.molecules[mol_name]["color"] = color_code
            self.update_lines()

    def update_lines(self):
        self.plot.clear_model_lines()
        for mol, props in self.molecules.items():
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
                m_obj = self.islat.molecules_dict[mol].is_visible = False
        self.islat.update_model_spectrum()
        self.plot.update_all_plots()
        #self.plot.update_model_plot()

        '''# If user has already selected a span, refresh the inspection plot with updated molecules
        if hasattr(self.plot, "current_selection") and self.plot.current_selection:
            xmin, xmax = self.plot.current_selection
            self.plot.update_line_inspection_plot(xmin, xmax)'''
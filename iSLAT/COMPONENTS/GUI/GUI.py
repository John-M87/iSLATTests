import tkinter as tk
from .MainPlot import iSLATPlot
from .Data_field import DataField
from .MoleculeWindow import MoleculeWindow
from .ControlPanel import ControlPanel
from .TopOptions import TopOptions
from .BottomOptions import BottomOptions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI:
    def __init__(self, master, molecule_data, wave_data, flux_data, config, islat_class_ref):
        self.master = master
        self.molecule_data = molecule_data
        self.wave_data = wave_data
        self.flux_data = flux_data
        self.config = config
        self.theme = config["theme"]
        self.islat_class = islat_class_ref
        self._popout_states = {}  # Track popout states for widgets

    def build_left_panel(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Top control buttons
        self.top_options = TopOptions(parent, self.islat_class, theme=self.theme)
        self.top_options.frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self._add_popout_button_to_corner(self.top_options.frame, "Top Options", self.top_options, parent, 0, 0, "grid", {"sticky": "ew", "padx": 5, "pady": 2})

        # Molecule table
        self.molecule_table = MoleculeWindow("Molecule Table", parent, self.molecule_data, self.plot, self.config, self.islat_class)
        self.molecule_table.frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self._add_popout_button_to_corner(self.molecule_table.frame, "Molecule Table", self.molecule_table, parent, 1, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

        # Spectrum file selector
        file_frame = tk.LabelFrame(parent, text="Spectrum File")
        file_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        self.file_label = tk.Label(file_frame, text="Loaded: File")
        self.file_label.pack()
        tk.Button(file_frame, text="Load Spectrum", command=self.islat_class.load_spectrum).pack()
        self._add_popout_button_to_corner(file_frame, "Spectrum File", file_frame, parent, 2, 0, "grid", {"sticky": "ew", "padx": 5, "pady": 5})

        # Control panel for input parameters
        control_panel_frame = tk.LabelFrame(parent, text="Control Panel")
        control_panel_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.control_panel = ControlPanel(control_panel_frame, self.islat_class)
        self._add_popout_button_to_corner(control_panel_frame, "Control Panel", control_panel_frame, parent, 3, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

        # Main data field
        self.data_field = DataField("Main Data Field", "", parent)
        self.data_field.frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self._add_popout_button_to_corner(self.data_field.frame, "Main Data Field", self.data_field.frame, parent, 4, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

    def _add_popout_button_to_corner(self, widget, title, content, parent, row, column, manager, manager_kwargs):
        # Store geometry info for re-adding
        self._popout_states[widget] = {
            "parent": parent,
            "row": row,
            "column": column,
            "manager": manager,
            "manager_kwargs": manager_kwargs,
            "title": title,
        }
        btn = tk.Button(widget, text="⧉", command=lambda: self._popout_window(title, widget), width=2, height=1, relief="flat", padx=0, pady=0)
        btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)

    #def _popout_window(self, title, content, widget):
    def _popout_window(self, title, widget):
        # Remove from grid/pack/place in main window
        state = self._popout_states.get(widget)
        if state and state["manager"] == "grid":
            widget.grid_forget()
        elif state and state["manager"] == "pack":
            widget.pack_forget()
        elif state and state["manager"] == "place":
            widget.place_forget()

        pop = tk.Toplevel(self.window)
        pop.title(title)

        def on_close():
            # Remove from popout and re-add to main window
            widget.pack_forget()
            widget.grid_forget()
            widget.place_forget()
            if state["manager"] == "grid":
                widget.master = state["parent"]
                widget.grid(row=state["row"], column=state["column"], **state["manager_kwargs"])
            elif state["manager"] == "pack":
                widget.master = state["parent"]
                widget.pack(**state["manager_kwargs"])
            elif state["manager"] == "place":
                widget.master = state["parent"]
                widget.place(**state["manager_kwargs"])
            pop.destroy()

        pop.protocol("WM_DELETE_WINDOW", on_close)

        # Add to popout window using the original geometry manager
        widget.pack_forget()
        widget.grid_forget()
        widget.place_forget()
        widget.master = pop
        if state["manager"] == "grid":
            widget.grid(row=0, column=0, sticky="nsew")
            pop.grid_rowconfigure(0, weight=1)
            pop.grid_columnconfigure(0, weight=1)
        elif state["manager"] == "pack":
            widget.pack(fill="both", expand=True)
        elif state["manager"] == "place":
            widget.place(relx=0, rely=0, relwidth=1, relheight=1)
        # Add popout button again in popout
        btn = tk.Button(widget, text="⧉", command=lambda: on_close(), width=2, height=1, relief="flat", padx=0, pady=0)
        btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
        
    def create_window(self):
        self.window = self.master
        self.window.title("iSLAT Version 5.00.00")
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=3)
        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=0)

        # Right side: plots
        right_frame = tk.Frame(self.window)
        right_frame.grid(row=0, column=1, sticky="nsew")
        self.plot = iSLATPlot(right_frame, self.wave_data, self.flux_data, self.theme, self.islat_class)
        if hasattr(self.plot, 'frame'):
            self._add_popout_button_to_corner(self.plot.frame, "Plot", self.plot, right_frame, 0, 0, "pack", {"fill": "both", "expand": True})
            self.plot.frame.pack(fill="both", expand=True)
        elif hasattr(self.plot, 'figure'):
            canvas = FigureCanvasTkAgg(self.plot.figure, master=right_frame)
            canvas_widget = canvas.get_tk_widget()
            self._add_popout_button_to_corner(canvas_widget, "Plot", self.plot, right_frame, 0, 0, "pack", {"fill": "both", "expand": True})
            canvas_widget.pack(fill="both", expand=True)

        # Left side: all controls
        left_frame = tk.Frame(self.window)
        left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.build_left_panel(left_frame)

        # Bottom function buttons
        self.bottom_options = BottomOptions(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.config)
        self.bottom_options.frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._add_popout_button_to_corner(self.bottom_options.frame, "Bottom Options", self.bottom_options, self.window, 1, 0, "grid", {"columnspan": 2, "sticky": "ew"})

    def start(self):
        self.create_window()
        self.window.mainloop()
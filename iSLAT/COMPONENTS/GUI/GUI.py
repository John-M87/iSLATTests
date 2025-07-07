import tkinter as tk
from tkinter import filedialog
#from .MainPlot import iSLATPlot
##from .Data_field import DataField
#from .MoleculeWindow import MoleculeWindow
#from .ControlPanel import ControlPanel
#from .TopOptions import TopOptions
#from .BottomOptions import BottomOptions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from iSLAT.COMPONENTS.GUI.Widgets.Plotting.MainPlot import PlotRenderer
from iSLAT.COMPONENTS.GUI.Widgets.BottomOptions import BottomOptions
from iSLAT.COMPONENTS.GUI.Widgets.TopOptions import TopOptions
from iSLAT.COMPONENTS.GUI.Widgets.MoleculeWindow import MoleculeWindow
from iSLAT.COMPONENTS.GUI.Widgets.Data_field import DataField

class iSLATGUI:
    """
    iSLATGUI class to handle the graphical user interface of the iSLAT application.
    This class is responsible for initializing and managing the GUI components.
    """

    def __init__(self, theme):
        self.theme = theme
        '''self.main = tk.Toplevel()
        self.plot_group = PlotRenderer(theme=theme)

        self.windows = {
            "main": self.main,
            "plot_group": self.plot_group,
        }'''

    def create_windows(self):
        self.main = tk.Toplevel()
        self.plot_group = PlotRenderer(parent_widget=self.main, theme=self.theme)

        self.windows = {
            "main": self.main,
            "plot_group": self.plot_group,
        }
        self.main.title("iSLAT Version 5.00.00")
        self.main.mainloop()

    def open_spectrum_file_selector(self, filetypes=[("All Files", "*.*")], default_path=None, title="Choose Spectrum Data File"):
        # Open a file dialog to select a spectrum file
        file_path = filedialog.askopenfilename(filetypes=filetypes, initialdir=default_path, title=title)
        return file_path

    def update_main_plots(self, data):
        """
        Update the main plots with new data.
        This method should be called whenever the data changes.
        """
        if "plot_group" in self.windows:
            self.plot_group.update_all_plots(data)

    def init_gui(self):
        # Initialize GUI components here
        print("Initializing GUI...")

    def start_main_loop(self):
        # Start the main loop of the GUI
        print("Starting main loop...")

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
        self._popout_windows = {}  # Track active popout windows

    def build_left_panel(self, parent: tk.Frame):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Main data field - create this first so we can pass it to other components
        self.data_field = DataField("Main Data Field", "", parent)
        self.data_field.frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self._add_popout_button_to_corner(self.data_field.frame, "Main Data Field", self.data_field.frame, parent, 4, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

        # Top control buttons - now we can pass data_field
        self.top_options = TopOptions(parent, self.islat_class, theme=self.theme, data_field=self.data_field)
        self.top_options.frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self._add_popout_button_to_corner(self.top_options.frame, "Top Options", self.top_options, parent, 0, 0, "grid", {"sticky": "ew", "padx": 5, "pady": 2})

        # Molecule table
        self.molecule_table = MoleculeWindow("Molecule Table", parent, self.molecule_data, self.plot, self.config, self.islat_class)
        self.molecule_table.frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self._add_popout_button_to_corner(self.molecule_table.frame, "Molecule Table", self.molecule_table, parent, 1, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

        # Spectrum file selector
        file_frame = tk.LabelFrame(parent, text="Spectrum File")
        file_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        
        self.file_label = tk.Label(file_frame, text=default_text)
        self.file_label.pack()
        tk.Button(file_frame, text="Load Spectrum", command=self.islat_class.load_spectrum).pack()
        self._add_popout_button_to_corner(file_frame, "Spectrum File", file_frame, parent, 2, 0, "grid", {"sticky": "ew", "padx": 5, "pady": 5})

        # Control panel for input parameters
        control_panel_frame = tk.LabelFrame(parent, text="Control Panel")
        control_panel_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.control_panel = ControlPanel(control_panel_frame, self.islat_class)
        self._add_popout_button_to_corner(control_panel_frame, "Control Panel", control_panel_frame, parent, 3, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

    def _add_popout_button_to_corner(self, widget, title, content, parent, row, column, manager, manager_kwargs):
        # Store geometry info for re-adding
        widget_id = id(widget)
        self._popout_states[widget_id] = {
            "widget": widget,
            "parent": parent,
            "row": row,
            "column": column,
            "manager": manager,
            "manager_kwargs": manager_kwargs,
            "title": title,
            "content": content,
            "is_popped_out": False
        }
        
        # Add the button after a small delay to ensure widget is fully initialized
        def add_button():
            try:
                btn = tk.Button(widget, text="⧉", command=lambda: self._toggle_popout(widget_id), 
                               width=2, height=1, relief="flat", padx=0, pady=0)
                btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
            except tk.TclError:
                # Widget might not be ready yet, try again later
                widget.after(100, add_button)
        
        widget.after(10, add_button)

    def _toggle_popout(self, widget_id):
        """Toggle between popout and pop-in for a widget."""
        state = self._popout_states.get(widget_id)
        if not state:
            return

        if state["is_popped_out"]:
            self._pop_in_widget(widget_id)
        else:
            self._popout_widget(widget_id)

    def _popout_widget(self, widget_id):
        """Pop out a widget to a separate window."""
        state = self._popout_states.get(widget_id)
        if not state or state["is_popped_out"]:
            return

        widget = state["widget"]
        
        # Remove from main window
        if state["manager"] == "grid":
            widget.grid_forget()
        elif state["manager"] == "pack":
            widget.pack_forget()
        elif state["manager"] == "place":
            widget.place_forget()

        # Create popout window
        # Use self.master instead of self.window to ensure we have a valid parent
        parent_window = getattr(self, 'window', self.master)
        pop = tk.Toplevel(parent_window)
        pop.title(state["title"])
        pop.geometry("600x400")  # Set reasonable default size
        
        # Configure popout window grid
        pop.grid_rowconfigure(0, weight=1)
        pop.grid_columnconfigure(0, weight=1)

        # Reparent the widget to the popout window
        widget.master = pop

        # Place the widget in the popout window
        widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Update the popout button in the widget to show pop-in functionality
        self._update_popout_button(widget, widget_id, is_popped_out=True)

        # Store the popout window reference
        self._popout_windows[widget_id] = pop
        state["is_popped_out"] = True

        # Handle window close event
        def on_close():
            self._pop_in_widget(widget_id)

        pop.protocol("WM_DELETE_WINDOW", on_close)

    def _pop_in_widget(self, widget_id):
        """Pop in a widget back to the main window."""
        state = self._popout_states.get(widget_id)
        if not state or not state["is_popped_out"]:
            return

        widget = state["widget"]
        pop = self._popout_windows.get(widget_id)
        
        if pop:
            try:
                # Remove from popout window
                widget.grid_forget()
                
                # Reparent back to original parent
                widget.master = state["parent"]
                
                # Place back in original location
                if state["manager"] == "grid":
                    widget.grid(row=state["row"], column=state["column"], **state["manager_kwargs"])
                elif state["manager"] == "pack":
                    widget.pack(**state["manager_kwargs"])
                elif state["manager"] == "place":
                    widget.place(**state["manager_kwargs"])
                
                # Update the popout button back to popout functionality
                self._update_popout_button(widget, widget_id, is_popped_out=False)
                
                # Destroy the popout window
                pop.destroy()
                
            except tk.TclError as e:
                print(f"Error during pop-in: {e}")
            finally:
                # Clean up references
                if widget_id in self._popout_windows:
                    del self._popout_windows[widget_id]
                state["is_popped_out"] = False

    def _update_popout_button(self, widget, widget_id, is_popped_out):
        """Update the popout button for the current state."""
        try:
            # Remove existing popout button
            for child in widget.winfo_children():
                if isinstance(child, tk.Button) and child.cget("text") == "⧉":
                    child.destroy()
                    break
            
            # Create new button with appropriate command
            btn = tk.Button(widget, text="⧉", command=lambda: self._toggle_popout(widget_id),
                           width=2, height=1, relief="flat", padx=0, pady=0)
            btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
        except tk.TclError as e:
            print(f"Error updating popout button: {e}")

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
        
        # Handle plot popout - create a container frame for better control
        plot_container = tk.Frame(right_frame)
        plot_container.pack(fill="both", expand=True)
        
        # Check if plot has a frame attribute, otherwise create one
        if hasattr(self.plot, 'frame'):
            plot_widget = self.plot.frame
            plot_widget.master = plot_container
            plot_widget.pack(fill="both", expand=True)
        elif hasattr(self.plot, 'figure'):
            canvas = FigureCanvasTkAgg(self.plot.figure, master=plot_container)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            plot_widget = plot_container
        else:
            # Fallback - use the plot container
            plot_widget = plot_container
            
        self._add_popout_button_to_corner(plot_widget, "Plot", plot_widget, right_frame, 0, 0, "pack", {"fill": "both", "expand": True})

        # Left side: all controls
        left_frame = tk.Frame(self.window)
        left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.build_left_panel(left_frame)

        # Bottom function buttons
        self.bottom_options = BottomOptions(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.config)
        self.bottom_options.frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._add_popout_button_to_corner(self.bottom_options.frame, "Bottom Options", self.bottom_options.frame, self.window, 1, 0, "grid", {"columnspan": 2, "sticky": "ew"})

    def cleanup_popouts(self):
        """Clean up all popout windows when the main window is closed."""
        for widget_id in list(self._popout_windows.keys()):
            self._pop_in_widget(widget_id)

    def start(self):
        self.create_window()
        
        # Set up cleanup on window close
        def on_closing():
            self.cleanup_popouts()
            self.window.destroy()
        
        self.window.protocol("WM_DELETE_WINDOW", on_closing)
        self.window.mainloop()
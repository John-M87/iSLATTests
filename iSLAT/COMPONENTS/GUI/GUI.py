import tkinter as tk
from tkinter import filedialog
import os
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
        if master is None:
            self.master = tk.Tk()
            self.master.title("iSLAT - Infrared Spectral Line Analysis Tool")
            self.master.resizable(True, True)
        else:
            self.master = master
        
        self.molecule_data = molecule_data
        self.wave_data = wave_data
        self.flux_data = flux_data
        self.config = config
        self.theme = config["theme"]
        self.islat_class = islat_class_ref
        self._popout_states = {}  # Track popout states for widgets
        self._popout_windows = {}  # Track active popout windows
        
        # Apply theme to root window
        self._apply_theme_to_widget(self.master)

    def _apply_theme_to_widget(self, widget):
        """Apply theme colors to a tkinter widget and its children."""
        try:
            # Apply theme to the widget itself
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'Toplevel', 'Tk']:
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'Label':
                widget.configure(bg=self.theme["background"], fg=self.theme["foreground"])
            elif widget_class == 'Button':
                # Only apply theme if the button doesn't have custom styling
                if widget.cget('bg') in ['SystemButtonFace', '#d9d9d9', '#ececec']:  # Default button colors
                    btn_theme = self.theme["buttons"].get("DefaultBotton", self.theme["buttons"]["DefaultBotton"])
                    widget.configure(
                        bg=btn_theme["background"],
                        fg=self.theme["foreground"],
                        activebackground=btn_theme["active_background"],
                        activeforeground=self.theme["foreground"]
                    )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Text':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Listbox':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Checkbutton':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["toolbar"]
                )
            elif widget_class == 'Radiobutton':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["toolbar"]
                )
            elif widget_class == 'Scale':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["selection_color"],
                    troughcolor=self.theme["toolbar"]
                )
            elif widget_class == 'Scrollbar':
                widget.configure(
                    bg=self.theme["toolbar"],
                    troughcolor=self.theme["background"],
                    activebackground=self.theme["selection_color"]
                )
            elif widget_class == 'LabelFrame':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Canvas':
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'Menu':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["selection_color"],
                    activeforeground=self.theme["background"]
                )
            
            # Recursively apply theme to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Some widgets might not support certain options
            pass
        
        # Apply theme to root window
        self._apply_theme_to_widget(self.master)
        
        # Apply theme to root window
        self._apply_theme_to_widget(self.master)

    def _apply_theme_to_widget(self, widget):
        """Apply theme colors to a tkinter widget and its children."""
        try:
            # Apply theme to the widget itself
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'Toplevel', 'Tk']:
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'Label':
                widget.configure(bg=self.theme["background"], fg=self.theme["foreground"])
            elif widget_class == 'Button':
                # Only apply theme if the button doesn't have custom styling
                if widget.cget('bg') in ['SystemButtonFace', '#d9d9d9', '#ececec']:  # Default button colors
                    btn_theme = self.theme["buttons"].get("DefaultBotton", self.theme["buttons"]["DefaultBotton"])
                    widget.configure(
                        bg=btn_theme["background"],
                        fg=self.theme["foreground"],
                        activebackground=btn_theme["active_background"],
                        activeforeground=self.theme["foreground"]
                    )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Text':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Listbox':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Checkbutton':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["toolbar"]
                )
            elif widget_class == 'Radiobutton':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["toolbar"]
                )
            elif widget_class == 'Scale':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["selection_color"],
                    troughcolor=self.theme["toolbar"]
                )
            elif widget_class == 'Scrollbar':
                widget.configure(
                    bg=self.theme["toolbar"],
                    troughcolor=self.theme["background"],
                    activebackground=self.theme["selection_color"]
                )
            elif widget_class == 'LabelFrame':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Canvas':
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'Menu':
                widget.configure(
                    bg=self.theme["toolbar"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["selection_color"],
                    activeforeground=self.theme["background"]
                )
            
            # Recursively apply theme to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Some widgets might not support certain options
            pass

    @staticmethod
    def file_selector(title : str = None, filetypes=None, initialdir=None, use_abspath=True):
        window_title = title if title else "Select File"
        if use_abspath and initialdir:
            initialdir = os.path.abspath(initialdir)
        elif initialdir is None:
            initialdir = os.getcwd()

        if filetypes is None:
            filetypes = [("All Files", "*.*")]
        elif isinstance(filetypes, str):
            filetypes = [(filetypes, "*.*")]
        else:
            filetypes = filetypes
        
        file_path = filedialog.askopenfilename(
            title=window_title,
            filetypes=filetypes,
            initialdir=initialdir
        )
        return file_path

    def build_left_panel(self, parent: tk.Frame):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Main data field - create this first so we can pass it to other components
        self.data_field = DataField("Main Data Field", "", parent)
        self.data_field.frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        
        # Apply theme to data field
        self._apply_theme_to_widget(self.data_field.frame)
        
        self._add_popout_button_to_corner(self.data_field.frame, "Main Data Field", self.data_field.frame, parent, 4, 0, "grid", {"sticky": "nsew", "padx": 5, "pady": 5})

        # Top control buttons - now we can pass data_field
        self.top_options = TopOptions(parent, self.islat_class, theme=self.theme, data_field=self.data_field)
        self.top_options.frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        
        # Apply theme to top options
        self._apply_theme_to_widget(self.top_options.frame)
        
        self._add_popout_button_to_corner(self.top_options.frame, "Top Options", self.top_options, parent, 0, 0, "grid", {"sticky": "ew", "padx": 5, "pady": 2})

        # Molecule table
        self.molecule_table = MoleculeWindow("Molecule Table", parent, self.molecule_data, self.plot, self.config, self.islat_class)
        self.molecule_table.frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Apply theme to molecule table
        self._apply_theme_to_widget(self.molecule_table.frame)
        
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
        load_spectrum_btn = tk.Button(file_frame, text="Load Spectrum", command=self.islat_class.load_spectrum)
        load_spectrum_btn.pack()
        
        # Apply theme to these widgets
        self._apply_theme_to_widget(file_frame)
        
        self._add_popout_button_to_corner(file_frame, "Spectrum File", file_frame, parent, 2, 0, "grid", {"sticky": "ew", "padx": 5, "pady": 5})

        # Control panel for input parameters
        control_panel_frame = tk.LabelFrame(parent, text="Control Panel")
        control_panel_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.control_panel = ControlPanel(control_panel_frame, self.islat_class)
        
        # Apply theme to control panel
        self._apply_theme_to_widget(control_panel_frame)
        
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
                # Apply theme to popout button
                btn.configure(
                    bg=self.theme["buttons"]["DefaultBotton"]["background"],
                    fg=self.theme["foreground"],
                    activebackground=self.theme["buttons"]["DefaultBotton"]["active_background"],
                    activeforeground=self.theme["foreground"]
                )
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
        
        # Apply theme to popout window
        self._apply_theme_to_widget(pop)
        
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
            # Apply theme to the button
            btn.configure(
                bg=self.theme["buttons"]["DefaultBotton"]["background"],
                fg=self.theme["foreground"],
                activebackground=self.theme["buttons"]["DefaultBotton"]["active_background"],
                activeforeground=self.theme["foreground"]
            )
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
        
        # Apply theme to right frame
        self._apply_theme_to_widget(right_frame)
        
        # Handle plot popout - create a container frame for better control
        plot_container = tk.Frame(right_frame)
        plot_container.pack(fill="both", expand=True)
        
        # Apply theme to plot container
        self._apply_theme_to_widget(plot_container)
        
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
        
        # Apply theme to left frame
        self._apply_theme_to_widget(left_frame)
        
        self.build_left_panel(left_frame)

        # Bottom function buttons
        self.bottom_options = BottomOptions(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.config)
        self.bottom_options.frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # Apply theme to bottom options frame
        self._apply_theme_to_widget(self.bottom_options.frame)
        
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
import tkinter as tk
from tkinter import filedialog
#from .MainPlot import iSLATPlot
##from .Data_field import DataField
#from .MoleculeWindow import MoleculeWindow
#from .ControlPanel import ControlPanel
#from .TopOptions import TopOptions
#from .BottomOptions import BottomOptions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from iSLAT.COMPONENTS.GUI.Widgets.Plotting.PlotRenderer import PlotRenderer
from iSLAT.COMPONENTS.GUI.Widgets.Plotting.MainSpectralPlot import MainSpectralPlot
from iSLAT.COMPONENTS.GUI.Widgets.Plotting.LineInspectionPlot import LineInspectionPlot
from iSLAT.COMPONENTS.GUI.Widgets.Plotting.PopulationDiagram import PopulationDiagram
from iSLAT.COMPONENTS.GUI.Widgets.BottomOptions import BottomOptions
from iSLAT.COMPONENTS.GUI.Widgets.TopOptions import TopOptions
from iSLAT.COMPONENTS.GUI.Widgets.MoleculeWindow import MoleculeWindow
from iSLAT.COMPONENTS.GUI.Widgets.Data_field import DataField
from iSLAT.COMPONENTS.GUI.Widgets.ControlPanel import ControlPanel

class iSLATGUI:
    """
    iSLATGUI class to handle the graphical user interface of the iSLAT application.
    This class is responsible for initializing and managing the GUI components.
    """

    def __init__(self, theme, molecule_data=None, wave_data=None, flux_data=None, config=None, islat_class_ref=None):
        self.theme = theme
        self.molecule_data = molecule_data or []
        self.wave_data = wave_data or []
        self.flux_data = flux_data or []
        self.config = config or {"theme": theme}
        self.islat_class = islat_class_ref
        
        self._popout_states = {}  # Track popout states for widgets
        self._popout_windows = {}  # Track active popout windows
        self._plot_data = {}  # Store plot data for persistence
        self.main = None
        self.plot_group = None
        self.windows = {}
        
        # Mapping of subplot types to their classes
        self._subplot_classes = {
            'MainSpectralPlot': MainSpectralPlot,
            'LineInspectionPlot': LineInspectionPlot,
            'PopulationDiagram': PopulationDiagram
        }
        '''self.main = tk.Toplevel()
        self.plot_group = PlotRenderer(theme=theme)

        self.windows = {
            "main": self.main,
            "plot_group": self.plot_group,
        }'''

    def create_windows(self):
        self.main = tk.Tk()
        self.main.title("iSLAT Version 5.00.00")
        
        # Configure main window grid
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=3)
        self.main.rowconfigure(0, weight=1)
        self.main.rowconfigure(1, weight=0)
        
        # Set up cleanup on window close
        self.main.protocol("WM_DELETE_WINDOW", self._on_main_window_close)
        
        # Create main layout
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()
        
        self.windows = {
            "main": self.main,
            "plot_group": self.plot_group,
        }
        
        self.main.mainloop()
    
    def _create_left_panel(self):
        """Create the left panel with all control widgets"""
        left_frame = tk.Frame(self.main)
        left_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)  # Make molecule window expandable
        left_frame.grid_rowconfigure(4, weight=1)  # Make data field expandable
        left_frame.grid_columnconfigure(0, weight=1)

        # Top control buttons
        self.top_options = TopOptions(left_frame, self.islat_class, theme=self.theme, data_field=None)  # Will set data_field later
        self.top_options.frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.register_widget_for_popout(self.top_options.frame, "Top Options", self.top_options)

        # Molecule table/window
        self.molecule_table = MoleculeWindow("Molecule Table", left_frame, self.molecule_data, None, self.config, self.islat_class)  # Will set plot later
        self.molecule_table.frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.register_widget_for_popout(self.molecule_table.frame, "Molecule Table", self.molecule_table)

        # Spectrum file selector
        self._create_spectrum_file_selector(left_frame)

        # Control panel for input parameters
        control_panel_frame = tk.LabelFrame(left_frame, text="Control Panel")
        control_panel_frame.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.control_panel = ControlPanel(control_panel_frame, self.islat_class)
        self.register_widget_for_popout(control_panel_frame, "Control Panel", self.control_panel)

        # Main data field - create last so we can pass it to other components
        self.data_field = DataField("Main Data Field", "", left_frame)
        self.data_field.frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        self.register_widget_for_popout(self.data_field.frame, "Main Data Field", self.data_field)
        
        # Update top_options to reference data_field
        if hasattr(self.top_options, 'set_data_field'):
            self.top_options.set_data_field(self.data_field)
    
    def _create_spectrum_file_selector(self, parent):
        """Create the spectrum file selector widget"""
        file_frame = tk.LabelFrame(parent, text="Spectrum File")
        file_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if self.islat_class and hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        
        self.file_label = tk.Label(file_frame, text=default_text)
        self.file_label.pack()
        
        # Create load button - handle case where islat_class might not have load_spectrum method
        load_command = None
        if self.islat_class and hasattr(self.islat_class, 'load_spectrum'):
            load_command = self.islat_class.load_spectrum
        else:
            load_command = self._default_load_spectrum
            
        tk.Button(file_frame, text="Load Spectrum", command=load_command).pack()
        self.register_widget_for_popout(file_frame, "Spectrum File", file_frame)
    
    def _default_load_spectrum(self):
        """Default spectrum loading function if islat_class doesn't have one"""
        file_path = self.open_spectrum_file_selector()
        if file_path:
            self.file_label.config(text=f"Selected: {file_path}")
    
    def _create_right_panel(self):
        """Create the right panel with plots"""
        right_frame = tk.Frame(self.main)
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Create plot renderer
        self.plot_group = PlotRenderer(parent_widget=right_frame, theme=self.theme)
        
        # Add popout functionality to plot group and individual subplots
        self.register_widget_for_popout(self.plot_group.plot_cluster_frame, "Plot Group", self.plot_group)
        
        # Add popout buttons to individual subplots dynamically
        subplot_names = {
            'main': 'Main Spectrum',
            'line': 'Line Inspection', 
            'population': 'Population Diagram'
        }
        
        subplot_objects = {
            'main': self.plot_group.main_plot,
            'line': self.plot_group.line_plot,
            'population': self.plot_group.population_plot
        }
        
        for subplot_key, subplot_name in subplot_names.items():
            if subplot_key in self.plot_group.subplot_frames:
                frame = self.plot_group.subplot_frames[subplot_key]
                subplot_obj = subplot_objects.get(subplot_key)
                self.add_subplot_popout(subplot_name, frame, subplot_obj)
        
        # Update molecule table to reference the plot
        if hasattr(self.molecule_table, 'set_plot'):
            self.molecule_table.set_plot(self.plot_group)
        
    def _create_bottom_panel(self):
        """Create the bottom panel with function buttons"""
        self.bottom_options = BottomOptions(self.main, self.islat_class, self.theme, self.plot_group, self.data_field, self.config)
        self.bottom_options.frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.register_widget_for_popout(self.bottom_options.frame, "Bottom Options", self.bottom_options)

    def open_spectrum_file_selector(self, filetypes=[("All Files", "*.*")], default_path=None, title="Choose Spectrum Data File"):
        # Open a file dialog to select a spectrum file
        file_path = filedialog.askopenfilename(filetypes=filetypes, initialdir=default_path, title=title)
        return file_path

    def update_main_plots(self, data):
        """
        Update the main plots with new data.
        This method should be called whenever the data changes.
        """
        # Store the data for persistence across popouts
        self._plot_data = data.copy() if data else {}
        
        if "plot_group" in self.windows and self.plot_group:
            self.plot_group.update_all_plots(data)
            
        # Update any popped out plot renderers
        for widget_id, state in self._popout_states.items():
            if state.get("is_popped_out") and state.get("popout_content"):
                content = state["popout_content"]
                if hasattr(content, 'update_all_plots'):
                    content.update_all_plots(data)
                elif hasattr(content, 'render_spectrum_plot') and 'wave_data' in data:
                    # Individual subplot update
                    self._update_individual_subplot(content, data)

    def init_gui(self):
        # Initialize GUI components here
        print("Initializing GUI...")

    def start_main_loop(self):
        # Start the main loop of the GUI
        print("Starting main loop...")
        
    def _add_popout_button_to_widget(self, widget, title, content_object=None):
        """Add a popout button to a widget that allows it to be popped out into its own window"""
        widget_id = id(widget)
        
        # Store widget info for popout management
        self._popout_states[widget_id] = {
            "widget": widget,
            "title": title,
            "content_object": content_object,
            "parent": widget.master,
            "is_popped_out": False,
            "pack_info": None,
            "grid_info": None
        }
        
        # Store current geometry info
        try:
            self._popout_states[widget_id]["pack_info"] = widget.pack_info()
        except tk.TclError:
            pass
        
        try:
            self._popout_states[widget_id]["grid_info"] = widget.grid_info()
        except tk.TclError:
            pass
        
        # Add the popout button after a delay to ensure widget is ready
        def add_button():
            try:
                btn = tk.Button(widget, text="⧉", command=lambda: self._toggle_popout(widget_id),
                               width=2, height=1, relief="flat", padx=0, pady=0)
                btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
            except tk.TclError:
                widget.after(100, add_button)
        
        widget.after(10, add_button)
    
    def _toggle_popout(self, widget_id):
        """Toggle between popout and pop-in for a widget"""
        state = self._popout_states.get(widget_id)
        if not state:
            return
        
        if state["is_popped_out"]:
            self._pop_in_widget(widget_id)
        else:
            self._popout_widget(widget_id)
    
    def _popout_widget(self, widget_id):
        """Pop out a widget to a separate window"""
        state = self._popout_states.get(widget_id)
        if not state or state["is_popped_out"]:
            return
        
        widget = state["widget"]
        
        # Store current geometry info before removing
        try:
            state["pack_info"] = widget.pack_info()
        except tk.TclError:
            state["pack_info"] = None
        
        try:
            state["grid_info"] = widget.grid_info()
        except tk.TclError:
            state["grid_info"] = None
        
        # Create popout window first
        popout_window = tk.Toplevel(self.main)
        popout_window.title(state["title"])
        popout_window.geometry("800x600")
        
        # Configure popout window
        popout_window.grid_rowconfigure(0, weight=1)
        popout_window.grid_columnconfigure(0, weight=1)
        
        # Handle different widget types
        if state["content_object"] and hasattr(state["content_object"], 'plot_cluster_frame'):
            # This is a PlotRenderer - hide the original and create new one
            widget.pack_forget()
            
            # Create new PlotRenderer in popout window
            new_plot_renderer = PlotRenderer(popout_window, self.theme)
            
            # Store references
            state["popout_content"] = new_plot_renderer
            state["original_widget"] = widget
            
            # Update main reference temporarily
            original_plot_group = self.plot_group
            self.plot_group = new_plot_renderer
            self.windows["plot_group"] = new_plot_renderer
            state["original_plot_group"] = original_plot_group
            
            # Transfer current data to the new plot renderer
            if self._plot_data:
                new_plot_renderer.update_all_plots(self._plot_data)
            
        elif state["content_object"] and hasattr(state["content_object"], 'canvas'):
            # This is an individual subplot - hide original and create new
            widget.pack_forget()
            
            # Create container in popout window
            container = tk.Frame(popout_window)
            container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            
            # Create new subplot based on type
            original_subplot = state["content_object"]
            class_name = self._get_subplot_class_name(original_subplot)
            new_subplot = self._create_subplot_instance(class_name, container)
            
            # Store references
            state["popout_content"] = new_subplot
            state["original_widget"] = widget
            state["popout_container"] = container
            state["subplot_class_name"] = class_name
            
            # Update reference in main plot renderer using dynamic attribute lookup
            attr_name = self._get_subplot_attribute_name(class_name)
            if attr_name and hasattr(self.plot_group, attr_name):
                state["original_subplot"] = getattr(self.plot_group, attr_name)
                setattr(self.plot_group, attr_name, new_subplot)
            
            # Update the new subplot with current data
            if self._plot_data:
                self._update_individual_subplot(new_subplot, self._plot_data)
                
        else:
            # For other widgets, try to reparent directly
            widget.pack_forget()
            widget.master = popout_window
            widget.pack(fill=tk.BOTH, expand=True)
            state["popout_content"] = widget
            state["original_widget"] = widget
        
        # Store references
        self._popout_windows[widget_id] = popout_window
        state["is_popped_out"] = True
        
        # Handle window close - this ensures the window can be closed properly
        def on_close():
            self._pop_in_widget(widget_id)
        
        popout_window.protocol("WM_DELETE_WINDOW", on_close)
    
    def _pop_in_widget(self, widget_id):
        """Pop in a widget back to the main window"""
        state = self._popout_states.get(widget_id)
        if not state or not state["is_popped_out"]:
            return
        
        popout_window = self._popout_windows.get(widget_id)
        if not popout_window:
            return
        
        try:
            # Handle different widget types
            if state["content_object"] and hasattr(state["content_object"], 'plot_cluster_frame'):
                # This is a PlotRenderer - restore the original
                original_widget = state["original_widget"]
                original_plot_group = state["original_plot_group"]
                
                # Restore the original widget
                if state["pack_info"]:
                    original_widget.pack(**state["pack_info"])
                elif state["grid_info"]:
                    original_widget.grid(**state["grid_info"])
                else:
                    original_widget.pack(fill=tk.BOTH, expand=True)
                
                # Restore original plot group reference
                self.plot_group = original_plot_group
                self.windows["plot_group"] = original_plot_group
                
                # Update the state
                state["widget"] = original_widget
                
            elif state["content_object"] and hasattr(state["content_object"], 'canvas'):
                # This is an individual subplot - restore the original
                original_widget = state["original_widget"]
                original_subplot = state.get("original_subplot")
                class_name = state.get("subplot_class_name")
                
                # Restore the original widget
                if state["pack_info"]:
                    original_widget.pack(**state["pack_info"])
                elif state["grid_info"]:
                    original_widget.grid(**state["grid_info"])
                else:
                    original_widget.pack(fill=tk.BOTH, expand=True)
                
                # Restore original subplot reference in main plot renderer using dynamic lookup
                if original_subplot and class_name:
                    attr_name = self._get_subplot_attribute_name(class_name)
                    if attr_name and hasattr(self.plot_group, attr_name):
                        setattr(self.plot_group, attr_name, original_subplot)
                
                # Update the state
                state["widget"] = original_widget
                
            else:
                # For other widgets, reparent back to original parent
                widget = state["popout_content"]
                widget.pack_forget()
                widget.master = state["parent"]
                
                # Restore geometry
                if state["pack_info"]:
                    widget.pack(**state["pack_info"])
                elif state["grid_info"]:
                    widget.grid(**state["grid_info"])
                else:
                    widget.pack(fill=tk.BOTH, expand=True)
            
            # Clean up
            popout_window.destroy()
            if widget_id in self._popout_windows:
                del self._popout_windows[widget_id]
            state["is_popped_out"] = False
            state["popout_content"] = None
            
            # Clean up stored references
            cleanup_keys = ["original_widget", "original_plot_group", "original_subplot", 
                          "popout_container", "subplot_class_name"]
            for key in cleanup_keys:
                if key in state:
                    del state[key]
            
        except tk.TclError as e:
            print(f"Error during pop-in: {e}")
            # Even if there's an error, clean up the window
            try:
                popout_window.destroy()
            except:
                pass
            if widget_id in self._popout_windows:
                del self._popout_windows[widget_id]
            state["is_popped_out"] = False
    
    def register_widget_for_popout(self, widget, title, content_object=None):
        """
        Register a widget for popout functionality.
        
        Args:
            widget: The tkinter widget to make popout-able
            title: The title for the popout window
            content_object: Optional object that contains the widget's functionality
        """
        self._add_popout_button_to_widget(widget, title, content_object)
    
    def add_subplot_popout(self, subplot_name, subplot_frame, subplot_object):
        """
        Add popout functionality to individual subplots.
        
        Args:
            subplot_name: Name of the subplot (e.g., "Main Plot", "Line Inspection")
            subplot_frame: The frame containing the subplot
            subplot_object: The subplot object (MainSpectralPlot, LineInspectionPlot, etc.)
        """
        self._add_popout_button_to_widget(subplot_frame, subplot_name, subplot_object)

    def _on_main_window_close(self):
        """Clean up all popout windows when main window closes"""
        # Close all popout windows
        for widget_id in list(self._popout_windows.keys()):
            popout_window = self._popout_windows.get(widget_id)
            if popout_window:
                try:
                    popout_window.destroy()
                except tk.TclError:
                    pass
        
        # Clear tracking dictionaries
        self._popout_windows.clear()
        self._popout_states.clear()
        
        # Destroy main window
        self.main.destroy()

    def _update_individual_subplot(self, subplot, data):
        """Update an individual subplot with data"""
        subplot_type = type(subplot).__name__
        
        if subplot_type == 'MainSpectralPlot' and 'wave_data' in data and 'flux_data' in data:
            subplot.render_spectrum_plot(
                data['wave_data'],
                data['flux_data'],
                data.get('molecules', []),
                summed_flux=data.get('summed_flux'),
                error_data=data.get('error_data')
            )
        elif subplot_type == 'LineInspectionPlot' and 'line_wave' in data and 'line_flux' in data:
            subplot.render_line_inspection(
                data['line_wave'],
                data['line_flux'],
                line_label=data.get('line_label')
            )
        elif subplot_type == 'PopulationDiagram' and 'active_molecule' in data:
            subplot.render_population_diagram(
                data['active_molecule'],
                wave_range=data.get('wave_range')
            )
    
    def _get_subplot_class_name(self, subplot_obj):
        """Get the class name of a subplot object"""
        return type(subplot_obj).__name__
    
    def _create_subplot_instance(self, class_name, parent):
        """Create a new instance of a subplot class"""
        if class_name in self._subplot_classes:
            return self._subplot_classes[class_name](parent, self.theme)
        else:
            raise ValueError(f"Unknown subplot class: {class_name}")
    
    def _get_subplot_attribute_name(self, class_name):
        """Get the attribute name in PlotRenderer for a given subplot class"""
        mapping = {
            'MainSpectralPlot': 'main_plot',
            'LineInspectionPlot': 'line_plot',
            'PopulationDiagram': 'population_plot'
        }
        return mapping.get(class_name)
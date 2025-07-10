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

class ResizableFrame(tk.Frame):
    """A frame that can be resized by dragging its borders."""
    
    def __init__(self, parent, orientation='vertical', sash_size=4, **kwargs):
        super().__init__(parent, **kwargs)
        self.orientation = orientation
        self.sash_size = sash_size
        self.frames = []
        self.sashes = []
        self.dragging = False
        self.drag_data = {"x": 0, "y": 0, "sash": None}
        self.total_weight = 0
        self.initialized = False
        
    def add_frame(self, frame, weight=1, minsize=50):
        """Add a frame to the resizable container."""
        frame_info = {
            'frame': frame, 
            'weight': weight, 
            'minsize': minsize,
            'current_size': 0  # Will be calculated in _calculate_initial_sizes
        }
        self.frames.append(frame_info)
        self.total_weight += weight
        
        # Create sash if not the first frame
        if len(self.frames) > 1:
            sash = tk.Frame(self, cursor='sb_v_double_arrow' if self.orientation == 'vertical' else 'sb_h_double_arrow')
            # Style the sash to make it more visible
            sash.configure(bg='#888888', relief='raised', bd=1)
            self.sashes.append(sash)
            self._bind_sash_events(sash, len(self.sashes) - 1)
        
        # Delay initial layout until after the widget is mapped
        self.after(10, self._initialize_layout)
        
        # Bind to configure events to handle window resizing
        self.bind("<Configure>", self._on_configure)
    
    def _apply_theme_to_sashes(self, theme):
        """Apply theme styling to sashes."""
        for sash in self.sashes:
            sash.configure(bg=theme.get("toolbar", "#888888"))
    
    def _on_configure(self, event):
        """Handle window resize events."""
        if event.widget == self and self.initialized:
            self._calculate_initial_sizes()
            self._layout_frames()
    
    def _initialize_layout(self):
        """Initialize the layout after the widget is mapped."""
        if not self.initialized and self.winfo_width() > 1 and self.winfo_height() > 1:
            self._calculate_initial_sizes()
            self._layout_frames()
            self.initialized = True
        elif not self.initialized:
            # Try again later if widget isn't ready
            self.after(50, self._initialize_layout)
    
    def _calculate_initial_sizes(self):
        """Calculate initial sizes for frames based on their weights."""
        if self.orientation == 'vertical':
            available_space = self.winfo_height() - (len(self.sashes) * self.sash_size)
        else:
            available_space = self.winfo_width() - (len(self.sashes) * self.sash_size)
        
        # Ensure minimum space
        total_minsize = sum(frame['minsize'] for frame in self.frames)
        available_space = max(available_space, total_minsize)
        
        # First, allocate minimum sizes for zero-weight frames
        remaining_space = available_space
        for frame_info in self.frames:
            if frame_info['weight'] == 0:
                frame_info['current_size'] = frame_info['minsize']
                remaining_space -= frame_info['minsize']
            else:
                remaining_space -= frame_info['minsize']
        
        # Then distribute remaining space among weighted frames
        total_weight = sum(frame['weight'] for frame in self.frames if frame['weight'] > 0)
        
        if total_weight > 0 and remaining_space > 0:
            for frame_info in self.frames:
                if frame_info['weight'] > 0:
                    proportional_size = int((frame_info['weight'] / total_weight) * remaining_space)
                    frame_info['current_size'] = frame_info['minsize'] + proportional_size
    
    def _bind_sash_events(self, sash, sash_index):
        """Bind mouse events to sash for dragging."""
        sash.bind("<Button-1>", lambda e: self._start_drag(e, sash_index))
        sash.bind("<B1-Motion>", lambda e: self._on_drag(e, sash_index))
        sash.bind("<ButtonRelease-1>", lambda e: self._end_drag(e, sash_index))
        
        # Add hover effects
        sash.bind("<Enter>", lambda e: sash.configure(bg='#999999'))
        sash.bind("<Leave>", lambda e: sash.configure(bg='#888888'))
        
    def _start_drag(self, event, sash_index):
        """Start dragging a sash."""
        self.dragging = True
        self.drag_data["sash"] = sash_index
        if self.orientation == 'vertical':
            self.drag_data["y"] = event.y_root
        else:
            self.drag_data["x"] = event.x_root
    
    def _on_drag(self, event, sash_index):
        """Handle sash dragging."""
        if not self.dragging:
            return
            
        if self.orientation == 'vertical':
            delta = event.y_root - self.drag_data["y"]
            self.drag_data["y"] = event.y_root
        else:
            delta = event.x_root - self.drag_data["x"]
            self.drag_data["x"] = event.x_root
        
        self._resize_frames(sash_index, delta)
    
    def _end_drag(self, event, sash_index):
        """End dragging."""
        self.dragging = False
        
    def _resize_frames(self, sash_index, delta):
        """Resize frames based on sash movement."""
        if sash_index >= len(self.frames) - 1:
            return
            
        frame1 = self.frames[sash_index]
        frame2 = self.frames[sash_index + 1]
        
        # Calculate new sizes
        new_size1 = max(frame1['minsize'], frame1['current_size'] + delta)
        new_size2 = max(frame2['minsize'], frame2['current_size'] - delta)
        
        # Adjust if one frame hits minimum
        if new_size1 == frame1['minsize'] and delta < 0:
            delta = frame1['minsize'] - frame1['current_size']
            new_size2 = frame2['current_size'] - delta
        elif new_size2 == frame2['minsize'] and delta > 0:
            delta = frame2['current_size'] - frame2['minsize']
            new_size1 = frame1['current_size'] + delta
            new_size2 = frame2['minsize']
        
        # Update sizes
        frame1['current_size'] = new_size1
        frame2['current_size'] = new_size2
        
        self._layout_frames()
    
    def _layout_frames(self):
        """Layout frames and sashes."""
        current_pos = 0
        
        for i, frame_info in enumerate(self.frames):
            frame = frame_info['frame']
            size = frame_info['current_size']
            
            if self.orientation == 'vertical':
                frame.place(x=0, y=current_pos, relwidth=1, height=size)
                current_pos += size
                
                # Place sash if not the last frame
                if i < len(self.sashes):
                    sash = self.sashes[i]
                    sash.place(x=0, y=current_pos, relwidth=1, height=self.sash_size)
                    current_pos += self.sash_size
            else:
                frame.place(x=current_pos, y=0, width=size, relheight=1)
                current_pos += size
                
                # Place sash if not the last frame
                if i < len(self.sashes):
                    sash = self.sashes[i]
                    sash.place(x=current_pos, y=0, width=self.sash_size, relheight=1)
                    current_pos += self.sash_size

class GUI:
    def __init__(self, master, molecule_data, wave_data, flux_data, config, islat_class_ref):
        if master is None:
            self.master = tk.Tk()
            self.master.title("iSLAT - Infrared Spectral Line Analysis Tool")
            self.master.resizable(True, True)
            # Set minimum size to maintain usability
            self.master.minsize(800, 600)
            # Configure initial window size based on screen dimensions
            self._configure_initial_size()
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

    def _configure_initial_size(self):
        """Configure initial window size based on screen resolution."""
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Use 80% of screen width and 75% of screen height
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.75)
        
        # Ensure minimum size constraints
        window_width = max(window_width, 800)
        window_height = max(window_height, 600)
        
        # Calculate position to center the window
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        
        self.master.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

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
        # Create a resizable frame container for the left panel
        self.left_resizable = ResizableFrame(parent, orientation='vertical', sash_size=4)
        self.left_resizable.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Apply theme to the resizable frame
        self.left_resizable.configure(bg=self.theme["background"])
        self.left_resizable._apply_theme_to_sashes(self.theme)

        # Apply theme to the resizable frame
        self.left_resizable.configure(bg=self.theme["background"])
        
        # Create individual frames for each component
        top_options_frame = tk.Frame(self.left_resizable)
        molecule_table_frame = tk.Frame(self.left_resizable)
        file_selector_frame = tk.Frame(self.left_resizable)
        control_panel_frame = tk.Frame(self.left_resizable)
        data_field_frame = tk.Frame(self.left_resizable)
        
        # Apply theme to frames
        for frame in [top_options_frame, molecule_table_frame, file_selector_frame, control_panel_frame, data_field_frame]:
            frame.configure(bg=self.theme["background"])
        
        # Add frames to resizable container with different weights and minimum sizes
        self.left_resizable.add_frame(top_options_frame, weight=0, minsize=80)
        self.left_resizable.add_frame(molecule_table_frame, weight=2, minsize=150)
        self.left_resizable.add_frame(file_selector_frame, weight=0, minsize=80)
        self.left_resizable.add_frame(control_panel_frame, weight=2, minsize=120)
        self.left_resizable.add_frame(data_field_frame, weight=4, minsize=200)

        # Main data field - create this first so we can pass it to other components
        self.data_field = DataField("Main Data Field", "", data_field_frame)
        self.data_field.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Apply theme to data field
        self._apply_theme_to_widget(self.data_field.frame)
        
        self._add_popout_button_to_corner(self.data_field.frame, "Main Data Field", self.data_field.frame, data_field_frame, 0, 0, "pack", {"fill": "both", "expand": True, "padx": 5, "pady": 5})

        # Top control buttons - now we can pass data_field
        self.top_options = TopOptions(top_options_frame, self.islat_class, theme=self.theme, data_field=self.data_field)
        self.top_options.frame.pack(fill="both", expand=True, padx=5, pady=2)
        
        # Apply theme to top options
        self._apply_theme_to_widget(self.top_options.frame)
        
        self._add_popout_button_to_corner(self.top_options.frame, "Top Options", self.top_options, top_options_frame, 0, 0, "pack", {"fill": "both", "expand": True, "padx": 5, "pady": 2})

        # Molecule table
        self.molecule_table = MoleculeWindow("Molecule Table", molecule_table_frame, self.molecule_data, self.plot, self.config, self.islat_class)
        self.molecule_table.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Apply theme to molecule table
        self._apply_theme_to_widget(self.molecule_table.frame)
        
        self._add_popout_button_to_corner(self.molecule_table.frame, "Molecule Table", self.molecule_table, molecule_table_frame, 0, 0, "pack", {"fill": "both", "expand": True, "padx": 5, "pady": 5})

        # Spectrum file selector
        file_frame = tk.LabelFrame(file_selector_frame, text="Spectrum File")
        file_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        
        self.file_label = tk.Label(file_frame, text=default_text, wraplength=250)
        self.file_label.pack(pady=2)
        load_spectrum_btn = tk.Button(file_frame, text="Load Spectrum", command=self.islat_class.load_spectrum)
        load_spectrum_btn.pack(pady=2)
        
        # Apply theme to these widgets
        self._apply_theme_to_widget(file_frame)
        
        self._add_popout_button_to_corner(file_frame, "Spectrum File", file_frame, file_selector_frame, 0, 0, "pack", {"fill": "both", "expand": True, "padx": 5, "pady": 5})

        # Control panel for input parameters
        control_frame = tk.LabelFrame(control_panel_frame, text="Control Panel")
        control_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.control_panel = ControlPanel(control_frame, self.islat_class)
        
        # Apply theme to control panel
        self._apply_theme_to_widget(control_frame)
        
        self._add_popout_button_to_corner(control_frame, "Control Panel", control_frame, control_panel_frame, 0, 0, "pack", {"fill": "both", "expand": True, "padx": 5, "pady": 5})

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
        
        # Set size based on screen resolution for popout windows
        screen_width = pop.winfo_screenwidth()
        screen_height = pop.winfo_screenheight()
        window_width = min(int(screen_width * 0.6), 800)
        window_height = min(int(screen_height * 0.6), 600)
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        pop.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        pop.minsize(400, 300)
        
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
        
        # Configure main window for resizable layout
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0, minsize=60)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create a main container frame
        main_container = tk.Frame(self.window)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Create horizontal resizable frame for left panel and plot area
        self.main_resizable = ResizableFrame(main_container, orientation='horizontal', sash_size=6)
        self.main_resizable.pack(fill="both", expand=True)
        
        # Apply theme to main resizable frame
        self.main_resizable.configure(bg=self.theme["background"])
        self.main_resizable._apply_theme_to_sashes(self.theme)
        
        # Create frames for left panel and right panel (plot)
        left_main_frame = tk.Frame(self.main_resizable)
        right_main_frame = tk.Frame(self.main_resizable)
        
        # Apply theme to main frames
        left_main_frame.configure(bg=self.theme["background"])
        right_main_frame.configure(bg=self.theme["background"])
        
        # Add frames to horizontal resizable container
        self.main_resizable.add_frame(left_main_frame, weight=1, minsize=350)
        self.main_resizable.add_frame(right_main_frame, weight=2, minsize=450)

        # Right side: plots
        right_frame = tk.Frame(right_main_frame)
        right_frame.pack(fill="both", expand=True)
        
        # Configure right frame for responsive plot
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
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
        left_frame = tk.Frame(left_main_frame)
        left_frame.pack(fill="both", expand=True)
        
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
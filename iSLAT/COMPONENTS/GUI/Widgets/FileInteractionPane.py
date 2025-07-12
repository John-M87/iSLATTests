import tkinter as tk
from tkinter import ttk
import os


class FileInteractionPane:
    def __init__(self, parent, islat_class, theme):
        """
        Initialize the File Interaction Pane widget.
        
        Args:
            parent: The parent widget to contain this pane
            islat_class: Reference to the main iSLAT class instance
            theme: Theme dictionary for styling
        """
        self.parent = parent
        self.islat_class = islat_class
        self.theme = theme
        
        # Create the main frame
        self.frame = tk.LabelFrame(parent, text="Spectrum File")
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Apply initial theme to the frame
        self._apply_theme_to_frame()
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        
        # Create the file label
        self.file_label = tk.Label(self.frame, text=default_text, wraplength=250)
        self.file_label.pack(pady=2)
        
        # Create the load spectrum button
        self.load_spectrum_btn = tk.Button(
            self.frame, 
            text="Load Spectrum", 
            command=self.islat_class.load_spectrum
        )
        self.load_spectrum_btn.pack(pady=2)
        
        # Apply theme to all widgets
        self.apply_theme(self.theme)
    
    def _apply_theme_to_frame(self):
        """Apply theme to the main frame."""
        self.frame.configure(
            bg=self.theme["background"],
            fg=self.theme["foreground"]
        )
    
    def apply_theme(self, theme):
        """
        Apply theme to all widgets in the file interaction pane.
        
        Args:
            theme: Theme dictionary containing color and style information
        """
        self.theme = theme
        
        # Apply theme to the main frame
        self.frame.configure(
            bg=theme["background"],
            fg=theme["foreground"]
        )
        
        # Apply theme to the file label
        self.file_label.configure(
            bg=theme["background"],
            fg=theme["foreground"]
        )
        
        # Apply theme to the load spectrum button
        btn_theme = theme["buttons"].get("DefaultBotton", theme["buttons"]["DefaultBotton"])
        self.load_spectrum_btn.configure(
            bg=btn_theme["background"],
            fg=theme["foreground"],
            activebackground=btn_theme["active_background"],
            activeforeground=theme["foreground"]
        )
        
        # Apply theme recursively to all child widgets
        self._apply_theme_to_widget(self.frame)
    
    def _apply_theme_to_widget(self, widget):
        """
        Recursively apply theme to a widget and its children.
        
        Args:
            widget: The widget to apply theme to
        """
        try:
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'Toplevel', 'Tk']:
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'LabelFrame':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Label':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Button':
                btn_theme = self.theme["buttons"].get("DefaultBotton", self.theme["buttons"]["DefaultBotton"])
                widget.configure(
                    bg=btn_theme["background"],
                    fg=self.theme["foreground"],
                    activebackground=btn_theme["active_background"],
                    activeforeground=self.theme["foreground"]
                )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"]
                )
            elif widget_class == 'Text':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"]
                )
            elif widget_class == 'Scrollbar':
                widget.configure(
                    bg=self.theme["background"],
                    troughcolor=self.theme["background"],
                    activebackground=self.theme["foreground"]
                )
            
            # Recursively apply theme to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Some widgets might not support certain options
            pass
    
    def update_file_label(self, filename=None):
        """
        Update the file label text.
        
        Args:
            filename: The filename to display. If None, checks islat_class for loaded spectrum name.
        """
        if filename:
            display_text = f"Loaded: {filename}"
        elif hasattr(self.islat_class, 'loaded_spectrum_name') and self.islat_class.loaded_spectrum_name:
            display_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        else:
            display_text = "No file loaded"
        
        self.file_label.configure(text=display_text)
    
    def refresh(self):
        """
        Refresh the file interaction pane to show current state.
        This method can be called when the GUI needs to update its display.
        """
        self.update_file_label()
        self.apply_theme(self.theme)
    
    def get_loaded_filename(self):
        """
        Get the currently loaded filename.
        
        Returns:
            str: The filename if loaded, None otherwise
        """
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            return self.islat_class.loaded_spectrum_name
        return None
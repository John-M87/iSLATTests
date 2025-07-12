import tkinter as tk

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
            # Update hover colors too
            def make_hover_handlers(s, theme_ref):
                return (
                    lambda e: s.configure(bg=theme_ref.get("selection_color", "#999999")),
                    lambda e: s.configure(bg=theme_ref.get("toolbar", "#888888"))
                )
            enter_handler, leave_handler = make_hover_handlers(sash, theme)
            sash.bind("<Enter>", enter_handler)
            sash.bind("<Leave>", leave_handler)
    
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
#!/usr/bin/env python3
"""
Multi-Function GUI Application
==============================

A tabbed GUI application that provides:
1. Face Anonymizer - Image privacy protection with blur, pixelate, emoji, replace modes
2. Speech to Text - Voice recording and transcription
3. Simple Text Input - Basic text input interface
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
from pathlib import Path
import cv2
from PIL import Image, ImageTk
import speech_recognition as sr
import pyaudio
import wave
import tempfile
import whisper
import numpy as np

# Import our main anonymizer
from face_anonymizer_main import detect_faces_yolo, anonymize_blur, anonymize_pixelate, anonymize_emoji, anonymize_synthetic, interactive_face_selection
from ultralytics import YOLO

class MultiFunctionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Function Application - Face Anonymizer | Speech to Text | Text Input")
        self.root.geometry("700x800")
        self.root.resizable(True, True)
        
        # Initialize all variables first
        self._init_variables()
        
        # Create the main notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_face_anonymizer_tab()
        self.create_speech_to_text_tab()
        self.create_text_input_tab()
        
        # Load YOLO model in background
        self.load_yolo_model()
    
    def _init_variables(self):
        """Initialize all GUI variables"""
        # Face Anonymizer variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.mode = tk.StringVar(value="blur")
        self.threshold = tk.DoubleVar(value=0.5)
        self.blocks = tk.IntVar(value=12)
        self.blur_strength = tk.IntVar(value=0)
        self.blur_type = tk.StringVar(value="gaussian")
        self.pixelate_method = tk.StringVar(value="nearest")
        self.blur_radius = tk.DoubleVar(value=0.0)
        self.pixelate_quality = tk.IntVar(value=50)
        self.selected_emoji = tk.StringVar(value="Baozou.png")
        self.synthetic_dir = tk.StringVar()
        self.random_faces = tk.BooleanVar()
        self.keep_largest = tk.BooleanVar()
        self.interactive = tk.BooleanVar()
        
        # Speech to Text variables
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_recording = tk.BooleanVar(value=False)
        self.speech_text = tk.StringVar()
        self.whisper_model = None
        self.current_recording = None
        
        # Load Whisper model in background
        self.load_whisper_model()
        
        # YOLO model
        self.model = None
        self.loading_model = False
    
    def create_face_anonymizer_tab(self):
        # Create Face Anonymizer tab
        self.face_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.face_tab, text="Face Anonymizer")
        
        # Main frame for face anonymizer
        main_frame = ttk.Frame(self.face_tab, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Face Anonymizer", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input file selection
        ttk.Label(main_frame, text="Input Image/Video:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_entry = ttk.Entry(main_frame, textvariable=self.input_path, width=50)
        self.input_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(row=1, column=2, pady=5)
        
        # Output file selection
        ttk.Label(main_frame, text="Output Path:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_entry = ttk.Entry(main_frame, textvariable=self.output_path, width=50)
        self.output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2, pady=5)
        
        # Mode selection
        ttk.Label(main_frame, text="Anonymization Mode:").grid(row=3, column=0, sticky=tk.W, pady=5)
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        modes = [("Blur", "blur"), ("Pixelate", "pixelate"), ("Emoji", "emoji"), ("Synthetic Face Swap", "replace")]
        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode, value=value).grid(row=0, column=i, padx=10)
        
        # Threshold
        ttk.Label(main_frame, text="Detection Threshold:").grid(row=4, column=0, sticky=tk.W, pady=5)
        threshold_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, variable=self.threshold, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
        self.threshold_label = ttk.Label(main_frame, text="0.5")
        self.threshold_label.grid(row=4, column=2, pady=5)
        
        # Mode-specific options frame
        self.options_frame = ttk.LabelFrame(main_frame, text="Mode Options", padding="10")
        self.options_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # General options
        options_frame = ttk.LabelFrame(main_frame, text="General Options", padding="10")
        options_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Checkbutton(options_frame, text="Interactive Face Selection", variable=self.interactive).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(options_frame, text="(Click faces to select/deselect them)", font=("Arial", 8)).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        ttk.Checkbutton(options_frame, text="Keep Largest Face (Main Subject)", variable=self.keep_largest).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.grid(row=8, column=0, columnspan=3, pady=5)
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="Process Image/Video", command=self.process_file, style="Accent.TButton")
        self.process_btn.grid(row=9, column=0, columnspan=3, pady=10)
        
        # Bind mode change and create initial options
        self.mode.trace('w', self.on_mode_change)
        self.selected_emoji.trace('w', lambda *args: self.update_emoji_preview())
        self.create_mode_options()
    
    def create_speech_to_text_tab(self):
        # Create Speech to Text tab
        self.speech_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.speech_tab, text="Speech to Text")
        
        # Main frame for speech to text
        speech_frame = ttk.Frame(self.speech_tab, padding="20")
        speech_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(speech_frame, text="Speech to Text", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(speech_frame, 
                                text="Click 'Start Recording' to begin recording your speech.\nClick 'Stop Recording' to end recording and convert to text using Whisper AI.\n(Powered by OpenAI Whisper - works offline, supports longer recordings)",
                                justify=tk.CENTER)
        instructions.pack(pady=(0, 20))
        
        # Recording controls
        controls_frame = ttk.Frame(speech_frame)
        controls_frame.pack(pady=10)
        
        self.record_btn = ttk.Button(controls_frame, text="🎤 Start Recording", command=self.start_recording)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(controls_frame, text="⏹ Stop Recording", command=self.stop_recording, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(controls_frame, text="🗑 Clear Text", command=self.clear_speech_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Recording status
        self.recording_status = ttk.Label(speech_frame, text="Loading Whisper AI model...", font=("Arial", 10))
        self.recording_status.pack(pady=5)
        
        # Text output area
        text_label = ttk.Label(speech_frame, text="Transcribed Text:", font=("Arial", 12, "bold"))
        text_label.pack(anchor=tk.W, pady=(20, 5))
        
        self.speech_text_area = scrolledtext.ScrolledText(speech_frame, width=70, height=20, wrap=tk.WORD)
        self.speech_text_area.pack(fill='both', expand=True, pady=5)
        
        # Save text button
        save_frame = ttk.Frame(speech_frame)
        save_frame.pack(pady=10)
        
        ttk.Button(save_frame, text="💾 Save Text", command=self.save_speech_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(save_frame, text="📋 Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
    
    def create_text_input_tab(self):
        # Create Simple Text Input tab
        self.text_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text="Text Input")
        
        # Main frame for text input
        text_frame = ttk.Frame(self.text_tab, padding="20")
        text_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(text_frame, text="Simple Text Input & Output", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(text_frame, 
                                text="Use the input area for typing text, then process it to display in the output area.",
                                justify=tk.CENTER)
        instructions.pack(pady=(0, 20))
        
        # Create a vertical paned window for input and output
        paned_window = ttk.PanedWindow(text_frame, orient=tk.VERTICAL)
        paned_window.pack(fill='both', expand=True, pady=(0, 10))
        
        # Top frame for output
        output_frame = ttk.Frame(paned_window)
        paned_window.add(output_frame, weight=1)
        
        # Text output area
        output_label = ttk.Label(output_frame, text="Text Output Area:", font=("Arial", 12, "bold"))
        output_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text_output_area = scrolledtext.ScrolledText(output_frame, width=70, height=12, wrap=tk.WORD, state='normal')
        self.text_output_area.pack(fill='both', expand=True, pady=5)
        
        # Bottom frame for input
        input_frame = ttk.Frame(paned_window)
        paned_window.add(input_frame, weight=1)
        
        # Text input area
        input_label = ttk.Label(input_frame, text="Text Input Area:", font=("Arial", 12, "bold"))
        input_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text_input_area = scrolledtext.ScrolledText(input_frame, width=70, height=12, wrap=tk.WORD)
        self.text_input_area.pack(fill='both', expand=True, pady=5)
        
        # Process controls (between input and output)
        process_frame = ttk.Frame(text_frame)
        process_frame.pack(pady=10)
        
        ttk.Button(process_frame, text="➡️ Copy to Output", command=self.copy_to_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="🔄 Transform Text", command=self.transform_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="🔠 Uppercase", command=self.make_uppercase).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="🔡 Lowercase", command=self.make_lowercase).pack(side=tk.LEFT, padx=5)
        
        # File and utility controls
        controls_frame = ttk.Frame(text_frame)
        controls_frame.pack(pady=10)
        
        ttk.Button(controls_frame, text="📁 Open File", command=self.open_text_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="💾 Save Input", command=self.save_text_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="💾 Save Output", command=self.save_output_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🗑 Clear Input", command=self.clear_text_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="🗑 Clear Output", command=self.clear_text_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="📊 Word Count", command=self.show_word_count).pack(side=tk.LEFT, padx=5)
        
        # Character/word count display
        self.text_stats = ttk.Label(text_frame, text="Input - Characters: 0 | Words: 0 | Output - Characters: 0 | Words: 0", font=("Arial", 10))
        self.text_stats.pack(pady=5)
        
        # Bind text change event for real-time stats
        self.text_input_area.bind('<KeyRelease>', self.update_text_stats)
        self.text_input_area.bind('<Button-1>', self.update_text_stats)
        self.text_output_area.bind('<KeyRelease>', self.update_text_stats)
        self.text_output_area.bind('<Button-1>', self.update_text_stats)
    
    def create_mode_options(self):
        # Clear existing widgets
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        
        mode = self.mode.get()
        
        if mode == "blur":
            # Blur type selection
            ttk.Label(self.options_frame, text="Blur Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
            blur_type_frame = ttk.Frame(self.options_frame)
            blur_type_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            blur_types = [("Gaussian", "gaussian"), ("Box", "box"), ("Median", "median")]
            for i, (text, value) in enumerate(blur_types):
                ttk.Radiobutton(blur_type_frame, text=text, variable=self.blur_type, value=value).grid(row=0, column=i, padx=10)
            
            # Blur strength
            ttk.Label(self.options_frame, text="Blur Strength (0=auto):").grid(row=1, column=0, sticky=tk.W, pady=5)
            blur_scale = ttk.Scale(self.options_frame, from_=0, to=50, variable=self.blur_strength, orient=tk.HORIZONTAL)
            blur_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
            ttk.Label(self.options_frame, textvariable=tk.StringVar(value="0")).grid(row=1, column=2, pady=5)
            
            # Blur radius (for Gaussian)
            ttk.Label(self.options_frame, text="Blur Radius (σ):").grid(row=2, column=0, sticky=tk.W, pady=5)
            radius_scale = ttk.Scale(self.options_frame, from_=0.0, to=5.0, variable=self.blur_radius, orient=tk.HORIZONTAL)
            radius_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
            ttk.Label(self.options_frame, textvariable=tk.StringVar(value="0.0")).grid(row=2, column=2, pady=5)
            
        elif mode == "pixelate":
            # Pixelation method selection
            ttk.Label(self.options_frame, text="Pixelation Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
            method_frame = ttk.Frame(self.options_frame)
            method_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            pixelate_methods = [("Nearest", "nearest"), ("Linear", "linear"), ("Cubic", "cubic")]
            for i, (text, value) in enumerate(pixelate_methods):
                ttk.Radiobutton(method_frame, text=text, variable=self.pixelate_method, value=value).grid(row=0, column=i, padx=10)
            
            # Block size
            ttk.Label(self.options_frame, text="Block Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
            blocks_scale = ttk.Scale(self.options_frame, from_=4, to=20, variable=self.blocks, orient=tk.HORIZONTAL)
            blocks_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
            ttk.Label(self.options_frame, textvariable=tk.StringVar(value="12")).grid(row=1, column=2, pady=5)
            
            # Pixelation quality
            ttk.Label(self.options_frame, text="Quality (%):").grid(row=2, column=0, sticky=tk.W, pady=5)
            quality_scale = ttk.Scale(self.options_frame, from_=10, to=100, variable=self.pixelate_quality, orient=tk.HORIZONTAL)
            quality_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
            ttk.Label(self.options_frame, textvariable=tk.StringVar(value="50")).grid(row=2, column=2, pady=5)
            
        elif mode == "emoji":
            ttk.Label(self.options_frame, text="Select Emoji:").grid(row=0, column=0, sticky=tk.W, pady=5)
            
            # Create emoji selection frame
            emoji_frame = ttk.Frame(self.options_frame)
            emoji_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            
            # Available emoji options
            emoji_options = [
                ("Baozou", "Baozou.png"),
                ("Cat", "Cat.png"), 
                ("Doge", "Doge.png"),
                ("Shrek", "Shrek.png"),
                ("Yao", "Yao.png")
            ]
            
            # Create radio buttons for emoji selection
            for i, (text, value) in enumerate(emoji_options):
                ttk.Radiobutton(emoji_frame, text=text, variable=self.selected_emoji, value=value).grid(row=0, column=i, padx=5)
            
            # Preview the selected emoji
            ttk.Label(self.options_frame, text="Preview:").grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
            self.emoji_preview_label = ttk.Label(self.options_frame, text="")
            self.emoji_preview_label.grid(row=1, column=1, columnspan=2, sticky=tk.W, pady=(10, 5))
            
            # Load and display emoji preview
            self.update_emoji_preview()
            
        elif mode == "replace":
            ttk.Label(self.options_frame, text="Synthetic Faces Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
            ttk.Entry(self.options_frame, textvariable=self.synthetic_dir, width=40).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=5)
            ttk.Button(self.options_frame, text="Browse", command=self.browse_synthetic_dir).grid(row=0, column=2, pady=5)
            
            ttk.Checkbutton(self.options_frame, text="Use Random Synthetic Faces", variable=self.random_faces).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
    
    def on_mode_change(self, *args):
        self.create_mode_options()
    
    def update_emoji_preview(self):
        """Update the emoji preview image"""
        try:
            emoji_file = self.selected_emoji.get()
            emoji_path = os.path.join("emoji", emoji_file)
            
            if os.path.exists(emoji_path):
                # Load and resize emoji for preview
                emoji_img = Image.open(emoji_path)
                emoji_img.thumbnail((50, 50), Image.Resampling.LANCZOS)  # Resize for preview
                emoji_photo = ImageTk.PhotoImage(emoji_img)
                
                # Update the preview label
                self.emoji_preview_label.configure(image=emoji_photo, text="")
                self.emoji_preview_label.image = emoji_photo  # Keep a reference
            else:
                self.emoji_preview_label.configure(image="", text=f"Emoji not found: {emoji_file}")
        except Exception as e:
            self.emoji_preview_label.configure(image="", text=f"Error loading emoji: {str(e)}")
    
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Image/Video",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            # Auto-generate output path
            if not self.output_path.get():
                input_path = Path(filename)
                output_path = input_path.parent / f"anonymized_{input_path.name}"
                self.output_path.set(str(output_path))
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)
    

    
    def browse_synthetic_dir(self):
        directory = filedialog.askdirectory(title="Select Synthetic Faces Folder")
        if directory:
            self.synthetic_dir.set(directory)
    
    def load_yolo_model(self):
        """Load YOLO model in background thread"""
        def load_model():
            try:
                self.status_var.set("Loading YOLO model...")
                self.progress.start()
                self.model = YOLO('yolov12l-face.pt')
                self.status_var.set("YOLO model loaded successfully")
                self.progress.stop()
            except Exception as e:
                self.status_var.set(f"Error loading YOLO model: {str(e)}")
                self.progress.stop()
                messagebox.showerror("Error", f"Failed to load YOLO model:\n{str(e)}")
        
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()
    
    def load_whisper_model(self):
        """Load Whisper model in background thread"""
        def load_model():
            try:
                self.recording_status.config(text="Loading Whisper model...")
                # Load the base model (you can change to "small", "medium", "large" for better accuracy but slower processing)
                self.whisper_model = whisper.load_model("base")
                self.recording_status.config(text="Whisper model loaded. Ready to record.")
            except Exception as e:
                self.recording_status.config(text=f"Error loading Whisper: {str(e)}")
                print(f"Whisper loading error: {e}")
        
        # Only start loading after the recording_status widget is created
        self.root.after(1000, lambda: threading.Thread(target=load_model, daemon=True).start())
    
    def process_file(self):
        """Process the selected file"""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output path")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "YOLO model is still loading. Please wait.")
            return
        
        # Validate mode-specific requirements
        mode = self.mode.get()
        if mode == "emoji":
            emoji_file = self.selected_emoji.get()
            emoji_path = os.path.join("emoji", emoji_file)
            if not os.path.exists(emoji_path):
                messagebox.showerror("Error", f"Selected emoji file not found: {emoji_path}")
                return
        
        if mode == "replace" and not self.synthetic_dir.get():
            messagebox.showerror("Error", "Please select a synthetic faces folder for replace mode")
            return
        
        # Show interactive mode info
        if self.interactive.get():
            messagebox.showinfo("Interactive Mode", 
                              "Interactive face selection enabled!\n\n"
                              "A window will open showing detected faces.\n"
                              "Click on faces to select (green) or deselect (red) them.\n"
                              "Press SPACE when done, or ESC to cancel.\n"
                              "Only selected faces will be anonymized.")
        
        # Start processing in background thread
        def process():
            try:
                self.status_var.set("Processing...")
                self.progress.start()
                self.process_btn.config(state='disabled')
                
                # Load image
                image = cv2.imread(self.input_path.get())
                if image is None:
                    raise Exception("Failed to load input image")
                
                # Detect faces
                faces = detect_faces_yolo(image, self.model, self.threshold.get())
                if not faces:
                    self.status_var.set("No faces detected")
                    return
                
                # Interactive face selection if enabled
                if self.interactive.get():
                    self.status_var.set("Starting interactive face selection...")
                    # Run interactive selection in main thread to avoid OpenCV issues
                    self.root.after(100, lambda: self.run_interactive_selection(image, faces))
                    return
                
                # Apply anonymization using the same method
                self.continue_processing(image, faces)
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            finally:
                self.progress.stop()
                self.process_btn.config(state='normal')
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def run_interactive_selection(self, image, faces):
        """Run interactive face selection in main thread"""
        try:
            # Ensure we're in the main thread
            if threading.current_thread() != threading.main_thread():
                self.root.after(0, lambda: self.run_interactive_selection(image, faces))
                return
            
            selected_faces = interactive_face_selection(image, faces)
            if not selected_faces:
                self.status_var.set("No faces selected for anonymization")
                self.progress.stop()
                self.process_btn.config(state='normal')
                return
            
            # Continue with processing using selected faces
            self.continue_processing(image, selected_faces)
            
        except Exception as e:
            self.status_var.set(f"Error in interactive selection: {str(e)}")
            self.progress.stop()
            self.process_btn.config(state='normal')
            messagebox.showerror("Error", f"Interactive selection failed:\n{str(e)}")
    
    def continue_processing(self, image, faces):
        """Continue processing after interactive selection"""
        try:
            mode = self.mode.get()
            
            # Apply anonymization
            if mode == "blur":
                result = anonymize_blur(image, faces, self.blur_strength.get(), 
                                      self.blur_type.get(), self.blur_radius.get())
            elif mode == "pixelate":
                result = anonymize_pixelate(image, faces, self.blocks.get(), 
                                          self.pixelate_method.get(), self.pixelate_quality.get())
            elif mode == "emoji":
                emoji_path = os.path.join("emoji", self.selected_emoji.get())
                result = anonymize_emoji(image, faces, emoji_path)
            elif mode == "replace":
                result = anonymize_synthetic(image, faces, self.synthetic_dir.get(), 
                                           self.random_faces.get(), self.keep_largest.get())
            
            # Save result
            os.makedirs(os.path.dirname(self.output_path.get()), exist_ok=True)
            cv2.imwrite(self.output_path.get(), result)
            
            face_count = len(faces)
            output_file = self.output_path.get()
            
            if self.interactive.get():
                self.status_var.set(f"Processed successfully! {face_count} faces selected and anonymized")
                success_msg = f"Processing complete!\n{face_count} faces selected and anonymized.\nSaved to: {output_file}"
            else:
                self.status_var.set(f"Processed successfully! {face_count} faces detected and anonymized")
                success_msg = f"Processing complete!\n{face_count} faces detected and anonymized.\nSaved to: {output_file}"
            
            # Show success message first
            messagebox.showinfo("Success", success_msg)
            
            # Automatically open the processed image
            self.open_processed_image(output_file)
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
        finally:
            self.progress.stop()
            self.process_btn.config(state='normal')
    
    def open_processed_image(self, image_path):
        """Automatically open the processed image in the default image viewer"""
        try:
            import platform
            import subprocess
            
            # Ask user if they want to open the image
            open_image = messagebox.askyesno(
                "Open Processed Image?", 
                f"Would you like to open the processed image?\n\n{os.path.basename(image_path)}\n\nThis will open it in your default image viewer."
            )
            
            if open_image:
                system = platform.system()
                if system == "Windows":
                    # Windows: use start command
                    os.startfile(image_path)
                elif system == "Darwin":  # macOS
                    # macOS: use open command
                    subprocess.run(["open", image_path])
                else:  # Linux and others
                    # Linux: use xdg-open
                    subprocess.run(["xdg-open", image_path])
                
                self.status_var.set(f"✅ Opened: {os.path.basename(image_path)}")
                
        except Exception as e:
            # If opening fails, show the file location instead
            error_msg = f"Could not automatically open the image.\n\nImage saved to:\n{image_path}\n\nError: {str(e)}"
            messagebox.showwarning("Cannot Open Image", error_msg)
    
    # Text Input Methods
    def open_text_file(self):
        """Open a text file"""
        filename = filedialog.askopenfilename(
            title="Open Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.text_input_area.delete("1.0", tk.END)
                self.text_input_area.insert("1.0", content)
                self.update_text_stats()
                messagebox.showinfo("Success", f"File loaded: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")
    
    def save_text_file(self):
        """Save text input to file"""
        text = self.text_input_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"File saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def clear_text_input(self):
        """Clear the text input area"""
        self.text_input_area.delete("1.0", tk.END)
        self.update_text_stats()
    
    def copy_to_output(self):
        """Copy input text to output area"""
        input_text = self.text_input_area.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "No input text to copy!")
            return
        
        self.text_output_area.delete("1.0", tk.END)
        self.text_output_area.insert("1.0", input_text)
        self.update_text_stats()
        messagebox.showinfo("Success", "Text copied to output area!")
    
    def transform_text(self):
        """Apply various transformations to the input text"""
        input_text = self.text_input_area.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "No input text to transform!")
            return
        
        # Simple transformations - reverse the text
        transformed_text = input_text[::-1]
        
        self.text_output_area.delete("1.0", tk.END)
        self.text_output_area.insert("1.0", f"Reversed Text:\n{transformed_text}")
        self.update_text_stats()
        messagebox.showinfo("Success", "Text transformed (reversed) and copied to output!")
    
    def make_uppercase(self):
        """Convert input text to uppercase in output"""
        input_text = self.text_input_area.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "No input text to convert!")
            return
        
        uppercase_text = input_text.upper()
        
        self.text_output_area.delete("1.0", tk.END)
        self.text_output_area.insert("1.0", uppercase_text)
        self.update_text_stats()
        messagebox.showinfo("Success", "Text converted to uppercase!")
    
    def make_lowercase(self):
        """Convert input text to lowercase in output"""
        input_text = self.text_input_area.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Warning", "No input text to convert!")
            return
        
        lowercase_text = input_text.lower()
        
        self.text_output_area.delete("1.0", tk.END)
        self.text_output_area.insert("1.0", lowercase_text)
        self.update_text_stats()
        messagebox.showinfo("Success", "Text converted to lowercase!")
    
    def save_output_file(self):
        """Save output text to file"""
        text = self.text_output_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No output text to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Output Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"Output file saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save output file: {str(e)}")
    
    def clear_text_output(self):
        """Clear the text output area"""
        self.text_output_area.delete("1.0", tk.END)
        self.update_text_stats()
    
    def show_word_count(self):
        """Show detailed word count statistics for both input and output"""
        input_text = self.text_input_area.get("1.0", tk.END).strip()
        output_text = self.text_output_area.get("1.0", tk.END).strip()
        
        if not input_text and not output_text:
            messagebox.showinfo("Word Count", "No text to analyze!")
            return
        
        def get_stats(text, label):
            if not text:
                return f"{label}: No text"
            
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            word_count = len(text.split())
            line_count = len(text.split('\n'))
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            
            return f"""{label}:
Characters (with spaces): {char_count}
Characters (without spaces): {char_count_no_spaces}
Words: {word_count}
Lines: {line_count}
Paragraphs: {paragraph_count}"""
        
        input_stats = get_stats(input_text, "INPUT TEXT")
        output_stats = get_stats(output_text, "OUTPUT TEXT")
        
        stats_message = f"""Text Statistics:

{input_stats}

{output_stats}"""
        
        messagebox.showinfo("Text Statistics", stats_message)
    
    def update_text_stats(self, event=None):
        """Update the real-time text statistics for both input and output"""
        try:
            input_text = self.text_input_area.get("1.0", tk.END).strip()
            output_text = self.text_output_area.get("1.0", tk.END).strip()
            
            input_char_count = len(input_text)
            input_word_count = len(input_text.split()) if input_text else 0
            
            output_char_count = len(output_text)
            output_word_count = len(output_text.split()) if output_text else 0
            
            self.text_stats.config(text=f"Input - Characters: {input_char_count} | Words: {input_word_count} | Output - Characters: {output_char_count} | Words: {output_word_count}")
        except:
            pass

def main():
    root = tk.Tk()
    app = MultiFunctionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
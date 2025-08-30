#!/usr/bin/env python3
"""
Enhanced Multi-Function GUI Application with Modern UI
======================================================

A modern, beautiful tabbed GUI application that provides:
1. Face Anonymizer - Image privacy protection with blur, pixelate, emoji, replace modes
2. Speech to Text - Voice recording and transcription using OpenAI Whisper
3. Simple Text Input - Advanced text processing interface

Enhanced with CustomTkinter for modern appearance and better user experience.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from tkinter import scrolledtext
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
# from tkinter_tooltip import ToolTip  # Optional tooltip library

# Import our main anonymizer
from face_anonymizer_main import detect_faces_yolo, anonymize_blur, anonymize_pixelate, anonymize_emoji, anonymize_synthetic, interactive_face_selection
from ultralytics import YOLO

# Set the appearance mode and color theme
ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class ModernMultiFunctionGUI:
    def __init__(self):
        # Create main window
        self.root = ctk.CTk()
        self.root.title("🎯 Multi-Function Studio - Face Anonymizer | Speech to Text | Text Processing")
        self.root.geometry("900x700")
        
        # Initialize all variables first
        self._init_variables()
        
        # Create header
        self.create_header()
        
        # Create the main notebook (tabbed interface)
        self.create_tabview()
        
        # Create tabs
        self.create_face_anonymizer_tab()
        self.create_speech_to_text_tab()
        self.create_text_input_tab()
        
        # Load YOLO model in background after UI is ready
        self.root.after(1000, lambda: threading.Thread(target=self.load_yolo_model, daemon=True).start())
        
    def _init_variables(self):
        # Face Anonymizer variables (matching original GUI)
        self.selected_image = None
        self.processed_image = None
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.mode = tk.StringVar(value="pixelate")  # Default to pixelate like original
        self.threshold = tk.DoubleVar(value=0.5)
        self.blocks = tk.IntVar(value=12)
        self.blur_strength = tk.IntVar(value=0)
        self.blur_type = tk.StringVar(value="gaussian")
        self.pixelate_method = tk.StringVar(value="cubic")  # Default to cubic like original
        self.blur_radius = tk.DoubleVar(value=0.0)
        self.pixelate_quality = tk.IntVar(value=50)
        self.selected_emoji = tk.StringVar(value="Baozou.png")
        self.emoji_choice = tk.StringVar(value="Baozou.png")  # For compatibility
        self.synthetic_dir = tk.StringVar()
        self.random_faces = tk.BooleanVar()
        self.keep_largest = tk.BooleanVar()
        self.interactive = tk.BooleanVar()
        self.model = None
        
        # Speech to Text variables
        self.recording = False
        self.audio_frames = []
        self.whisper_model = None
        self.audio_stream = None
        self.recognizer = sr.Recognizer()
        
        # Text Input variables - no special variables needed for basic text input
        
    def create_header(self):
        """Create a modern header with app title and description"""
        header_frame = ctk.CTkFrame(self.root, height=80, corner_radius=0)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Main title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="🎯 Multi-Function Studio", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(15, 5))
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame, 
            text="Advanced Face Privacy • AI Speech Recognition • Smart Text Processing",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        )
        subtitle_label.pack(pady=(0, 10))
        
    def create_tabview(self):
        """Create modern tabbed interface"""
        self.tabview = ctk.CTkTabview(self.root, width=880, height=580)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Add tabs
        self.face_tab = self.tabview.add("🖼️ Face Anonymizer")
        self.speech_tab = self.tabview.add("🎤 Speech to Text")
        self.text_tab = self.tabview.add("📝 Text Processing")
        
    def create_face_anonymizer_tab(self):
        """Create enhanced Face Anonymizer tab with all original settings"""
        # Main scrollable frame
        main_frame = ctk.CTkScrollableFrame(self.face_tab, width=840, height=520)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Welcome section
        welcome_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        welcome_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            welcome_frame,
            text="🛡️ Face Privacy Protection Tool",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(15, 5))
        
        ctk.CTkLabel(
            welcome_frame,
            text="Protect privacy in images using advanced AI face detection and anonymization",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        ).pack(pady=(0, 15))
        
        # File paths section
        paths_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        paths_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            paths_frame,
            text="📁 File Selection",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))
        
        # Input file
        input_container = ctk.CTkFrame(paths_frame, fg_color="transparent")
        input_container.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(input_container, text="Input Image/Video:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        input_row = ctk.CTkFrame(input_container, fg_color="transparent")
        input_row.pack(fill="x", pady=(5, 0))
        
        self.input_entry = ctk.CTkEntry(input_row, textvariable=self.input_path, width=400)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkButton(input_row, text="Browse", command=self.browse_input, width=80).pack(side="right")
        
        # Output file
        output_container = ctk.CTkFrame(paths_frame, fg_color="transparent")
        output_container.pack(fill="x", padx=20, pady=(15, 20))
        
        ctk.CTkLabel(output_container, text="Output Path:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        output_row = ctk.CTkFrame(output_container, fg_color="transparent")
        output_row.pack(fill="x", pady=(5, 0))
        
        self.output_entry = ctk.CTkEntry(output_row, textvariable=self.output_path, width=400)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkButton(output_row, text="Browse", command=self.browse_output, width=80).pack(side="right")
        
        # Anonymization mode section
        mode_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        mode_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            mode_frame,
            text="⚙️ Anonymization Mode",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 15))
        
        # Mode selection with modern radio buttons
        mode_selection_frame = ctk.CTkFrame(mode_frame)
        mode_selection_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        modes = [
            ("🌫️ Blur", "blur"),
            ("🔲 Pixelate", "pixelate"), 
            ("😀 Emoji", "emoji"),
            ("🤖 Synthetic Face Swap", "replace")
        ]
        
        for i, (text, value) in enumerate(modes):
            radio = ctk.CTkRadioButton(
                mode_selection_frame,
                text=text,
                variable=self.mode,
                value=value,
                command=self.create_mode_options,
                font=ctk.CTkFont(size=12)
            )
            radio.grid(row=0, column=i, padx=20, pady=5, sticky="w")
        
        # Detection threshold
        threshold_container = ctk.CTkFrame(mode_frame, fg_color="transparent")
        threshold_container.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(threshold_container, text="Detection Threshold:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        threshold_row = ctk.CTkFrame(threshold_container, fg_color="transparent")
        threshold_row.pack(fill="x", pady=(5, 0))
        
        self.threshold_slider = ctk.CTkSlider(threshold_row, from_=0.1, to=0.9, variable=self.threshold, width=300)
        self.threshold_slider.pack(side="left", padx=(0, 10))
        
        self.threshold_label = ctk.CTkLabel(threshold_row, text="0.5", font=ctk.CTkFont(size=12))
        self.threshold_label.pack(side="left")
        
        # Bind threshold update
        self.threshold.trace('w', self.update_threshold_label)
        
        # Mode-specific options frame
        self.options_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        self.options_frame.pack(fill="x", pady=(0, 20))
        
        # General options
        general_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        general_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            general_frame,
            text="🔧 General Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 15))
        
        general_options_container = ctk.CTkFrame(general_frame)
        general_options_container.pack(fill="x", padx=20, pady=(0, 20))
        
        self.interactive_checkbox = ctk.CTkCheckBox(
            general_options_container,
            text="Interactive Face Selection",
            variable=self.interactive,
            font=ctk.CTkFont(size=12)
        )
        self.interactive_checkbox.pack(anchor="w", pady=5)
        
        ctk.CTkLabel(
            general_options_container,
            text="(Click faces to select/deselect them)",
            font=ctk.CTkFont(size=10),
            text_color=("gray60", "gray40")
        ).pack(anchor="w", padx=(25, 0))
        
        self.keep_largest_checkbox = ctk.CTkCheckBox(
            general_options_container,
            text="Keep Largest Face (Main Subject)",
            variable=self.keep_largest,
            font=ctk.CTkFont(size=12)
        )
        self.keep_largest_checkbox.pack(anchor="w", pady=5)
        
        # Status and process section
        status_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        status_frame.pack(fill="x", pady=(0, 20))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Select an image to get started",
            font=ctk.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        self.status_label.pack(pady=15)
        
        # Process button
        self.process_btn = ctk.CTkButton(
            main_frame,
            text="� Process Image/Video",
            command=self.process_image,
            width=250,
            height=50,
            corner_radius=25,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.process_btn.pack(pady=20)
        
        # Create initial mode options
        self.create_mode_options()
        
    def create_speech_to_text_tab(self):
        """Create enhanced Speech to Text tab"""
        # Main frame
        main_frame = ctk.CTkFrame(self.speech_tab, corner_radius=15)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        ctk.CTkLabel(
            main_frame,
            text="🎤 AI-Powered Speech Recognition",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            main_frame,
            text="Unlimited recording • Offline processing • Superior accuracy with OpenAI Whisper",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        ).pack(pady=(0, 20))
        
        # Status section
        status_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        status_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.speech_status = ctk.CTkLabel(
            status_frame,
            text="🔄 Loading Whisper AI model...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("orange", "yellow")
        )
        self.speech_status.pack()
        
        # Recording controls
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Record button
        self.record_btn = ctk.CTkButton(
            controls_frame,
            text="🎤 Start Recording",
            command=self.toggle_recording,
            width=200,
            height=50,
            corner_radius=25,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.record_btn.pack(pady=20)
        
        # Text output area
        text_frame = ctk.CTkFrame(main_frame)
        text_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            text_frame,
            text="📝 Transcription Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10), anchor="w")
        
        # Use regular tkinter ScrolledText for better text handling
        self.speech_text = scrolledtext.ScrolledText(
            text_frame,
            width=70,
            height=15,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#2B2B2B",
            fg="white",
            insertbackground="white",
            selectbackground="#404040"
        )
        self.speech_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Action buttons
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        button_config = {
            "width": 140,
            "height": 35,
            "corner_radius": 15,
            "font": ctk.CTkFont(size=12, weight="bold")
        }
        
        ctk.CTkButton(action_frame, text="💾 Save", command=self.save_speech_text, **button_config).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(action_frame, text="📋 Copy", command=self.copy_speech_text, **button_config).pack(side="left", padx=5, pady=10)
        ctk.CTkButton(action_frame, text="🗑 Clear", command=self.clear_speech_text, **button_config).pack(side="left", padx=5, pady=10)
        
        # Load Whisper model in background after UI is ready
        self.root.after(2000, lambda: threading.Thread(target=self.load_whisper_model, daemon=True).start())
        
    def create_text_input_tab(self):
        """Create enhanced Text Processing tab"""
        # Main frame
        main_frame = ctk.CTkFrame(self.text_tab, corner_radius=15)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        ctk.CTkLabel(
            main_frame,
            text="📝 Advanced Text Processing",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            main_frame,
            text="Smart text manipulation • File operations • Real-time analytics",
            font=ctk.CTkFont(size=14),
            text_color=("gray60", "gray40")
        ).pack(pady=(0, 20))
        
        # Text areas frame
        text_areas_frame = ctk.CTkFrame(main_frame)
        text_areas_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        # Output area (top)
        output_label = ctk.CTkLabel(
            text_areas_frame,
            text="📤 Output Area",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        output_label.pack(fill="x", padx=15, pady=(15, 5))
        
        self.text_output_area = scrolledtext.ScrolledText(
            text_areas_frame,
            width=70,
            height=12,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#1A1A1A",
            fg="white",
            insertbackground="white",
            selectbackground="#404040"
        )
        self.text_output_area.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # Input area (bottom)
        input_label = ctk.CTkLabel(
            text_areas_frame,
            text="📥 Input Area",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        input_label.pack(fill="x", padx=15, pady=(10, 5))
        
        self.text_input_area = scrolledtext.ScrolledText(
            text_areas_frame,
            width=70,
            height=12,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#2B2B2B",
            fg="white",
            insertbackground="white",
            selectbackground="#404040"
        )
        self.text_input_area.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Controls frame
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # Processing buttons
        process_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        process_frame.pack(pady=15)
        
        button_config = {
            "width": 120,
            "height": 32,
            "corner_radius": 15,
            "font": ctk.CTkFont(size=11, weight="bold")
        }
        
        # Row 1 - Main operations
        row1 = ctk.CTkFrame(process_frame, fg_color="transparent")
        row1.pack(pady=(0, 8))
        
        ctk.CTkButton(row1, text="➡️ Copy to Output", command=self.copy_to_output, **button_config).pack(side="left", padx=3)
        
        # Row 2 - File operations
        row2 = ctk.CTkFrame(process_frame, fg_color="transparent")
        row2.pack(pady=(0, 8))
        
        ctk.CTkButton(row2, text="📁 Open File", command=self.open_text_file, **button_config).pack(side="left", padx=3)
        ctk.CTkButton(row2, text="💾 Save Input", command=self.save_text_file, **button_config).pack(side="left", padx=3)
        ctk.CTkButton(row2, text="💾 Save Output", command=self.save_output_file, **button_config).pack(side="left", padx=3)
        
        # Row 3 - Clear operations
        row3 = ctk.CTkFrame(process_frame, fg_color="transparent")
        row3.pack()
        
        ctk.CTkButton(row3, text="🗑 Clear Input", command=self.clear_text_input, **button_config).pack(side="left", padx=3)
        ctk.CTkButton(row3, text="🗑 Clear Output", command=self.clear_text_output, **button_config).pack(side="left", padx=3)
        
        # Stats display
        self.text_stats = ctk.CTkLabel(
            main_frame,
            text="Input: 0 chars, 0 words | Output: 0 chars, 0 words",
            font=ctk.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        self.text_stats.pack(pady=(0, 15))
        
        # Bind events for real-time stats
        self.text_input_area.bind('<KeyRelease>', self.update_text_stats)
        self.text_input_area.bind('<Button-1>', self.update_text_stats)
        self.text_output_area.bind('<KeyRelease>', self.update_text_stats)
        self.text_output_area.bind('<Button-1>', self.update_text_stats)

    # Include all the original methods with minimal changes
    def load_yolo_model(self):
        """Load YOLO model in background"""
        def load_model():
            try:
                def update_status_safe(message):
                    self.root.after(0, lambda: self.status_label.configure(text=message))
                
                update_status_safe("Loading AI face detection model...")
                self.model = YOLO('yolov12l-face.pt')
                update_status_safe("✅ Face detection model loaded successfully!")
            except Exception as e:
                def update_error():
                    self.status_label.configure(text=f"⚠️ Warning: {str(e)}")
                self.root.after(0, update_error)
        
        load_model()
    
    def update_status(self, message):
        """Update status label safely from any thread"""
        if hasattr(self, 'status_label'):
            self.root.after(0, lambda: self.status_label.configure(text=message))
    
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.selected_image = file_path
            self.update_status(f"Selected: {os.path.basename(file_path)}")
            self.process_btn.configure(state="normal")
            
    def create_mode_options(self):
        """Create mode-specific options matching the original GUI"""
        # Clear existing mode-specific widgets
        for widget in self.options_frame.winfo_children():
            if hasattr(widget, '_mode_specific'):
                widget.destroy()
        
        # Header
        ctk.CTkLabel(
            self.options_frame,
            text="🔧 Mode Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 15))
        
        mode = self.mode.get()
        
        # Create mode-specific options frame
        mode_options_frame = ctk.CTkFrame(self.options_frame)
        mode_options_frame.pack(fill="x", padx=20, pady=(0, 15))
        mode_options_frame._mode_specific = True
        
        if mode == "blur":
            # Blur type selection
            blur_type_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            blur_type_container.pack(fill="x", padx=15, pady=15)
            
            ctk.CTkLabel(blur_type_container, text="Blur Type:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            blur_type_frame = ctk.CTkFrame(blur_type_container, fg_color="transparent")
            blur_type_frame.pack(fill="x", pady=(5, 0))
            
            blur_types = [("Gaussian", "gaussian"), ("Box", "box"), ("Median", "median")]
            for i, (text, value) in enumerate(blur_types):
                ctk.CTkRadioButton(blur_type_frame, text=text, variable=self.blur_type, value=value, font=ctk.CTkFont(size=11)).pack(side="left", padx=15)
            
            # Blur strength
            blur_strength_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            blur_strength_container.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(blur_strength_container, text="Blur Strength (0=auto):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            strength_row = ctk.CTkFrame(blur_strength_container, fg_color="transparent")
            strength_row.pack(fill="x", pady=(5, 0))
            
            self.blur_strength_slider = ctk.CTkSlider(strength_row, from_=0, to=50, variable=self.blur_strength, width=250)
            self.blur_strength_slider.pack(side="left", padx=(0, 10))
            
            self.blur_strength_label = ctk.CTkLabel(strength_row, text="0", font=ctk.CTkFont(size=12))
            self.blur_strength_label.pack(side="left")
            
            self.blur_strength.trace('w', self.update_blur_strength_label)
            
            # Blur radius
            blur_radius_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            blur_radius_container.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(blur_radius_container, text="Blur Radius (σ):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            radius_row = ctk.CTkFrame(blur_radius_container, fg_color="transparent")
            radius_row.pack(fill="x", pady=(5, 0))
            
            self.blur_radius_slider = ctk.CTkSlider(radius_row, from_=0.0, to=5.0, variable=self.blur_radius, width=250)
            self.blur_radius_slider.pack(side="left", padx=(0, 10))
            
            self.blur_radius_label = ctk.CTkLabel(radius_row, text="0.0", font=ctk.CTkFont(size=12))
            self.blur_radius_label.pack(side="left")
            
            self.blur_radius.trace('w', self.update_blur_radius_label)
            
        elif mode == "pixelate":
            # Pixelation method selection
            pixelate_method_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            pixelate_method_container.pack(fill="x", padx=15, pady=15)
            
            ctk.CTkLabel(pixelate_method_container, text="Pixelation Method:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            method_frame = ctk.CTkFrame(pixelate_method_container, fg_color="transparent")
            method_frame.pack(fill="x", pady=(5, 0))
            
            pixelate_methods = [("Nearest", "nearest"), ("Linear", "linear"), ("Cubic", "cubic")]
            for i, (text, value) in enumerate(pixelate_methods):
                ctk.CTkRadioButton(method_frame, text=text, variable=self.pixelate_method, value=value, font=ctk.CTkFont(size=11)).pack(side="left", padx=15)
            
            # Block size
            block_size_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            block_size_container.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(block_size_container, text="Block Size:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            block_row = ctk.CTkFrame(block_size_container, fg_color="transparent")
            block_row.pack(fill="x", pady=(5, 0))
            
            self.blocks_slider = ctk.CTkSlider(block_row, from_=4, to=20, variable=self.blocks, width=250)
            self.blocks_slider.pack(side="left", padx=(0, 10))
            
            self.blocks_label = ctk.CTkLabel(block_row, text="12", font=ctk.CTkFont(size=12))
            self.blocks_label.pack(side="left")
            
            self.blocks.trace('w', self.update_blocks_label)
            
            # Pixelation quality
            quality_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            quality_container.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(quality_container, text="Quality (%):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            quality_row = ctk.CTkFrame(quality_container, fg_color="transparent")
            quality_row.pack(fill="x", pady=(5, 0))
            
            self.quality_slider = ctk.CTkSlider(quality_row, from_=10, to=100, variable=self.pixelate_quality, width=250)
            self.quality_slider.pack(side="left", padx=(0, 10))
            
            self.quality_label = ctk.CTkLabel(quality_row, text="50", font=ctk.CTkFont(size=12))
            self.quality_label.pack(side="left")
            
            self.pixelate_quality.trace('w', self.update_quality_label)
            
        elif mode == "emoji":
            # Emoji selection
            emoji_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            emoji_container.pack(fill="x", padx=15, pady=15)
            
            ctk.CTkLabel(emoji_container, text="Select Emoji:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            
            # Available emoji options
            emoji_options = [
                ("Baozou", "Baozou.png"),
                ("Cat", "Cat.png"), 
                ("Doge", "Doge.png"),
                ("Shrek", "Shrek.png"),
                ("Yao", "Yao.png")
            ]
            
            emoji_selection_frame = ctk.CTkFrame(emoji_container, fg_color="transparent")
            emoji_selection_frame.pack(fill="x", pady=(5, 0))
            
            # Create radio buttons for emoji selection
            for i, (text, value) in enumerate(emoji_options):
                ctk.CTkRadioButton(emoji_selection_frame, text=text, variable=self.selected_emoji, value=value, font=ctk.CTkFont(size=11)).pack(side="left", padx=10)
            
            # Preview section
            preview_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            preview_container.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(preview_container, text="Preview:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            self.emoji_preview_label = ctk.CTkLabel(preview_container, text="")
            self.emoji_preview_label.pack(anchor="w", pady=(5, 0))
            
            # Load and display emoji preview
            self.selected_emoji.trace('w', lambda *args: self.update_emoji_preview())
            self.update_emoji_preview()
            
        elif mode == "replace":
            # Synthetic faces folder
            synthetic_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            synthetic_container.pack(fill="x", padx=15, pady=15)
            
            ctk.CTkLabel(synthetic_container, text="Synthetic Faces Folder:", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
            synthetic_row = ctk.CTkFrame(synthetic_container, fg_color="transparent")
            synthetic_row.pack(fill="x", pady=(5, 0))
            
            self.synthetic_entry = ctk.CTkEntry(synthetic_row, textvariable=self.synthetic_dir, width=300)
            self.synthetic_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
            
            ctk.CTkButton(synthetic_row, text="Browse", command=self.browse_synthetic_dir, width=80).pack(side="right")
            
            # Random faces option
            random_container = ctk.CTkFrame(mode_options_frame, fg_color="transparent")
            random_container.pack(fill="x", padx=15, pady=10)
            
            self.random_faces_checkbox = ctk.CTkCheckBox(
                random_container,
                text="Use Random Synthetic Faces",
                variable=self.random_faces,
                font=ctk.CTkFont(size=12)
            )
            self.random_faces_checkbox.pack(anchor="w")
    
    def process_image(self):
        """Process the selected image using the same logic as original GUI"""
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
        
        try:
            self.update_status("Processing...")
            self.process_btn.configure(text="Processing...", state="disabled")
            
            # Run processing in background thread
            threading.Thread(target=self._process_image_background, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.update_status("❌ Processing failed")
            self.process_btn.configure(text="🚀 Process Image/Video", state="normal")
    
    def _process_image_background(self):
        """Background image processing using original GUI logic"""
        try:
            # Load image
            image = cv2.imread(self.input_path.get())
            if image is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load input image"))
                return
            
            # Detect faces using YOLO
            faces = detect_faces_yolo(image, self.model, self.threshold.get())
            if not faces:
                self.root.after(0, lambda: self.update_status("No faces detected"))
                self.root.after(0, lambda: messagebox.showinfo("Info", "No faces detected in the image."))
                return
            
            # Interactive face selection if enabled
            if self.interactive.get():
                self.root.after(0, lambda: self.run_interactive_selection(image, faces))
                return
            
            # Continue with processing
            self.continue_processing(image, faces)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, lambda: self.update_status("❌ Processing failed"))
        finally:
            self.root.after(0, lambda: self._reset_process_button())
    
    def run_interactive_selection(self, image, faces):
        """Run interactive face selection in main thread"""
        try:
            selected_faces = interactive_face_selection(image, faces)
            if not selected_faces:
                self.update_status("No faces selected for anonymization")
                self._reset_process_button()
                return
            
            # Continue with processing using selected faces
            self.continue_processing(image, selected_faces)
            
        except Exception as e:
            self.update_status(f"Error in interactive selection: {str(e)}")
            self._reset_process_button()
            messagebox.showerror("Error", f"Interactive selection failed: {str(e)}")
    
    def continue_processing(self, image, faces):
        """Continue processing after interactive selection"""
        try:
            mode = self.mode.get()
            
            # Apply anonymization using same methods as original
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
                self.update_status(f"✅ Processed! {face_count} faces selected and anonymized")
                success_msg = f"Processing complete!\n{face_count} faces selected and anonymized.\nSaved to: {output_file}"
            else:
                self.update_status(f"✅ Processed! {face_count} faces detected and anonymized")
                success_msg = f"Processing complete!\n{face_count} faces detected and anonymized.\nSaved to: {output_file}"
            
            # Show success message first
            messagebox.showinfo("Success", success_msg)
            
            # Automatically open the processed image
            self.open_processed_image(output_file)
            
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self._reset_process_button()
    
    def _reset_process_button(self):
        """Reset process button state"""
        self.process_btn.configure(text="🚀 Process Image/Video", state="normal")
    
    # Speech to Text methods
    def load_whisper_model(self):
        """Load Whisper model in background"""
        try:
            def update_status_safe(text, color):
                self.root.after(0, lambda: self.speech_status.configure(text=text, text_color=color))
            
            update_status_safe("🔄 Loading Whisper AI model...", ("orange", "yellow"))
            self.whisper_model = whisper.load_model("base")
            update_status_safe("✅ Whisper model loaded. Ready to record!", ("green", "lightgreen"))
            self.root.after(0, lambda: self.record_btn.configure(state="normal"))
        except Exception as e:
            def update_error():
                self.speech_status.configure(text=f"❌ Failed to load Whisper: {str(e)}", text_color=("red", "lightcoral"))
            self.root.after(0, update_error)
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio"""
        try:
            self.recording = True
            self.audio_frames = []
            self.record_btn.configure(text="⏹ Stop Recording", fg_color=("red", "darkred"))
            self.speech_status.configure(text="🔴 Recording... Click 'Stop' when finished", text_color=("red", "lightcoral"))
            
            # Start recording in background thread
            threading.Thread(target=self._record_audio, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
            self.recording = False
            self._reset_record_button()
    
    def _record_audio(self):
        """Record audio in background thread"""
        try:
            audio_format = pyaudio.paInt16
            channels = 1
            rate = 16000  # Optimal for Whisper
            chunk = 1024
            
            p = pyaudio.PyAudio()
            stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
            
            while self.recording:
                data = stream.read(chunk)
                self.audio_frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {str(e)}"))
            self.root.after(0, lambda: self._reset_record_button())
    
    def stop_recording(self):
        """Stop recording and process audio"""
        self.recording = False
        self.record_btn.configure(state="disabled", text="🔄 Processing...")
        self.speech_status.configure(text="🔄 Processing audio with Whisper AI...", text_color=("orange", "yellow"))
        
        # Process audio in background thread
        threading.Thread(target=self._process_audio, daemon=True).start()
    
    def _process_audio(self):
        """Process recorded audio with Whisper"""
        try:
            if not self.audio_frames:
                self.root.after(0, lambda: messagebox.showwarning("Warning", "No audio recorded"))
                self.root.after(0, lambda: self._reset_record_button())
                return
            
            # Convert audio frames to numpy array
            audio_data = b''.join(self.audio_frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_array)
            text = result["text"].strip()
            
            if text:
                # Add text to the text area
                self.root.after(0, lambda: self.speech_text.insert(tk.END, f"{text}\n\n"))
                self.root.after(0, lambda: self.speech_text.see(tk.END))
                self.root.after(0, lambda: self.speech_status.configure(text="✅ Transcription complete! Ready to record", text_color=("green", "lightgreen")))
            else:
                self.root.after(0, lambda: self.speech_status.configure(text="⚠️ No speech detected. Try again", text_color=("orange", "yellow")))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Transcription failed: {str(e)}"))
            self.root.after(0, lambda: self.speech_status.configure(text="❌ Transcription failed", text_color=("red", "lightcoral")))
        
        finally:
            self.root.after(0, lambda: self._reset_record_button())
    
    def _reset_record_button(self):
        """Reset record button state"""
        self.record_btn.configure(text="🎤 Start Recording", state="normal", fg_color=("#1f538d", "#14375e"))
    
    def save_speech_text(self):
        """Save speech text to file"""
        text = self.speech_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"Text saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def copy_speech_text(self):
        """Copy speech text to clipboard"""
        text = self.speech_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to copy")
            return
            
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Success", "Text copied to clipboard")
    
    def clear_speech_text(self):
        """Clear speech text area"""
        if messagebox.askyesno("Confirm", "Clear all transcribed text?"):
            self.speech_text.delete(1.0, tk.END)
    
    # Text Input methods
    def copy_to_output(self):
        """Copy input text to output area"""
        text = self.text_input_area.get(1.0, tk.END)
        self.text_output_area.delete(1.0, tk.END)
        self.text_output_area.insert(1.0, text)
        self.update_text_stats()
    
    def open_text_file(self):
        """Open text file into input area"""
        file_path = filedialog.askopenfilename(
            title="Open Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.text_input_area.delete(1.0, tk.END)
                self.text_input_area.insert(1.0, content)
                self.update_text_stats()
                messagebox.showinfo("Success", f"File loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {str(e)}")
    
    def save_text_file(self):
        """Save input text to file"""
        text = self.text_input_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"Text saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def save_output_file(self):
        """Save output text to file"""
        text = self.text_output_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No text to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                messagebox.showinfo("Success", f"Text saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def clear_text_input(self):
        """Clear input text area"""
        if messagebox.askyesno("Confirm", "Clear input text?"):
            self.text_input_area.delete(1.0, tk.END)
            self.update_text_stats()
    
    def clear_text_output(self):
        """Clear output text area"""
        if messagebox.askyesno("Confirm", "Clear output text?"):
            self.text_output_area.delete(1.0, tk.END)
            self.update_text_stats()
    
    def update_text_stats(self, event=None):
        """Update text statistics display"""
        try:
            input_text = self.text_input_area.get(1.0, tk.END).strip()
            output_text = self.text_output_area.get(1.0, tk.END).strip()
            
            input_chars = len(input_text)
            input_words = len(input_text.split()) if input_text else 0
            
            output_chars = len(output_text)
            output_words = len(output_text.split()) if output_text else 0
            
            self.text_stats.configure(
                text=f"Input: {input_chars} chars, {input_words} words | Output: {output_chars} chars, {output_words} words"
            )
        except:
            pass  # Ignore errors during text stats update
    
    def update_threshold_label(self, *args):
        """Update threshold label"""
        try:
            value = self.threshold.get()
            self.threshold_label.configure(text=f"{value:.1f}")
        except:
            pass
    
    def update_blur_strength_label(self, *args):
        """Update blur strength label"""
        try:
            value = int(self.blur_strength.get())
            self.blur_strength_label.configure(text=str(value))
        except:
            pass
    
    def update_blur_radius_label(self, *args):
        """Update blur radius label"""
        try:
            value = self.blur_radius.get()
            self.blur_radius_label.configure(text=f"{value:.1f}")
        except:
            pass
    
    def update_blocks_label(self, *args):
        """Update blocks label"""
        try:
            value = int(self.blocks.get())
            self.blocks_label.configure(text=str(value))
        except:
            pass
    
    def update_quality_label(self, *args):
        """Update quality label"""
        try:
            value = int(self.pixelate_quality.get())
            self.quality_label.configure(text=str(value))
        except:
            pass
    
    def browse_input(self):
        """Browse for input file"""
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
            self.selected_image = filename  # For compatibility
            # Auto-generate output path
            if not self.output_path.get():
                from pathlib import Path
                input_path = Path(filename)
                output_path = input_path.parent / f"anonymized_{input_path.name}"
                self.output_path.set(str(output_path))
            
            self.update_status(f"Selected: {os.path.basename(filename)}")
            self.process_btn.configure(state="normal")
    
    def browse_output(self):
        """Browse for output file"""
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
        """Browse for synthetic faces directory"""
        directory = filedialog.askdirectory(title="Select Synthetic Faces Folder")
        if directory:
            self.synthetic_dir.set(directory)
    
    def update_emoji_preview(self):
        """Update the emoji preview image"""
        try:
            emoji_file = self.selected_emoji.get()
            emoji_path = os.path.join("emoji", emoji_file)
            
            if os.path.exists(emoji_path):
                # Load and resize emoji for preview
                from PIL import Image, ImageTk
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
                
                self.update_status(f"✅ Opened: {os.path.basename(image_path)}")
                
        except Exception as e:
            # If opening fails, show the file location instead
            error_msg = f"Could not automatically open the image.\n\nImage saved to:\n{image_path}\n\nError: {str(e)}"
            messagebox.showwarning("Cannot Open Image", error_msg)
    
def main():
    """Main function to run the application"""
    try:
        app = ModernMultiFunctionGUI()
        app.root.mainloop()
    except Exception as e:
        print(f"Failed to start application: {e}")
        import tkinter.messagebox as mb
        mb.showerror("Startup Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple launcher for the Face Anonymizer GUI
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from face_anonymizer_gui_modern import main
    print("Starting Face Anonymizer GUI...")
    main()
except ImportError as e:
    print(f"Error importing GUI: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error starting GUI: {e}")
    print("Please check that all files are present and dependencies are installed.")

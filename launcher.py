#!/usr/bin/env python3
"""
GUI Version Launcher
====================

Choose which version of the GUI to run:
1. Classic GUI (Original)
2. Modern GUI (Enhanced with CustomTkinter)
"""

import os
import subprocess
import sys

def main():
    print("🎯 Multi-Function GUI Launcher")
    print("=" * 40)
    print()
    print("Choose which version to run:")
    print("1. 📼 Classic GUI (Original tkinter)")
    print("2. ✨ Modern GUI (Enhanced with CustomTkinter)")
    print("3. ❓ What's New in the Enhanced GUI")
    print("4. ❌ Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()
        
        if choice == "1":
            print("\n🚀 Launching Classic GUI...")
            try:
                subprocess.run([sys.executable, "face_anonymizer_gui.py"])
            except FileNotFoundError:
                print("❌ Classic GUI file not found!")
            break
            
        elif choice == "2":
            print("\n✨ Launching Modern GUI...")
            try:
                subprocess.run([sys.executable, "face_anonymizer_gui_modern.py"])
            except FileNotFoundError:
                print("❌ Modern GUI file not found!")
            break
            
        elif choice == "3":
            show_whats_new()
            
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def show_whats_new():
    """Show what's new in the enhanced GUI"""
    print("\n" + "="*60)
    print("✨ WHAT'S NEW IN THE ENHANCED GUI")
    print("="*60)
    print()
    print("🎨 VISUAL ENHANCEMENTS:")
    print("  • Modern dark theme with professional appearance")
    print("  • Rounded buttons and smooth animations")
    print("  • Card-based layout with better organization")
    print("  • Real-time value display on all sliders")
    print()
    print("🚀 NEW FEATURES:")
    print("  • Automatic image opening after processing")
    print("  • Enhanced error handling and user feedback")
    print("  • Thread-safe operations (no UI freezing)")
    print("  • Improved Whisper speech recognition")
    print()
    print("⚙️ ALL ORIGINAL SETTINGS PRESERVED:")
    print("  • Complete feature parity with original GUI")
    print("  • All anonymization modes and settings")
    print("  • Interactive face selection")
    print("  • Same processing algorithms")
    print()
    print("🎯 WORKFLOW IMPROVEMENTS:")
    print("  • Process image → Success message → Auto-open result")
    print("  • User decides whether to open the processed image")
    print("  • Better status indicators and progress feedback")
    print()
    print("="*60)
    print("Press Enter to continue...")
    input()

if __name__ == "__main__":
    main()

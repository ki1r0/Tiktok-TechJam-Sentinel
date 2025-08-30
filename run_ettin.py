import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import warnings
warnings.filterwarnings("ignore")

class EttinDetector:
    def __init__(self, model_path="pii_ettin_encoder_1b_v1/final"):
        """Initialize Ettin PII detector"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 Loading Ettin PII Detection Model...")
        print(f"   Model Path: {model_path}")
        print(f"   Device: {self.device}")
        
        try:
            # Load tokenizer and model from your fine-tuned model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create NER pipeline for easier inference
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",  # Aggregates subword tokens
                device=0 if self.device.type == "cuda" else -1
            )
            
            print(f"   ✅ Ettin model loaded successfully")
            print(f"   📊 Model Info:")
            print(f"      Architecture: {self.model.config.architectures[0]}")
            print(f"      Classes: {len(self.model.config.id2label)} PII types")
            print(f"      Labels: {list(self.model.config.id2label.values())[:10]}...")
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise

    def predict_pii(self, text, threshold=0.5):
        """Predict PII labels using Ettin model"""
        
        # Use pipeline for inference
        ner_results = self.ner_pipeline(text)
        
        # Process results
        pii_found = set()
        word_predictions = []
        
        # Track which parts of text have been labeled
        labeled_spans = []
        
        for result in ner_results:
            if result['score'] >= threshold:
                # Extract entity info
                entity_group = result['entity_group']
                word = result['word']
                confidence = result['score']
                start = result['start']
                end = result['end']
                
                pii_found.add(entity_group)
                labeled_spans.append((start, end, entity_group, word, confidence))
        
        # Sort by start position
        labeled_spans.sort(key=lambda x: x[0])
        
        # Create word-level predictions by splitting text
        words = text.split()
        text_pos = 0
        
        for word in words:
            word_start = text.find(word, text_pos)
            word_end = word_start + len(word)
            text_pos = word_end
            
            # Check if this word overlaps with any labeled span
            word_label = 'O'
            word_confidence = 1.0
            
            for span_start, span_end, label, span_word, confidence in labeled_spans:
                # Check overlap
                if word_start < span_end and word_end > span_start:
                    word_label = label
                    word_confidence = confidence
                    break
            
            word_predictions.append({
                'word': word,
                'label': word_label,
                'confidence': word_confidence
            })
        
        return ner_results, pii_found, word_predictions

    def format_results(self, text, ner_results, pii_found, word_predictions):
        """Format prediction results nicely"""
        print(f"\n{'='*80}")
        print(f"📝 TEXT: {text}")
        print(f"{'='*80}")
        
        if not pii_found:
            print("✅ NO PII DETECTED")
            return
        
        print(f"🤖 PII DETECTED: {len(pii_found)} types found")
        print(f"   Types: {', '.join(sorted(pii_found))}")
        
        # Show word-level predictions
        print(f"\n🔤 WORD-LEVEL PREDICTIONS:")
        for word_pred in word_predictions:
            if word_pred['label'] != 'O':
                print(f"   🔍 {word_pred['label']}: '{word_pred['word']}' (conf: {word_pred['confidence']:.3f})")
        
        # Show detailed entity-level predictions from pipeline
        print(f"\n📋 ENTITY-LEVEL PREDICTIONS:")
        for result in ner_results:
            if result['score'] >= 0.5:
                print(f"   🔍 {result['entity_group']}: '{result['word']}' "
                      f"(conf: {result['score']:.3f}) [pos: {result['start']}-{result['end']}]")

    def interactive_mode(self):
        """Run the model in interactive mode for testing"""
        print(f"\n{'='*80}")
        print("🎯 INTERACTIVE PII DETECTION MODE")
        print("Enter text to analyze (type 'quit' to exit)")
        print(f"{'='*80}")
        
        while True:
            try:
                text = input("\n📝 Enter text to analyze: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("❌ Please enter some text")
                    continue
                
                # Get predictions
                ner_results, pii_found, word_predictions = self.predict_pii(text)
                
                # Format and display results
                self.format_results(text, ner_results, pii_found, word_predictions)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def create_pii_gui():
    """Create a standalone PII detection GUI"""
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
    import customtkinter as ctk
    
    # Set appearance
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    class PIIDetectorGUI:
        def __init__(self):
            self.root = ctk.CTk()
            self.root.title("🔍 PII Detection Tool - Ettin Encoder")
            self.root.geometry("900x700")
            
            # Initialize detector
            self.detector = None
            self.threshold = tk.DoubleVar(value=0.5)
            
            self.create_ui()
            self.load_model()
            
        def create_ui(self):
            # Header
            header = ctk.CTkLabel(
                self.root,
                text="🔍 PII Detection Tool",
                font=ctk.CTkFont(size=24, weight="bold")
            )
            header.pack(pady=20)
            
            # Model status
            self.status_label = ctk.CTkLabel(
                self.root,
                text="🔄 Loading Ettin PII detection model...",
                font=ctk.CTkFont(size=14),
                text_color=("orange", "yellow")
            )
            self.status_label.pack(pady=10)
            
            # Input area
            input_frame = ctk.CTkFrame(self.root)
            input_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            ctk.CTkLabel(
                input_frame,
                text="📝 Enter text to analyze for PII:",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(anchor="w", padx=15, pady=(15, 5))
            
            self.text_input = scrolledtext.ScrolledText(
                input_frame,
                width=70,
                height=15,
                wrap=tk.WORD,
                font=("Consolas", 11),
                bg="#2B2B2B",
                fg="white",
                insertbackground="white",
                selectbackground="#404040"
            )
            self.text_input.pack(fill="both", expand=True, padx=15, pady=(0, 15))
            
            # Controls
            controls_frame = ctk.CTkFrame(self.root)
            controls_frame.pack(fill="x", padx=20, pady=10)
            
            # Threshold control
            threshold_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
            threshold_frame.pack(fill="x", padx=15, pady=10)
            
            ctk.CTkLabel(threshold_frame, text="Detection Threshold:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
            
            threshold_slider = ctk.CTkSlider(threshold_frame, from_=0.1, to=0.9, variable=self.threshold, width=200)
            threshold_slider.pack(side="left", padx=(10, 10))
            
            self.threshold_label = ctk.CTkLabel(threshold_frame, text="0.5", font=ctk.CTkFont(size=12))
            self.threshold_label.pack(side="left")
            
            self.threshold.trace('w', self.update_threshold_label)
            
            # Buttons
            button_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
            button_frame.pack(pady=15)
            
            self.analyze_btn = ctk.CTkButton(
                button_frame,
                text="🔍 Analyze for PII",
                command=self.analyze_text,
                width=150,
                height=40,
                state="disabled"
            )
            self.analyze_btn.pack(side="left", padx=10)
            
            ctk.CTkButton(
                button_frame,
                text="🗑 Clear",
                command=self.clear_text,
                width=100,
                height=40
            ).pack(side="left", padx=10)
            
            ctk.CTkButton(
                button_frame,
                text="📋 Paste",
                command=self.paste_text,
                width=100,
                height=40
            ).pack(side="left", padx=10)
        
        def load_model(self):
            """Load the PII detection model"""
            def load_in_background():
                try:
                    self.detector = EttinDetector()
                    self.root.after(0, lambda: self.status_label.configure(
                        text="✅ Ettin PII detection model loaded successfully!",
                        text_color=("green", "lightgreen")
                    ))
                    self.root.after(0, lambda: self.analyze_btn.configure(state="normal"))
                except Exception as e:
                    self.root.after(0, lambda: self.status_label.configure(
                        text=f"❌ Failed to load model: {str(e)}",
                        text_color=("red", "lightcoral")
                    ))
            
            import threading
            threading.Thread(target=load_in_background, daemon=True).start()
        
        def update_threshold_label(self, *args):
            value = self.threshold.get()
            self.threshold_label.configure(text=f"{value:.1f}")
        
        def analyze_text(self):
            text = self.text_input.get(1.0, tk.END).strip()
            if not text:
                messagebox.showwarning("Warning", "Please enter some text to analyze")
                return
            
            if self.detector is None:
                messagebox.showerror("Error", "PII detection model not loaded")
                return
            
            try:
                # Run PII detection
                ner_results, pii_found, word_predictions = self.detector.predict_pii(text, self.threshold.get())
                
                # Show results
                self.show_results(text, ner_results, pii_found, word_predictions)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to analyze text: {str(e)}")
        
        def show_results(self, text, ner_results, pii_found, word_predictions):
            # Create results window
            results_window = ctk.CTkToplevel(self.root)
            results_window.title("🔍 PII Detection Results")
            results_window.geometry("800x600")
            results_window.transient(self.root)
            
            # Main frame
            main_frame = ctk.CTkScrollableFrame(results_window, width=760, height=560)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Summary
            if pii_found:
                summary_text = f"⚠️ PII DETECTED: {len(pii_found)} types found"
                summary_color = ("red", "lightcoral")
                pii_types = ", ".join(sorted(pii_found))
            else:
                summary_text = "✅ NO PII DETECTED"
                summary_color = ("green", "lightgreen")
                pii_types = "No sensitive information found"
            
            ctk.CTkLabel(
                main_frame,
                text=summary_text,
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=summary_color
            ).pack(pady=10)
            
            ctk.CTkLabel(
                main_frame,
                text=f"Types: {pii_types}",
                font=ctk.CTkFont(size=12),
                text_color=("gray60", "gray40")
            ).pack(pady=(0, 15))
            
            # Word-level results (Primary display)
            if pii_found:
                ctk.CTkLabel(
                    main_frame,
                    text="🔤 Word-Level PII Detection:",
                    font=ctk.CTkFont(size=16, weight="bold")
                ).pack(anchor="w", pady=(10, 10))
                
                # Show word-level predictions
                word_results_frame = ctk.CTkFrame(main_frame)
                word_results_frame.pack(fill="x", pady=(0, 15))
                
                pii_words_found = False
                for word_pred in word_predictions:
                    if word_pred['label'] != 'O':
                        pii_words_found = True
                        word_item = ctk.CTkFrame(word_results_frame)
                        word_item.pack(fill="x", padx=10, pady=5)
                        
                        # Word and PII label
                        ctk.CTkLabel(
                            word_item,
                            text=f"🚨 Word: '{word_pred['word']}'",
                            font=ctk.CTkFont(size=14, weight="bold"),
                            text_color=("white", "white")
                        ).pack(anchor="w", padx=10, pady=(8, 2))
                        
                        # PII type and confidence
                        ctk.CTkLabel(
                            word_item,
                            text=f"   📍 PII Type: {word_pred['label']} | Confidence: {word_pred['confidence']:.3f}",
                            font=ctk.CTkFont(size=12),
                            text_color=("red", "lightcoral")
                        ).pack(anchor="w", padx=10, pady=(0, 8))
                
                if not pii_words_found:
                    ctk.CTkLabel(
                        word_results_frame,
                        text="No PII detected at word level",
                        font=ctk.CTkFont(size=12),
                        text_color=("gray60", "gray40")
                    ).pack(padx=10, pady=10)
                
                # Token-level details (Secondary, collapsible)
                ctk.CTkLabel(
                    main_frame,
                    text="🔍 Token-Level Details (Advanced):",
                    font=ctk.CTkFont(size=14, weight="bold")
                ).pack(anchor="w", pady=(15, 5))
                
                token_frame = ctk.CTkFrame(main_frame)
                token_frame.pack(fill="x", pady=(0, 15))
                
                for result in ner_results:
                    if result['score'] >= self.threshold.get():
                        token_item = ctk.CTkFrame(token_frame)
                        token_item.pack(fill="x", padx=10, pady=3)
                        
                        ctk.CTkLabel(
                            token_item,
                            text=f"Token: '{result['word']}' → {result['entity_group']}",
                            font=ctk.CTkFont(size=11),
                            text_color=("gray70", "gray50")
                        ).pack(anchor="w", padx=8, pady=3)
                        
                        ctk.CTkLabel(
                            token_item,
                            text=f"   Confidence: {result['score']:.3f} | Position: {result['start']}-{result['end']}",
                            font=ctk.CTkFont(size=9),
                            text_color=("gray60", "gray40")
                        ).pack(anchor="w", padx=8, pady=(0, 3))
            
            ctk.CTkButton(
                main_frame,
                text="Close",
                command=results_window.destroy,
                width=100
            ).pack(pady=20)
        
        def clear_text(self):
            self.text_input.delete(1.0, tk.END)
        
        def paste_text(self):
            try:
                clipboard_text = self.root.clipboard_get()
                self.text_input.delete(1.0, tk.END)
                self.text_input.insert(1.0, clipboard_text)
            except:
                messagebox.showwarning("Warning", "No text in clipboard")
        
        def run(self):
            self.root.mainloop()
    
    # Create and run GUI
    app = PIIDetectorGUI()
    app.run()

def main():
    """Main function to run the Ettin PII detector"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        # Run GUI version
        create_pii_gui()
    else:
        # Run CLI version
        print("="*80)
        print("🤖 ETTIN ENCODER PII DETECTION MODEL")
        print("Running Ettin 1B parameter model locally")
        print("Run with --gui flag for GUI interface")
        print("="*80)
        
        # Initialize detector
        detector = EttinDetector()
        
        # Test cases to demonstrate the model
        test_cases = [
            {
                "name": "Personal Information",
                "text": "Hi, my name is John Smith and I live at 123 Main Street, New York. You can reach me at john.smith@email.com or call (555) 123-4567."
            },
            {
                "name": "Financial Information", 
                "text": "Please charge my credit card 4532-1234-5678-9012, the CVV is 123. My account number is AC789012345 and routing number is 021000021."
            },
            {
                "name": "Government IDs",
                "text": "My SSN is 123-45-6789 and my driver's license number is DL987654321. I was born on 03/15/1985 and I'm 38 years old."
            }
        ]
        
        # Run test cases
        print(f"\n🧪 RUNNING TEST CASES:")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 TEST CASE {i}: {test_case['name']}")
            print("-" * 60)
            
            # Get predictions
            ner_results, pii_found, word_predictions = detector.predict_pii(test_case['text'])
            
            # Format and display results
            detector.format_results(test_case['text'], ner_results, pii_found, word_predictions)
        
        # Start interactive mode
        detector.interactive_mode()

if __name__ == "__main__":
    main()

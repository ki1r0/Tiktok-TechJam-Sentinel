import os
import torch
import joblib
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForTokenClassification
import warnings
warnings.filterwarnings("ignore")

class PIIDetector:
    def __init__(self, model_path="./pii_ettin_encoder_400m_v1"):
        """Initialize PII detector with trained model"""
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔍 Loading PII Detection Model...")
        print(f"   Model path: {model_path}")
        print(f"   Device: {self.device}")
        
        # Load model and tokenizer from final directory
        final_model_path = os.path.join(model_path, "final")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(final_model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(final_model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"   ✅ Model loaded successfully")
            
            # Load label mappings
            self.label_to_id = joblib.load(os.path.join(model_path, "label_to_id.joblib"))
            self.id_to_label = joblib.load(os.path.join(model_path, "id_to_label.joblib"))
            
            # Load training summary for model info
            self.summary = joblib.load(os.path.join(model_path, "training_summary.joblib"))
            
            print(f"   📊 Model Info:")
            print(f"      Classes: {self.summary['n_classes']}")
            print(f"      F1-Macro Score: {self.summary['best_f1_macro']:.4f}")
            print(f"      Training samples: {self.summary['train_samples']:,}")
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise

    def predict_pii(self, text, threshold=0.5):
        """Predict PII labels for each token in the text"""
        
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)
        
        # Convert tokens back to text with labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = predicted_labels[0].cpu().numpy()
        predictions_probs = predictions[0].cpu().numpy()
        
        results = []
        pii_found = set()
        
        for i, (token, label_id) in enumerate(zip(tokens, predicted_labels)):
            if token in ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>']:  # Skip special tokens
                continue
            
            label = self.id_to_label[label_id]
            confidence = predictions_probs[i][label_id]
            
            # Get character offsets for this token
            if i < len(offset_mapping):
                start, end = offset_mapping[i]
                token_text = text[start:end] if start != end else token
            else:
                token_text = token
            
            results.append({
                'token': token_text,
                'token_raw': token,
                'label': label,
                'confidence': confidence,
                'token_id': i,
                'char_start': start if i < len(offset_mapping) else 0,
                'char_end': end if i < len(offset_mapping) else 0
            })
            
            # Track PII types found (excluding 'O' and special tokens)
            if label != 'O' and label != 'IGNORE' and confidence > threshold:
                pii_found.add(label)
        
        # Group tokens into words
        word_predictions = self._group_tokens_into_words(text, results)
        
        return results, pii_found, word_predictions

    def _group_tokens_into_words(self, text, token_results):
        """Group tokens into words and determine word-level labels"""
        words = text.split()
        word_predictions = []
        text_pos = 0  # Track position in text
        
        for word in words:
            # Find the actual position of this word in the text
            word_start = text.find(word, text_pos)
            word_end = word_start + len(word)
            text_pos = word_end  # Update position for next word
            
            # Find tokens that belong to this word
            word_tokens = []
            
            for token_result in token_results:
                token_start = token_result['char_start']
                token_end = token_result['char_end']
                
                # Check if token overlaps with word
                if token_start < word_end and token_end > word_start:
                    word_tokens.append(token_result)
            
            if not word_tokens:
                # If no tokens found, assume it's 'O' (Outside)
                word_predictions.append({
                    'word': word,
                    'label': 'O',
                    'confidence': 1.0,
                    'tokens': []
                })
                continue
            
            # Determine word label from tokens
            pii_labels = [t['label'] for t in word_tokens if t['label'] != 'O' and t['label'] != 'IGNORE']
            
            if not pii_labels:
                # All tokens are 'O', so word is 'O'
                word_label = 'O'
                word_confidence = max([t['confidence'] for t in word_tokens])
            else:
                # Find most common PII label
                label_counts = Counter(pii_labels)
                word_label = label_counts.most_common(1)[0][0]
                
                # Average confidence of tokens with this label
                label_confidences = [t['confidence'] for t in word_tokens if t['label'] == word_label]
                word_confidence = sum(label_confidences) / len(label_confidences) if label_confidences else 0.0
            
            word_predictions.append({
                'word': word,
                'label': word_label,
                'confidence': word_confidence,
                'tokens': word_tokens
            })
        
        return word_predictions

    def format_results(self, text, results, pii_found, word_predictions):
        """Format prediction results nicely"""
        print(f"\n{'='*80}")
        print(f"📝 TEXT: {text}")
        print(f"{'='*80}")
        
        if not pii_found:
            print("✅ NO PII DETECTED")
            return
        
        print(f"🚨 PII DETECTED: {len(pii_found)} types found")
        print(f"   Types: {', '.join(sorted(pii_found))}")
        
        # Show word-level predictions
        print(f"\n🔤 WORD-LEVEL PREDICTIONS:")
        for word_pred in word_predictions:
            if word_pred['label'] != 'O':
                print(f"   🔍 {word_pred['label']}: '{word_pred['word']}' (conf: {word_pred['confidence']:.3f})")
        
        # Show detailed token-level predictions
        print(f"\n📋 TOKEN-LEVEL PREDICTIONS:")
        current_pii = None
        pii_text = ""
        
        for result in results:
            token = result['token']
            label = result['label']
            confidence = result['confidence']
            
            if label != 'O' and label != 'IGNORE' and confidence > 0.5:
                if label != current_pii:
                    if current_pii is not None:
                        print(f"   🔍 {current_pii}: '{pii_text.strip()}' (conf: {confidence:.3f})")
                    current_pii = label
                    pii_text = token
                else:
                    pii_text += f" {token}"
            elif current_pii is not None:
                print(f"   🔍 {current_pii}: '{pii_text.strip()}' (conf: {confidence:.3f})")
                current_pii = None
                pii_text = ""
        
        # Handle case where PII continues to end
        if current_pii is not None:
            print(f"   🔍 {current_pii}: '{pii_text.strip()}')")

def main():
    """Test the PII detector on 5 different comments with various PII types"""
    
    print("="*80)
    print("🎯 PII DETECTION EVALUATION")
    print("Testing Ettin Encoder 1B Token-Level PII Classifier")
    print("="*80)
    
    # Initialize detector
    detector = PIIDetector()
    
    # Test cases with different PII types
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
        },
        {
            "name": "Medical Information",
            "text": "Patient ID MED123456 has an appointment on 2024-01-15. Insurance policy number is INS789012345. Previous medical record shows treatment in 2023."
        },
        {
            "name": "Mixed Business Data",
            "text": "Company ABC Corp, employee Sarah Johnson (ID: EMP98765) earned $75,000 last year. Her IBAN is GB33BUKB20201555555555 and tax ID is TX123456789."
        }
    ]
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST CASE {i}: {test_case['name']}")
        print("-" * 60)
        
        # Get predictions
        results, pii_found, word_predictions = detector.predict_pii(test_case['text'])
        
        # Format and display results
        detector.format_results(test_case['text'], results, pii_found, word_predictions)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 EVALUATION COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Model Details:")
    print(f"   Model: {detector.summary['model_name']}")
    print(f"   Classes supported: {detector.summary['n_classes']}")
    print(f"   Training samples: {detector.summary['train_samples']:,}")
    print(f"   Training F1-Macro: {detector.summary['best_f1_macro']:.4f}")
    print(f"   Total test cases: {len(test_cases)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
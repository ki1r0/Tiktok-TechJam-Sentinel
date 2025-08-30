import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.preprocessing import LabelEncoder  # Back to LabelEncoder for token classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,  # Changed for token classification
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification  # Changed for token classification
)
from datasets import Dataset, load_dataset
import warnings
warnings.filterwarnings("ignore")

# Configuration
class Config:
    VER = 1
    
    # ========================================
    # Ettin Encoder Model Configuration - Easy to Change!
    # ========================================
    # Choose one of these Ettin encoder variants:
    # MODEL_NAME = "jhu-clsp/ettin-encoder-17m"    # XXS - Mobile/Edge devices
    # MODEL_NAME = "jhu-clsp/ettin-encoder-32m"    # XS - Fast inference
    # MODEL_NAME = "jhu-clsp/ettin-encoder-68m"    # Small - Balanced performance
    # MODEL_NAME = "jhu-clsp/ettin-encoder-150m"   # Base - Standard use cases
    MODEL_NAME = "jhu-clsp/ettin-encoder-400m"   # Large - High accuracy needs
    # MODEL_NAME = "jhu-clsp/ettin-encoder-1b"     # XL - Best performance (1B params)
    
    # Training Mode Options:
    # False = 80-20 split with validation (default)
    # True = 100% of data, no validation
    USE_FULL_DATASET = False  # Use validation split for PII detection
    
    # Model-specific hyperparameters (automatically adjusted based on model size)
    if "17m" in MODEL_NAME:
        EPOCHS = 15
        MAX_LEN = 256
        LEARNING_RATE = 1e-4
        BATCH_SIZE = 64
        EVAL_BATCH_SIZE = 128
        model_size = "17m"
    elif "32m" in MODEL_NAME:
        EPOCHS = 12
        MAX_LEN = 256
        LEARNING_RATE = 8e-5
        BATCH_SIZE = 64
        EVAL_BATCH_SIZE = 128
        model_size = "32m"
    elif "68m" in MODEL_NAME:
        EPOCHS = 10
        MAX_LEN = 256
        LEARNING_RATE = 6e-5
        BATCH_SIZE = 48
        EVAL_BATCH_SIZE = 96
        model_size = "68m"
    elif "150m" in MODEL_NAME:
        EPOCHS = 8
        MAX_LEN = 256
        LEARNING_RATE = 5e-5
        BATCH_SIZE = 32
        EVAL_BATCH_SIZE = 64
        model_size = "150m"
    elif "400m" in MODEL_NAME:
        EPOCHS = 6
        MAX_LEN = 256
        LEARNING_RATE = 4e-5
        BATCH_SIZE = 24
        EVAL_BATCH_SIZE = 48
        model_size = "400m"
    elif "1b" in MODEL_NAME:
        EPOCHS = 5
        MAX_LEN = 256
        LEARNING_RATE = 5e-5
        BATCH_SIZE = 16  # Reduced for 1B model
        EVAL_BATCH_SIZE = 32
        model_size = "1b"
    else:
        # Default settings
        EPOCHS = 8
        MAX_LEN = 256
        LEARNING_RATE = 5e-5
        BATCH_SIZE = 32
        EVAL_BATCH_SIZE = 64
        model_size = "ettin"
    
    # Training configuration
    SEED = 42
    TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
    
    # Auto-generate output directory based on model variant
    OUTPUT_DIR = f"./pii_ettin_encoder_{model_size}_v{VER}"

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_gpu_availability():
    """Check GPU availability and setup"""
    print("\n🔍 GPU Configuration:")
    print("="*50)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"🚀 GPU Available:")
        print(f"   Device Count: {device_count}")
        print(f"   Current Device: {current_device}")
        print(f"   Device Name: {device_name}")
        print(f"   Device Memory: {device_memory:.1f} GB")
        
        # Enable optimizations for H100/A100
        if "H100" in device_name or "A100" in device_name:
            print("   High-end GPU detected - enabling optimizations")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        device = torch.device("cuda")
        print(f"   ✅ Using device: {device}")
        return device
    else:
        print("❌ No GPU available - will use CPU")
        print("   Note: Training will be significantly slower on CPU")
        device = torch.device("cpu")
        print(f"   Using device: {device}")
        return device

def create_output_directories():
    """Create all necessary output directories"""
    print(f"\n📁 Setting up output directories...")
    
    try:
        # Create base directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        print(f"   ✅ Base output directory: {Config.OUTPUT_DIR}")
        
        # Create subdirectories
        subdirs = ["checkpoints", "final", "logs"]
        for subdir in subdirs:
            dir_path = os.path.join(Config.OUTPUT_DIR, subdir)
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ Created: {subdir}/")
        
        print(f"   📁 All directories ready")
        return Config.OUTPUT_DIR
        
    except Exception as e:
        print(f"   ❌ Error creating directories: {e}")
        raise

def load_and_preprocess_pii_data():
    """Load and preprocess PII masking data for token classification"""
    print("\n📊 Loading PII masking dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("ai4privacy/pii-masking-200k")
    print(f"   Loaded dataset: {len(dataset['train'])} train samples")
    
    # Convert to pandas for easier manipulation
    train_df = dataset['train'].to_pandas()
    
    # Use source_text as input
    train_df['text'] = train_df['source_text']
    
    # Extract token-level labels from span_labels
    def create_token_labels(text, span_labels_str):
        """Create token-level labels from character-level spans"""
        try:
            span_labels = eval(span_labels_str) if isinstance(span_labels_str, str) else span_labels_str
            
            # Initialize all characters as 'O' (Outside)
            char_labels = ['O'] * len(text)
            
            # Apply labels from spans
            for span in span_labels:
                if len(span) >= 3:  # [start, end, label]
                    start, end, label = span[0], span[1], span[2]
                    for i in range(start, min(end, len(text))):
                        char_labels[i] = label
            
            return char_labels
                
        except Exception as e:
            # Return all 'O' labels if parsing fails
            return ['O'] * len(text)
    
    train_df['char_labels'] = train_df.apply(
        lambda row: create_token_labels(row['text'], row['span_labels']), axis=1
    )
    
    print(f"   Dataset shape: {train_df.shape}")
    
    # Analyze label distribution at character level
    all_labels = []
    for label_list in train_df['char_labels']:
        all_labels.extend(label_list)
    
    label_counts = pd.Series(all_labels).value_counts()
    print(f"   Total unique labels found: {len(label_counts)}")
    print(f"   'O' (Outside) tokens: {label_counts.get('O', 0)} ({label_counts.get('O', 0)/len(all_labels)*100:.1f}%)")
    
    print(f"\n   Top 10 PII labels:")
    for label, count in label_counts.head(10).items():
        print(f"      {label}: {count} chars ({count/len(all_labels)*100:.1f}%)")
    
    # Show example
    print(f"\n   Example with token-level labels:")
    example = train_df.iloc[0]
    print(f"   Text: {example['text'][:100]}...")
    print(f"   Char labels (first 20): {example['char_labels'][:20]}")
    
    return train_df

def format_input_for_pii_detection(text):
    """
    Create Ettin encoder prompt for multilabel PII classification
    Using "What types of PII?" style classification
    """
    return (
        f"Text: {text}\n"
        f"What types of PII does this contain?"
    )

def compute_token_classification_metrics(eval_pred):
    """Compute F1-score and accuracy for token classification"""
    predictions, labels = eval_pred
    
    # Get predictions (argmax for each token)
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (for padding tokens)
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        for (p, l) in zip(prediction, label):
            if l != -100:  # -100 is the ignored index for padding
                true_predictions.append(p)
                true_labels.append(l)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, true_predictions)
    
    # Use macro F1 to give equal weight to all classes (including rare PII types)
    f1_macro = f1_score(true_labels, true_predictions, average='macro', zero_division=0)
    # Use micro F1 for overall performance
    f1_micro = f1_score(true_labels, true_predictions, average='micro', zero_division=0)
    # Weighted F1 considers class imbalance
    f1_weighted = f1_score(true_labels, true_predictions, average='weighted', zero_division=0)
    
    precision_macro = precision_score(true_labels, true_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, true_predictions, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,      # Primary metric for model selection
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }

def prepare_datasets_for_pii(train_df, tokenizer):
    """Prepare datasets for token-level PII detection training"""
    print("\n🔧 Preparing datasets for token-level PII detection...")
    
    def align_labels_with_tokens(text, char_labels, tokenizer):
        """Align character-level labels with tokenized text"""
        # Tokenize the text
        tokenized = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        offset_mapping = tokenized['offset_mapping']
        
        # Initialize token labels
        token_labels = []
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0:  # Special tokens like [CLS], [SEP]
                token_labels.append('IGNORE')
            else:
                # Get the most common label for this token's character span
                if start < len(char_labels) and end <= len(char_labels):
                    char_span_labels = char_labels[start:end]
                    if char_span_labels:
                        # Use the most common label in this span
                        most_common_label = max(set(char_span_labels), key=char_span_labels.count)
                        token_labels.append(most_common_label)
                    else:
                        token_labels.append('O')
                else:
                    token_labels.append('O')
        
        return tokens, token_labels, tokenized['input_ids'], tokenized['attention_mask']
    
    # Process all examples
    processed_data = []
    all_labels = set()
    
    print("   Processing examples...")
    for idx, row in train_df.iterrows():
        if idx % 10000 == 0:
            print(f"      Processed {idx}/{len(train_df)} examples...")
        
        tokens, token_labels, input_ids, attention_mask = align_labels_with_tokens(
            row['text'], row['char_labels'], tokenizer
        )
        
        processed_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': token_labels
        })
        
        all_labels.update(token_labels)
    
    # Create label encoder
    all_labels = sorted(list(all_labels))
    print(f"   All labels found: {all_labels}")
    
    label_to_id = {label: idx for idx, label in enumerate(all_labels)}
    id_to_label = {idx: label for idx, label in enumerate(all_labels)}
    
    # Convert labels to IDs
    for item in processed_data:
        item['labels'] = [label_to_id[label] if label != 'IGNORE' else -100 for label in item['labels']]
    
    print(f"   Number of classes: {len(all_labels)}")
    print(f"   Classes: {all_labels[:10]}{'...' if len(all_labels) > 10 else ''}")
    
    # Create DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    return processed_df, label_to_id, id_to_label, len(all_labels)

def tokenize_function(examples, tokenizer):
    """Tokenization function for Ettin encoder"""
    return tokenizer(
        examples["formatted_text"], 
        padding="max_length", 
        truncation=True, 
        max_length=Config.MAX_LEN
    )

def train_model(train_dataset, val_dataset, model, tokenizer, device):
    """Train the token-level PII detection model"""
    print(f"\n🚀 Starting Token-Level PII Detection Training")
    print(f"="*60)
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"   Training Mode: {'Full Dataset' if Config.USE_FULL_DATASET else '80-20 Split'}")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    print(f"   Max length: {Config.MAX_LEN}")
    
    # Training arguments optimized for Ettin encoder
    training_args = TrainingArguments(
        output_dir=f"{Config.OUTPUT_DIR}/checkpoints",
        do_train=True,
        do_eval=val_dataset is not None,
        eval_strategy="steps" if val_dataset is not None else "no",
        save_strategy="steps",
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=0.1,  # Ettin benefits from warmup
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=f"{Config.OUTPUT_DIR}/logs",
        logging_steps=100,  # Reduced logging frequency
        save_steps=2000,     # Less frequent saving
        eval_steps=2000 if val_dataset is not None else 5000,  # Less frequent evaluation
        save_total_limit=3, # Keep more checkpoints for better model selection
        metric_for_best_model="f1_macro" if val_dataset is not None else None,  # Use F1-macro for model selection
        greater_is_better=True,
        load_best_model_at_end=val_dataset is not None,  # Load best checkpoint at end
        report_to="none",
        # Precision settings optimized for Ettin encoder
        bf16=device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8,  # Use bf16 for newer GPUs
        fp16=device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8,   # Use fp16 for older GPUs
        dataloader_num_workers=2 if device.type == "cuda" else 0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # Gradient accumulation for larger effective batch size with smaller models
        gradient_accumulation_steps=2 if Config.model_size in ["17m", "32m"] else 1,
    )
    
    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_token_classification_metrics if val_dataset is not None else None,
    )
    
    # Train
    print(f"\n⏰ Training started...")
    trainer.train()
    
    # Get best score
    best_score = trainer.state.best_metric if trainer.state.best_metric else 0.0
    print(f"\n🎯 Training completed!")
    if val_dataset is not None:
        print(f"   Best F1-Macro Score: {best_score:.4f}")
    else:
        print(f"   Training completed (no validation in full dataset mode)")
    
    return trainer, best_score

def main():
    """Main training function"""
    print("="*80)
    print("TOKEN-LEVEL PII DETECTION - ETTIN ENCODER TRAINING")
    print("="*80)
    
    # Display model configuration
    print(f"\n🤖 Model Configuration:")
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Model Size: {Config.model_size}")
    print(f"   Training Mode: {'Full Dataset (100%)' if Config.USE_FULL_DATASET else '80-20 Split'}")
    print(f"   Output Directory: {Config.OUTPUT_DIR}")
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Max Length: {Config.MAX_LEN}")
    print(f"   Learning Rate: {Config.LEARNING_RATE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Train Split: {Config.TRAIN_SPLIT:.0%}")
    
    print(f"\n💡 Ettin Encoder Info:")
    print(f"   • Part of paired encoder-decoder suite")
    print(f"   • Trained with identical data to Ettin decoders")
    print(f"   • Uses ModernBERT tokenizer (50,368 vocab)")
    print(f"   • Optimized for token classification tasks")
    print(f"   • Deep but efficient architecture")
    print(f"   • Outperforms ModernBERT on NER tasks")
    
    # Set seed
    set_seed(Config.SEED)
    
    # Check GPU and get device
    device = check_gpu_availability()
    
    # Create output directories
    create_output_directories()
    
    # Load and preprocess PII data
    train_df = load_and_preprocess_pii_data()
    
    # Initialize tokenizer
    print(f"\n📝 Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    print(f"   ✅ Tokenizer loaded successfully")
    print(f"   Vocab size: {len(tokenizer)}")
    
    # Prepare datasets for token classification
    processed_df, label_to_id, id_to_label, n_classes = prepare_datasets_for_pii(train_df, tokenizer)
    
    # Split data based on training mode
    if Config.USE_FULL_DATASET:
        print(f"\n🎯 Using full dataset for training (100%)...")
        train_split = processed_df
        val_split = None
        print(f"   Training set: {len(train_split)} samples (100%)")
        print(f"   Validation: None (full dataset mode)")
    else:
        print(f"\n✂️ Splitting data (80-20 random split)...")
        train_split, val_split = train_test_split(
            processed_df,
            test_size=1-Config.TRAIN_SPLIT,  # 20% for validation
            random_state=Config.SEED,
            shuffle=True
        )
        print(f"   Training set: {len(train_split)} samples ({len(train_split)/len(processed_df)*100:.1f}%)")
        print(f"   Validation set: {len(val_split)} samples ({len(val_split)/len(processed_df)*100:.1f}%)")
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_split.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_split.reset_index(drop=True)) if val_split is not None else None
    
    print(f"   ✅ Dataset preparation completed")
    
    # Initialize model for token classification
    print(f"\n🧠 Loading model: {Config.MODEL_NAME}")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            Config.MODEL_NAME,
            num_labels=n_classes,
            id2label=id_to_label,
            label2id=label_to_id
        )
        print(f"   ✅ Ettin encoder model loaded successfully for token classification")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print(f"   💡 Make sure you have the latest transformers version:")
        print(f"      pip install git+https://github.com/huggingface/transformers.git")
        raise
    
    # Move model to device
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Information:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params/1e6:.1f}M parameters")
    print(f"   Expected size: ~{Config.model_size}")
    
    # Train model
    trainer, best_score = train_model(train_dataset, val_dataset, model, tokenizer, device)
    
    # Save final model
    final_model_path = f"{Config.OUTPUT_DIR}/final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")
    
    # Save preprocessing artifacts
    joblib.dump(label_to_id, f"{Config.OUTPUT_DIR}/label_to_id.joblib")
    joblib.dump(id_to_label, f"{Config.OUTPUT_DIR}/id_to_label.joblib")
    print(f"   ✅ Label mappings saved")
    
    # Save training summary
    summary = {
        'model_name': Config.MODEL_NAME,
        'best_f1_macro': best_score,  # Changed from best_accuracy
        'n_classes': n_classes,
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'train_samples': len(train_split),
        'val_samples': len(val_split) if val_split is not None else 0,
        'epochs': Config.EPOCHS,
        'batch_size': Config.BATCH_SIZE,
        'learning_rate': Config.LEARNING_RATE,
        'max_length': Config.MAX_LEN,
        'use_full_dataset': Config.USE_FULL_DATASET,
        'problem_type': 'token_classification'
    }
    joblib.dump(summary, f"{Config.OUTPUT_DIR}/training_summary.joblib")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 TOKEN-LEVEL PII DETECTION TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📊 Training Summary:")
    print(f"   Model: {Config.MODEL_NAME}")
    print(f"   Model Size: {Config.model_size} ({total_params/1e6:.1f}M params)")
    print(f"   Task: Token-Level PII Classification ({n_classes} classes)")
    print(f"   Classes: {n_classes}")
    if val_split is not None:
        print(f"   Best F1-Macro Score: {best_score:.4f}")
    print(f"   Training Mode: {'Full Dataset (100%)' if Config.USE_FULL_DATASET else '80-20 Split'}")
    print(f"   Training Samples: {len(train_split)}")
    if val_split is not None:
        print(f"   Validation Samples: {len(val_split)}")
    print(f"\n📁 Output Directory: {Config.OUTPUT_DIR}")
    print(f"   ✅ Final model: final/")
    print(f"   ✅ Checkpoints: checkpoints/")
    print(f"   ✅ Label mappings: label_to_id.joblib, id_to_label.joblib")
    print(f"   ✅ Training summary: training_summary.joblib")
    print(f"   💡 Ettin encoder trained for token-level PII detection!")
    print(f"   🏆 Ready to predict PII labels for each token")
    print(f"   📋 Sample labels: {list(label_to_id.keys())[:5]}{'...' if n_classes > 5 else ''}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

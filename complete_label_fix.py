#!/usr/bin/env python3
"""
Complete fix for Bengali hate speech classification with proper label preprocessing
Fixes the -1 label issue and implements 75% accuracy targeting
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    Trainer, TrainingArguments
)
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import os

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # debug CUDA asserts synchronously

# STEP 1: Define correct label mapping
label2id = {
    "None": 0,
    "Religious Hate": 1, 
    "Sexism": 2,
    "Political Hate": 3,
    "Profane": 4,
    "Abusive": 5
}

id2label = {v: k for k, v in label2id.items()}

print("Label mapping:")
for label, id_val in label2id.items():
    print(f"  {label} -> {id_val}")

# STEP 2: Load and preprocess data with CORRECT label mapping
def load_and_preprocess_data():
    """Load TSV files and apply correct label mapping"""
    
    # Load training data
    train_df = pd.read_csv("blp25_hatespeech_subtask_1A_train.tsv", sep='\t')
    eval_df = pd.read_csv("blp25_hatespeech_subtask_1A_dev.tsv", sep='\t')
    
    print(f"Loaded {len(train_df)} training samples, {len(eval_df)} eval samples")
    
    # Check unique labels in data
    print("Unique labels in training data:", train_df['label'].unique())
    print("Unique labels in eval data:", eval_df['label'].unique())
    
    # Map text labels to numeric IDs
    train_df['label_id'] = train_df['label'].map(label2id)
    eval_df['label_id'] = eval_df['label'].map(label2id)
    
    # Check for unmapped labels
    train_unmapped = train_df['label_id'].isna().sum()
    eval_unmapped = eval_df['label_id'].isna().sum()
    
    if train_unmapped > 0:
        print(f"WARNING: {train_unmapped} unmapped training labels")
        print("Unmapped labels:", train_df[train_df['label_id'].isna()]['label'].unique())
    
    if eval_unmapped > 0:
        print(f"WARNING: {eval_unmapped} unmapped eval labels")
        print("Unmapped labels:", eval_df[eval_df['label_id'].isna()]['label'].unique())
    
    # Remove unmapped samples
    train_df = train_df.dropna(subset=['label_id'])
    eval_df = eval_df.dropna(subset=['label_id'])
    
    # Convert to int
    train_df['label_id'] = train_df['label_id'].astype(int)
    eval_df['label_id'] = eval_df['label_id'].astype(int)
    
    print(f"After cleaning: {len(train_df)} training, {len(eval_df)} eval samples")
    
    # Verify label ranges
    train_labels = train_df['label_id'].values
    eval_labels = eval_df['label_id'].values
    
    print(f"Training label range: {train_labels.min()} to {train_labels.max()}")
    print(f"Eval label range: {eval_labels.min()} to {eval_labels.max()}")
    print(f"Unique training labels: {np.unique(train_labels)}")
    print(f"Unique eval labels: {np.unique(eval_labels)}")
    
    return train_df, eval_df

# STEP 3: Tokenization function
def tokenize_function(examples, tokenizer):
    """Tokenize text and ensure labels are correct"""
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    # Ensure labels are integers in correct range
    labels = []
    for label in examples["label_id"]:
        label_int = int(label)
        if 0 <= label_int <= 5:
            labels.append(label_int)
        else:
            print(f"WARNING: Invalid label {label_int}, clamping to 0")
            labels.append(0)
    
    tokenized["labels"] = labels
    return tokenized

# STEP 4: Custom trainer with safe loss computation
class SafeTargetedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Validate labels
        num_classes = logits.size(-1)
        if labels is not None:
            # Ensure correct dtype for loss indexing
            if labels.dtype != torch.long:
                labels = labels.long()
            valid_mask = (labels >= 0) & (labels < num_classes)
            if not valid_mask.all():
                print(f"WARNING: Invalid labels found. Clamping to valid range.")
                labels = torch.clamp(labels, 0, num_classes-1)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits.view(-1, num_classes), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# STEP 5: Metrics computation
def compute_metrics_safe(eval_pred):
    """Safe metrics computation with error handling"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Ensure labels are in valid range
    labels = np.clip(labels, 0, 5)
    predictions = np.clip(predictions, 0, 5)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
    }

# MAIN EXECUTION
if __name__ == "__main__":
    print("=== COMPLETE LABEL FIX FOR 75% ACCURACY ===")
    
    # Load data with correct preprocessing
    train_df, eval_df = load_and_preprocess_data()
    
    # Initialize tokenizer and model
    model_name = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6,
        id2label=id2label,
        label2id=label2id
    )
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label_id']])
    
    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True
    )
    
    # Remove unused columns and set torch format explicitly
    keep_cols = ["input_ids", "attention_mask", "labels"]
    drop_train = [c for c in train_dataset.column_names if c not in keep_cols]
    drop_eval = [c for c in eval_dataset.column_names if c not in keep_cols]
    if drop_train:
        train_dataset = train_dataset.remove_columns(drop_train)
    if drop_eval:
        eval_dataset = eval_dataset.remove_columns(drop_eval)
    
    train_dataset.set_format(type="torch", columns=keep_cols)
    eval_dataset.set_format(type="torch", columns=keep_cols)
    
    # Calculate strategic class weights for 75% accuracy
    train_labels = np.array([int(label) for label in train_df['label_id']])
    unique_labels = np.unique(train_labels)
    
    print(f"Final unique labels: {unique_labels}")
    
    # Calculate balanced weights for existing classes
    balanced_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_labels
    )
    
    # Create full weight array
    full_weights = np.ones(6)
    for i, class_id in enumerate(unique_labels):
        if 0 <= class_id < 6:
            full_weights[class_id] = balanced_weights[i]
    
    # Apply strategic adjustments for 75% accuracy target
    strategic_weights = full_weights.copy()
    strategic_weights[1] *= 1.5  # Religious Hate boost
    strategic_weights[2] *= 2.0  # Sexism boost (most critical)
    strategic_weights[0] *= 0.8  # None reduction (over-predicted)
    
    # Moderate with uniform weights (85% strategic + 15% uniform)
    uniform_weights = np.ones(6)
    final_weights = 0.85 * strategic_weights + 0.15 * uniform_weights
    
    class_weights = torch.FloatTensor(final_weights)
    
    print("\nFINAL class weights for 75% accuracy:")
    for i, weight in enumerate(class_weights):
        print(f"  {id2label[i]}: {weight:.4f}")
    
    # Training arguments optimized for 75% accuracy
    training_args = TrainingArguments(
        output_dir="./results_complete_fix",
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.08,
        learning_rate=1.8e-5,
        weight_decay=0.005,
        lr_scheduler_type="cosine",
        logging_steps=500,
        eval_steps=1000,
        save_steps=1000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=False,  # disable mixed precision while debugging CUDA asserts
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        remove_unused_columns=True,  # drop non-model columns automatically
        label_smoothing_factor=0.03,
        gradient_checkpointing=False,  # simplify graph during debug
    )
    
    # Initialize trainer
    trainer = SafeTargetedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_safe,
    )
    
    print("\n=== STARTING TRAINING ===")
    print("Target: 75% accuracy with minority class detection")
    print("Expected training time: ~45 minutes")
    
    # Train the model
    trainer.train()
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    results = trainer.evaluate()
    
    print("Final Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    if results['eval_accuracy'] >= 0.75:
        print("🎉 SUCCESS: Achieved 75%+ accuracy target!")
    else:
        print(f"📊 Current: {results['eval_accuracy']:.1%}, Need: {0.75 - results['eval_accuracy']:.1%} more")
    
    print("\n=== TRAINING COMPLETE ===")

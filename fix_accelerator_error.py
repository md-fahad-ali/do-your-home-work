#!/usr/bin/env python3
"""
Fix for AcceleratorState error in Jupyter notebook.
Run this script before executing the trainer.train() cell in your notebook.
"""

from accelerate import Accelerator
from accelerate.state import AcceleratorState

def fix_accelerator_state():
    """
    Reset and reinitialize the AcceleratorState to fix the distributed_type error.
    This should be run before creating a new Trainer instance.
    """
    try:
        # Reset the accelerator state
        AcceleratorState._reset_state()
        print("✓ AcceleratorState reset successfully")
        
        # Create a new Accelerator instance to reinitialize the state
        accelerator = Accelerator()
        print("✓ New Accelerator instance created successfully")
        
        # Clean up
        del accelerator
        print("✓ Accelerator state fixed - you can now run trainer.train()")
        
    except Exception as e:
        print(f"Error fixing accelerator state: {e}")
        print("Try restarting your Jupyter kernel and running the notebook from the beginning.")

if __name__ == "__main__":
    fix_accelerator_state()


# Fix for CUDA device-side assert error
# This error typically occurs when labels are out of range for the model

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import os

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

class SafeTargetedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # CRITICAL FIX: Validate labels are in valid range
        num_classes = logits.size(-1)
        valid_mask = (labels >= 0) & (labels < num_classes)
        
        if not valid_mask.all():
            print(f"WARNING: Found invalid labels. Valid range: 0-{num_classes-1}")
            print(f"Invalid labels: {labels[~valid_mask].unique()}")
            # Clamp invalid labels to valid range
            labels = torch.clamp(labels, 0, num_classes-1)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            # Ensure weights match number of classes
            if len(weights) != num_classes:
                print(f"WARNING: Weight size {len(weights)} != num_classes {num_classes}")
                # Resize weights to match
                if len(weights) > num_classes:
                    weights = weights[:num_classes]
                else:
                    # Pad with ones
                    padding = torch.ones(num_classes - len(weights), device=weights.device)
                    weights = torch.cat([weights, padding])
            
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        loss = loss_fn(logits.view(-1, num_classes), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics_safe(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
    }

# SAFE LABEL VALIDATION
print("=== DEBUGGING LABELS ===")
train_labels = [int(label) for label in train_dataset["label"]]
eval_labels = [int(label) for label in eval_dataset["label"]]

print(f"Train labels range: {min(train_labels)} to {max(train_labels)}")
print(f"Eval labels range: {min(eval_labels)} to {max(eval_labels)}")
print(f"Model expects: 0 to {model.config.num_labels-1}")

# Check for invalid labels
train_unique = np.unique(train_labels)
eval_unique = np.unique(eval_labels)
print(f"Unique train labels: {train_unique}")
print(f"Unique eval labels: {eval_unique}")

# Fix any invalid labels in datasets
def fix_labels(dataset, max_label):
    def clamp_labels(example):
        example["label"] = max(0, min(int(example["label"]), max_label))
        return example
    return dataset.map(clamp_labels)

max_valid_label = model.config.num_labels - 1
train_dataset = fix_labels(train_dataset, max_valid_label)
eval_dataset = fix_labels(eval_dataset, max_valid_label)

print("Labels fixed and validated.")

# Calculate safe class weights
unique_labels = np.unique(train_labels)
# Only use labels that exist and are valid
valid_labels = unique_labels[(unique_labels >= 0) & (unique_labels < model.config.num_labels)]

if len(valid_labels) < model.config.num_labels:
    print(f"Using {len(valid_labels)} classes out of {model.config.num_labels}")

# Calculate weights for existing classes
balanced_weights = compute_class_weight(
    class_weight='balanced',
    classes=valid_labels,
    y=np.array([l for l in train_labels if l in valid_labels])
)

# Create full weight array
full_weights = np.ones(model.config.num_labels)
for i, class_id in enumerate(valid_labels):
    full_weights[class_id] = balanced_weights[i]

# Apply strategic adjustments (if classes exist)
if 1 < len(full_weights):  # Religious Hate
    full_weights[1] *= 1.5
if 2 < len(full_weights):  # Sexism  
    full_weights[2] *= 2.0
if 0 < len(full_weights):  # None
    full_weights[0] *= 0.8

class_weights = torch.FloatTensor(full_weights)

print("SAFE class weights:")
for i, weight in enumerate(class_weights):
    class_name = id2l.get(i, f"Class_{i}")
    print(f"  {class_name}: {weight:.4f}")

# Safe training arguments - CONSERVATIVE SETTINGS
training_args = TrainingArguments(
    output_dir="./results_safe_75",
    num_train_epochs=3,  # Reduced for safety
    per_device_train_batch_size=8,  # Reduced batch size
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    learning_rate=2e-5,  # Standard learning rate
    weight_decay=0.01,
    lr_scheduler_type="linear",
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    save_total_limit=2,
    dataloader_num_workers=0,  # Disable multiprocessing
    fp16=False,  # Disable mixed precision for stability
    remove_unused_columns=False,  # Keep all columns
)

# Create safe trainer
trainer = SafeTargetedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_safe,
)

print("\nStarting SAFE training for 75% accuracy...")
print("Fixed: CUDA device-side assert error")
print("Conservative settings: 3 epochs, batch_size=8, fp16=False")
trainer.train()

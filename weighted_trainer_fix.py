import torch
import torch.nn as nn
from transformers import Trainer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            # Move class weights to same device as logits
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def calculate_class_weights(train_labels, method='balanced'):
    """Calculate class weights for imbalanced dataset"""
    
    # Convert to numpy array if needed
    if isinstance(train_labels[0], torch.Tensor):
        train_labels = [int(label) for label in train_labels]
    
    train_labels = np.array(train_labels)
    classes = np.unique(train_labels)
    
    if method == 'balanced':
        # Use sklearn's balanced approach
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=train_labels
        )
    elif method == 'inverse_freq':
        # Manual inverse frequency calculation
        class_counts = np.bincount(train_labels)
        total_samples = len(train_labels)
        class_weights = total_samples / (len(classes) * class_counts)
    
    # Convert to torch tensor
    weight_tensor = torch.FloatTensor(class_weights)
    
    # Print weights for verification
    print("Class weights calculated:")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f}")
    
    return weight_tensor

def create_weighted_training_args():
    """Create training arguments optimized for class imbalance"""
    from transformers import TrainingArguments
    
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=6,  # From memory: 5-7 epochs optimal
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_ratio=0.06,  # From memory: not 0.0
        learning_rate=2e-5,  # From memory: not 3e-5
        weight_decay=0.01,  # From memory: not 0.0
        lr_scheduler_type="cosine",  # From memory: not linear
        logging_dir="./logs",
        logging_steps=500,
        eval_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",  # Focus on macro-F1, not accuracy
        greater_is_better=True,
        save_total_limit=3,
        dataloader_num_workers=4,
        label_smoothing_factor=0.05,  # From memory: helps with regularization
    )

def compute_metrics_with_macro_f1(eval_pred):
    """Compute metrics focusing on macro-F1 for imbalanced classes"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
    }

# Example usage:
"""
# In your notebook, use this code:

# 1. Calculate class weights
train_labels = [int(label) for label in train_dataset["label"]]
class_weights = calculate_class_weights(train_labels, method='balanced')

# 2. Create training arguments
training_args = create_weighted_training_args()

# 3. Create weighted trainer
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_with_macro_f1,
)

# 4. Train with weighted loss
trainer.train()
"""

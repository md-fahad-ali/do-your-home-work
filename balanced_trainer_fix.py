# Balanced approach - less aggressive weights + focal loss for better accuracy

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedTrainer(Trainer):
    def __init__(self, class_weights=None, use_focal=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        
        if use_focal and class_weights is not None:
            self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.use_focal and hasattr(self, 'focal_loss'):
            loss = self.focal_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        elif self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

def calculate_moderate_weights(train_labels):
    """Calculate less aggressive class weights for better accuracy balance"""
    train_labels = np.array([int(label) for label in train_labels])
    classes = np.unique(train_labels)
    
    # Use balanced weights but reduce extremes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    
    # Moderate the weights to prevent over-correction
    # Cap maximum weight at 10x minimum weight
    min_weight = class_weights.min()
    max_weight = min_weight * 10
    class_weights = np.clip(class_weights, min_weight, max_weight)
    
    # Normalize to prevent loss explosion
    class_weights = class_weights / class_weights.sum() * len(classes)
    
    return torch.FloatTensor(class_weights)

def compute_metrics_balanced(eval_pred):
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

# Usage for your Colab:
"""
# Calculate moderate class weights
train_labels = [int(label) for label in train_dataset["label"]]
class_weights = calculate_moderate_weights(train_labels)

print("Moderate class weights:")
for i, weight in enumerate(class_weights):
    print(f"  {id2l[i]}: {weight:.4f}")

# Optimized training arguments for higher accuracy
training_args = TrainingArguments(
    output_dir="./results_balanced",
    num_train_epochs=5,  # Reduced from 6 to prevent overfitting
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,  # Increased warmup for stability
    learning_rate=1.5e-5,  # Slightly lower for better convergence
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=500,
    eval_steps=500,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",  # Focus on accuracy now
    greater_is_better=True,
    save_total_limit=3,
    label_smoothing_factor=0.1,  # Increased for better generalization
    dataloader_num_workers=2,
)

# Create balanced trainer with focal loss
trainer = BalancedTrainer(
    class_weights=class_weights,
    use_focal=True,  # Use focal loss for better balance
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_balanced,
)

# Train
trainer.train()
"""

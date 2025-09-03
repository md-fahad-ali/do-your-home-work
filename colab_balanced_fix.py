# Balanced solution for 70%+ accuracy - paste this in new Colab cell

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
        # Move alpha to same device as inputs if needed
        alpha = self.alpha
        if alpha is not None:
            alpha = alpha.to(inputs.device)
        
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=alpha, reduction='none')
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

# Calculate moderate class weights (less aggressive than before)
train_labels = [int(label) for label in train_dataset["label"]]
classes = np.unique(train_labels)

# Use balanced weights but moderate them
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=np.array(train_labels)
)

# Cap maximum weight to prevent over-correction
min_weight = class_weights.min()
max_weight = min_weight * 8  # Reduced from 10 to 8
class_weights = np.clip(class_weights, min_weight, max_weight)

# Normalize
class_weights = class_weights / class_weights.sum() * len(classes)
class_weights = torch.FloatTensor(class_weights)

print("Moderate class weights for better accuracy:")
for i, weight in enumerate(class_weights):
    print(f"  {id2l[i]}: {weight:.4f}")

# Optimized training arguments for 70%+ accuracy
training_args = TrainingArguments(
    output_dir="./results_balanced",
    num_train_epochs=5,  # Reduced to prevent overfitting
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,  # Increased warmup
    learning_rate=1.5e-5,  # Slightly lower LR
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=500,
    eval_steps=500,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",  # Focus on accuracy
    greater_is_better=True,
    save_total_limit=3,
    label_smoothing_factor=0.1,  # Increased smoothing
    dataloader_num_workers=2,
)

# Create balanced trainer with focal loss
trainer = BalancedTrainer(
    class_weights=class_weights,
    use_focal=True,  # Focal loss for better balance
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_balanced,
)

print("\nStarting balanced training for 70%+ accuracy...")
trainer.train()

# Evaluate with new approach
print("\nEvaluating balanced model...")
eval_predictions = trainer.predict(eval_dataset)
predictions = np.argmax(eval_predictions.predictions, axis=1)
labels = eval_dataset["label"]

if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# Generate comparison confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2l.values()))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Balanced Approach (Target: 70%+ Accuracy)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nClassification Report - Balanced Approach:")
print(classification_report(labels, predictions, target_names=list(id2l.values())))

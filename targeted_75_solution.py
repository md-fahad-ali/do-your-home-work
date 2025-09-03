# Targeted solution for 75% accuracy based on baseline analysis
# Current: 69.63% accuracy, Religious Hate & Sexism have 0% recall
# Disable wandb for quick testing
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

class TargetedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics_targeted(eval_pred):
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

# TARGETED CLASS WEIGHTS based on baseline analysis
# Problem: Religious Hate & Sexism have 0% recall
# Solution: Strategic weights to force model to learn these classes

# BEFORE RUNNING: Make sure your variables are defined in your notebook:
# - train_dataset (your training dataset)
# - eval_dataset (your evaluation dataset) 
# - model (your trained model)
# - tokenizer (your tokenizer)
# - id2l (your label mapping dictionary)

# Extract labels from your training dataset
train_labels = [int(label) for label in train_dataset["label"]]

# Debug: Check what labels actually exist in the data
unique_labels = np.unique(train_labels)
print(f"Unique labels in training data: {unique_labels}")
print(f"Expected classes (0-{model.config.num_labels-1}): {list(range(model.config.num_labels))}")

# Use only the classes that actually exist in the training data
classes = unique_labels

# Calculate balanced weights
balanced_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=np.array(train_labels)
)

# If some classes are missing, pad the weights array
if len(classes) < model.config.num_labels:
    print(f"Warning: Only {len(classes)} classes found in training data, expected {model.config.num_labels}")
    full_weights = np.ones(model.config.num_labels)  # Default weight of 1.0 for missing classes
    for i, class_id in enumerate(classes):
        full_weights[class_id] = balanced_weights[i]
    balanced_weights = full_weights

# STRATEGIC ADJUSTMENT based on baseline failures:
# - Religious Hate & Sexism need STRONG boost (0% recall)
# - None needs reduction (over-predicted)
# - Others need mild adjustment

# Map class indices to names for clarity
class_names = ['None', 'Religious Hate', 'Sexism', 'Political Hate', 'Profane', 'Abusive']

# Custom strategic weights
strategic_weights = balanced_weights.copy()

# Boost zero-recall classes significantly
religious_hate_idx = 1  # Religious Hate
sexism_idx = 2          # Sexism
none_idx = 0           # None (over-predicted)

# Strategic adjustments:
strategic_weights[religious_hate_idx] *= 1.5  # Extra boost for Religious Hate
strategic_weights[sexism_idx] *= 2.0          # Strong boost for Sexism (most rare)
strategic_weights[none_idx] *= 0.8            # Reduce None (over-predicted)

# Cap maximum weight to avoid over-correction
max_weight = strategic_weights.min() * 6
strategic_weights = np.clip(strategic_weights, strategic_weights.min(), max_weight)

# Moderate with uniform weights (85% strategic, 15% uniform)
uniform_weights = np.ones_like(strategic_weights)
final_weights = 0.85 * strategic_weights + 0.15 * uniform_weights

class_weights = torch.FloatTensor(final_weights)

print("TARGETED class weights for 75% accuracy:")
print("(Based on baseline: Religious Hate & Sexism have 0% recall)")
for i, weight in enumerate(class_weights):
    print(f"  {class_names[i]}: {weight:.4f}")

# Optimized training arguments for 75% target
training_args = TrainingArguments(
    output_dir="./results_targeted_75",
    num_train_epochs=5,  # Conservative epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,   # Increased warmup for stability
    learning_rate=1.5e-5,  # Lower LR for fine control
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=500,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",  # Focus on accuracy target
    greater_is_better=True,
    save_total_limit=3,
    label_smoothing_factor=0.05,
    dataloader_num_workers=2,
    fp16=True,
)

# Create targeted trainer
trainer = TargetedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_targeted,
)

print("\nStarting TARGETED training for 75% accuracy...")
print("Goal: Fix 0% recall for Religious Hate & Sexism while maintaining overall accuracy")
trainer.train()

# Detailed evaluation
print("\nEvaluating targeted model...")
eval_predictions = trainer.predict(eval_dataset)
predictions = np.argmax(eval_predictions.predictions, axis=1)
labels = eval_dataset["label"]

if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# Calculate improvement
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Per-class performance comparison
precision, recall, f1, support = precision_recall_fscore_support(
    labels, predictions, average=None, zero_division=0
)

print(f"\nTARGETED MODEL RESULTS:")
print(f"Overall Accuracy: {accuracy_score(labels, predictions):.4f}")
print(f"\nPer-Class Recall Comparison:")
print(f"{'Class':<15} {'Baseline':<10} {'Targeted':<10} {'Improvement'}")
print("-" * 50)

baseline_recalls = [0.8587, 0.0000, 0.0000, 0.5567, 0.8153, 0.3777]  # From your baseline
for i, class_name in enumerate(class_names):
    improvement = recall[i] - baseline_recalls[i]
    print(f"{class_name:<15} {baseline_recalls[i]:<10.4f} {recall[i]:<10.4f} {improvement:+.4f}")

# Confusion matrix
cm = confusion_matrix(labels, predictions)
print(f"\nConfusion Matrix:")
print("True\\Pred", end="")
for name in class_names:
    print(f"{name[:8]:>8}", end="")
print()

for i, name in enumerate(class_names):
    print(f"{name[:8]:<8}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i,j]:>8}", end="")
    print()

print(f"\nDetailed Classification Report:")
print(classification_report(labels, predictions, target_names=class_names))

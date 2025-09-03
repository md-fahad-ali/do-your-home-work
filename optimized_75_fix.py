# Optimized approach for 75% accuracy target

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

class OptimizedTrainer(Trainer):
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

def compute_metrics_optimized(eval_pred):
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

# Calculate even milder class weights - prioritize accuracy
train_labels = [int(label) for label in train_dataset["label"]]
classes = np.unique(train_labels)

# Use balanced weights but make them extremely mild
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=np.array(train_labels)
)

# Make weights very mild - cap at 2x minimum weight
min_weight = class_weights.min()
max_weight = min_weight * 2  # Very conservative for 75% accuracy
class_weights = np.clip(class_weights, min_weight, max_weight)

# Even more conservative - 80% uniform, 20% balanced
uniform_weights = np.ones_like(class_weights)
class_weights = 0.8 * uniform_weights + 0.2 * class_weights

class_weights = torch.FloatTensor(class_weights)

print("Ultra-mild class weights for 75% accuracy:")
for i, weight in enumerate(class_weights):
    print(f"  {id2l[i]}: {weight:.4f}")

# Optimized training arguments for 75% accuracy
training_args = TrainingArguments(
    output_dir="./results_75_percent",
    num_train_epochs=6,  # Increased for better convergence
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.08,  # Slightly increased warmup
    learning_rate=1.8e-5,  # Slightly lower for stability
    weight_decay=0.005,  # Reduced weight decay
    lr_scheduler_type="cosine",
    logging_steps=500,
    eval_steps=500,
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    save_total_limit=3,
    label_smoothing_factor=0.03,  # Reduced label smoothing
    dataloader_num_workers=2,
    fp16=True,  # Mixed precision for better training
    gradient_checkpointing=True,  # Memory optimization
)

# Create optimized trainer
trainer = OptimizedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_optimized,
)

print("\nStarting optimized training for 75% accuracy...")
trainer.train()

# Evaluate
print("\nEvaluating optimized model...")
eval_predictions = trainer.predict(eval_dataset)
predictions = np.argmax(eval_predictions.predictions, axis=1)
labels = eval_dataset["label"]

if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

# Generate confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2l.values()))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Optimized for 75% Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nClassification Report - Optimized for 75%:")
print(classification_report(labels, predictions, target_names=list(id2l.values())))

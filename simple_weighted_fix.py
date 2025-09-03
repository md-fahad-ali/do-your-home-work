# Simple weighted approach for 70%+ accuracy - no focal loss

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

class SimpleWeightedTrainer(Trainer):
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

def compute_metrics_simple(eval_pred):
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

# Calculate very mild class weights - just enough to help minority classes
train_labels = [int(label) for label in train_dataset["label"]]
classes = np.unique(train_labels)

# Use balanced weights but make them much milder
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=np.array(train_labels)
)

# Make weights much milder - cap at 3x minimum weight
min_weight = class_weights.min()
max_weight = min_weight * 3  # Very conservative
class_weights = np.clip(class_weights, min_weight, max_weight)

# Further reduce the effect by averaging with uniform weights
uniform_weights = np.ones_like(class_weights)
class_weights = 0.7 * uniform_weights + 0.3 * class_weights  # 70% uniform, 30% balanced

class_weights = torch.FloatTensor(class_weights)

print("Mild class weights for 70%+ accuracy:")
for i, weight in enumerate(class_weights):
    print(f"  {id2l[i]}: {weight:.4f}")

# Conservative training arguments focused on accuracy
training_args = TrainingArguments(
    output_dir="./results_simple",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.06,  # Back to proven value
    learning_rate=2e-5,  # Back to proven value
    weight_decay=0.01,
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
    label_smoothing_factor=0.05,  # Back to proven value
    dataloader_num_workers=2,
)

# Create simple weighted trainer (NO focal loss)
trainer = SimpleWeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_simple,
)

print("\nStarting simple weighted training for 70%+ accuracy...")
trainer.train()

# Evaluate
print("\nEvaluating simple weighted model...")
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
plt.title("Confusion Matrix - Simple Weighted Approach")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nClassification Report - Simple Weighted:")
print(classification_report(labels, predictions, target_names=list(id2l.values())))

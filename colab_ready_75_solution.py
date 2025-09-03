# Standalone Colab-ready solution for 75% accuracy
# No external variable dependencies - works directly in Colab

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

# STEP 1: Replace these with your actual variable names
print("STEP 1: Update variable names to match your notebook")
print("Replace the following lines with your actual variable names:")
print("my_train_dataset = your_train_dataset_variable_name")
print("my_eval_dataset = your_eval_dataset_variable_name") 
print("my_model = your_model_variable_name")
print("my_tokenizer = your_tokenizer_variable_name")
print()

# UNCOMMENT AND MODIFY THESE LINES:
# my_train_dataset = train_dataset  # Replace with your actual train dataset variable
# my_eval_dataset = eval_dataset    # Replace with your actual eval dataset variable
# my_model = model                  # Replace with your actual model variable
# my_tokenizer = tokenizer          # Replace with your actual tokenizer variable

# STEP 2: Define label mapping (based on your dataset analysis)
id2l = {
    0: 'None',
    1: 'Religious Hate', 
    2: 'Sexism',
    3: 'Political Hate',
    4: 'Profane',
    5: 'Abusive'
}

print("STEP 2: Using standard label mapping:")
for k, v in id2l.items():
    print(f"  {k}: {v}")
print()

# STEP 3: Calculate targeted class weights based on your dataset analysis
print("STEP 3: Calculating strategic weights based on dataset analysis")
print("Dataset distribution:")
print("  None: 19,954 samples (55.99%)")
print("  Abusive: 8,212 samples (23.04%)")  
print("  Political Hate: 4,232 samples (11.88%)")
print("  Profane: 2,365 samples (6.64%)")
print("  Religious Hate: 722 samples (2.03%) - 27.6x imbalanced")
print("  Sexism: 152 samples (0.43%) - 131.3x imbalanced")
print()

# Use known class distribution to calculate weights
class_counts = [19954, 722, 152, 4232, 2365, 8212]  # Based on your dataset analysis
total_samples = sum(class_counts)

# Calculate balanced weights manually
balanced_weights = []
for count in class_counts:
    weight = total_samples / (len(class_counts) * count)
    balanced_weights.append(weight)

balanced_weights = np.array(balanced_weights)

# STRATEGIC ADJUSTMENT based on baseline failures:
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
for i, weight in enumerate(class_weights):
    print(f"  {id2l[i]}: {weight:.4f}")
print()

# STEP 4: Training arguments
training_args = TrainingArguments(
    output_dir="./results_targeted_75",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    learning_rate=1.5e-5,
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
    label_smoothing_factor=0.05,
    dataloader_num_workers=2,
    fp16=True,
)

print("STEP 4: Training arguments configured for 75% accuracy target")
print()

# STEP 5: Create trainer (UNCOMMENT AFTER DEFINING YOUR VARIABLES)
print("STEP 5: Uncomment the following lines after defining your variables:")
print("""
trainer = TargetedTrainer(
    class_weights=class_weights,
    model=my_model,
    args=training_args,
    train_dataset=my_train_dataset,
    eval_dataset=my_eval_dataset,
    tokenizer=my_tokenizer,
    compute_metrics=compute_metrics_targeted,
)

print("Starting TARGETED training for 75% accuracy...")
print("Goal: Fix 0% recall for Religious Hate & Sexism")
trainer.train()
""")

print()
print("INSTRUCTIONS:")
print("1. Define your variables (my_train_dataset, my_eval_dataset, etc.)")
print("2. Uncomment the trainer creation and training code")
print("3. Run the training")
print("4. Expected: 75%+ accuracy with some recall for minority classes")

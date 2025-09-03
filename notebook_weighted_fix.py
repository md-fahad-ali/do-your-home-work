# Add this code to your notebook to fix the class imbalance issue

# 1. Import the weighted trainer
import sys
sys.path.append('/home/fahad/Desktop/blp')
from weighted_trainer_fix import WeightedTrainer, calculate_class_weights, create_weighted_training_args, compute_metrics_with_macro_f1

# 2. Calculate class weights based on your training data
train_labels = [int(label) for label in train_dataset["label"]]
class_weights = calculate_class_weights(train_labels, method='balanced')

# 3. Create optimized training arguments
training_args = create_weighted_training_args()

# 4. Create the weighted trainer
trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_with_macro_f1,
)

# 5. Train the model with weighted loss
print("Starting training with weighted loss to handle class imbalance...")
trainer.train()

# 6. Evaluate and generate new confusion matrix
print("\nEvaluating model with weighted loss...")
eval_predictions = trainer.predict(eval_dataset)
predictions = np.argmax(eval_predictions.predictions, axis=1)
labels = eval_dataset["label"]

# Convert tensors if needed
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()
if isinstance(predictions, torch.Tensor):
    predictions = predictions.cpu().numpy()

# Generate new confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

cm = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(id2l.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - After Weighted Loss Fix")
plt.show()

# Print detailed classification report
print("\nClassification Report - After Weighted Loss:")
print(classification_report(labels, predictions, target_names=list(id2l.values())))

print(f"\nClass weights used:")
for i, weight in enumerate(class_weights):
    print(f"  {id2l[i]}: {weight:.4f}")

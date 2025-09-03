# Code to analyze your baseline model performance

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Analyze dataset class distribution
def analyze_dataset(train_dataset, eval_dataset, id2l):
    print("=== DATASET ANALYSIS ===")
    
    # Train set analysis
    train_labels = [int(label) for label in train_dataset["label"]]
    train_counter = Counter(train_labels)
    total_train = len(train_labels)
    
    print(f"\nTrain Dataset ({total_train} samples):")
    for class_id, count in sorted(train_counter.items()):
        percentage = (count / total_train) * 100
        print(f"  {id2l[class_id]}: {count} samples ({percentage:.1f}%)")
    
    # Eval set analysis
    eval_labels = eval_dataset["label"]
    if isinstance(eval_labels, torch.Tensor):
        eval_labels = eval_labels.cpu().numpy()
    
    # Convert to int if needed
    eval_labels = [int(label) for label in eval_labels]
    eval_counter = Counter(eval_labels)
    total_eval = len(eval_labels)
    
    print(f"\nEval Dataset ({total_eval} samples):")
    for class_id, count in sorted(eval_counter.items()):
        percentage = (count / total_eval) * 100
        print(f"  {id2l[class_id]}: {count} samples ({percentage:.1f}%)")
    
    return train_labels, eval_labels

# Evaluate baseline model
def evaluate_baseline_model(model, eval_dataset, tokenizer, id2l):
    print("\n=== BASELINE MODEL EVALUATION ===")
    
    # Get predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            inputs = tokenizer(
                eval_dataset[i]["text"], 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            predictions.append(pred)
            true_labels.append(eval_dataset[i]["label"])
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    
    for i, class_name in enumerate(id2l.values()):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\nMacro Averages:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-Score: {macro_f1:.4f}")
    
    # Classes with 0% recall
    zero_recall_classes = [id2l[i] for i, r in enumerate(recall) if r == 0.0]
    if zero_recall_classes:
        print(f"\nClasses with 0% recall: {zero_recall_classes}")
    else:
        print(f"\nAll classes have some recall")
    
    return predictions, true_labels, accuracy

# Generate detailed confusion matrix
def plot_confusion_matrix(true_labels, predictions, id2l):
    print("\n=== CONFUSION MATRIX ===")
    
    cm = confusion_matrix(true_labels, predictions)
    
    # Print numerical confusion matrix
    print("\nNumerical Confusion Matrix:")
    print("True\\Pred", end="")
    for class_name in id2l.values():
        print(f"{class_name[:8]:>8}", end="")
    print()
    
    for i, class_name in enumerate(id2l.values()):
        print(f"{class_name[:8]:<8}", end="")
        for j in range(len(id2l)):
            print(f"{cm[i,j]:>8}", end="")
        print()
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(id2l.values()),
                yticklabels=list(id2l.values()))
    plt.title('Baseline Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return cm

# Main analysis function
def run_baseline_analysis():
    print("BASELINE MODEL ANALYSIS")
    print("=" * 50)
    
    # 1. Dataset analysis
    train_labels, eval_labels = analyze_dataset(train_dataset, eval_dataset, id2l)
    
    # 2. Model evaluation
    predictions, true_labels, accuracy = evaluate_baseline_model(model, eval_dataset, tokenizer, id2l)
    
    # 3. Confusion matrix
    cm = plot_confusion_matrix(true_labels, predictions, id2l)
    
    # 4. Classification report
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(true_labels, predictions, target_names=list(id2l.values())))
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels,
        'confusion_matrix': cm,
        'train_distribution': Counter(train_labels),
        'eval_distribution': Counter(eval_labels)
    }

# Run the analysis
if __name__ == "__main__":
    results = run_baseline_analysis()

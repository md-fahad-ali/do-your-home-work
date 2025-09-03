# Analyze exact class distribution from training TSV file (no pandas)

from collections import Counter

def analyze_training_dataset(file_path):
    print("=== TRAINING DATASET ANALYSIS ===")
    
    # Read the TSV file manually
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = lines[1:]
    
    print(f"Total dataset size: {len(data_lines)} samples")
    
    # Extract labels and texts
    labels = []
    texts = []
    
    for line in data_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 3:  # id, text, label
            labels.append(parts[2])  # label is 3rd column
            texts.append(parts[1])   # text is 2nd column
    
    # Class distribution
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"\nExact Class Distribution:")
    print(f"{'Class':<15} {'Count':<8} {'Percentage':<12} {'Imbalance Ratio'}")
    print("-" * 55)
    
    # Sort by count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate imbalance ratios (majority class / current class)
    majority_count = sorted_classes[0][1]
    
    for class_name, count in sorted_classes:
        percentage = (count / total_samples) * 100
        imbalance_ratio = majority_count / count
        print(f"{class_name:<15} {count:<8} {percentage:<11.2f}% {imbalance_ratio:<.1f}x")
    
    # Most imbalanced classes
    print(f"\nMost Imbalanced Classes:")
    minority_classes = sorted_classes[-3:]  # Bottom 3
    for class_name, count in reversed(minority_classes):
        percentage = (count / total_samples) * 100
        imbalance_ratio = majority_count / count
        print(f"  {class_name}: {count} samples ({percentage:.2f}%) - {imbalance_ratio:.1f}x imbalanced")
    
    # Sample examples for each class
    print(f"\nSample Examples by Class:")
    for class_name, _ in sorted_classes:
        # Find examples of this class
        examples = []
        for i, label in enumerate(labels):
            if label == class_name and len(examples) < 2:
                examples.append(texts[i])
        
        print(f"\n{class_name}:")
        for i, text in enumerate(examples, 1):
            # Truncate long texts
            text_preview = text[:80] + "..." if len(text) > 80 else text
            print(f"  {i}. {text_preview}")
    
    return {
        'total_samples': total_samples,
        'class_counts': dict(class_counts),
        'sorted_classes': sorted_classes,
        'majority_class': sorted_classes[0][0],
        'minority_classes': [cls[0] for cls in sorted_classes[-3:]]
    }

# Run analysis
if __name__ == "__main__":
    file_path = "/home/fahad/Desktop/blp/blp25_hatespeech_subtask_1A_train.tsv"
    results = analyze_training_dataset(file_path)

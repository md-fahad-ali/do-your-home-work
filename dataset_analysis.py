# Analyze exact class distribution from training TSV file

import pandas as pd
from collections import Counter
import numpy as np

def analyze_training_dataset(file_path):
    print("=== TRAINING DATASET ANALYSIS ===")
    
    # Read the TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    print(f"Total dataset size: {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Class distribution
    class_counts = Counter(df['label'])
    total_samples = len(df)
    
    print(f"\nExact Class Distribution:")
    print(f"{'Class':<15} {'Count':<8} {'Percentage':<12} {'Imbalance Ratio'}")
    print("-" * 50)
    
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
        sample_texts = df[df['label'] == class_name]['text'].head(2).tolist()
        print(f"\n{class_name}:")
        for i, text in enumerate(sample_texts, 1):
            # Truncate long texts
            text_preview = text[:100] + "..." if len(text) > 100 else text
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

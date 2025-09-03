#!/usr/bin/env python3
"""
Script to fix label preprocessing issues in the Unsloth notebook
"""

import json
import sys

def fix_label_preprocessing(notebook_path):
    """Fix label preprocessing to convert string labels to integers"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = []
    
    # Find the data loading cell and add label preprocessing
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix the data loading section to add label mapping
            if "train_df = pd.read_csv('blp25_hatespeech_subtask_1A_train.tsv'" in source:
                new_source = source.replace(
                    "print(f\"Samples > 128 chars: {(train_lengths > 128).sum()} ({(train_lengths > 128).mean()*100:.1f}%)\")\n"
                    "print(f\"Samples > 256 chars: {(train_lengths > 256).sum()} ({(train_lengths > 256).mean()*100:.1f}%)\")",
                    
                    "print(f\"Samples > 128 chars: {(train_lengths > 128).sum()} ({(train_lengths > 128).mean()*100:.1f}%)\")\n"
                    "print(f\"Samples > 256 chars: {(train_lengths > 256).sum()} ({(train_lengths > 256).mean()*100:.1f}%)\")\n"
                    "\n"
                    "# Convert string labels to binary integers for hate speech detection\n"
                    "# Map all hate speech categories to 1, non-hate to 0\n"
                    "def convert_labels_to_binary(df):\n"
                    "    # Create binary labels: 1 for any hate speech, 0 for non-hate\n"
                    "    hate_categories = ['Abusive', 'Political Hate', 'Profane', 'Religious Hate', 'Sexism']\n"
                    "    df['label'] = df['label'].apply(lambda x: 1 if x in hate_categories else 0)\n"
                    "    return df\n"
                    "\n"
                    "# Apply label conversion\n"
                    "train_df = convert_labels_to_binary(train_df)\n"
                    "dev_df = convert_labels_to_binary(dev_df)\n"
                    "test_df = convert_labels_to_binary(test_df)\n"
                    "\n"
                    "print(\"\\nBinary label distribution after conversion:\")\n"
                    "print(f\"Train - Non-hate (0): {(train_df['label'] == 0).sum()}, Hate (1): {(train_df['label'] == 1).sum()}\")\n"
                    "print(f\"Dev - Non-hate (0): {(dev_df['label'] == 0).sum()}, Hate (1): {(dev_df['label'] == 1).sum()}\")\n"
                    "print(f\"Test - Non-hate (0): {(test_df['label'] == 0).sum()}, Hate (1): {(test_df['label'] == 1).sum()}\")"
                )
                
                cell['source'] = new_source.split('\n')
                # Add newlines back properly
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]] if cell['source'] else []
                changes_made.append("Added binary label conversion for hate speech detection")
            
            # Fix tokenization to remove return_tensors="pt" which causes issues
            elif "return_tensors=\"pt\"" in source:
                new_source = source.replace('return_tensors="pt"', 'return_tensors=None')
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]] if cell['source'] else []
                changes_made.append("Fixed tokenization return_tensors parameter")
            
            # Fix the tokenization function to use proper padding
            elif "padding='max_length'" in source:
                new_source = source.replace("padding='max_length'", "padding=False")
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]] if cell['source'] else []
                changes_made.append("Fixed padding in tokenization function")
    
    # Write the fixed notebook
    output_path = notebook_path.replace('.ipynb', '_LABEL_FIXED.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    return output_path, changes_made

if __name__ == "__main__":
    notebook_path = "/home/fahad/Desktop/blp/subtask_1A_DistilBERT_Unsloth_Optimized.ipynb"
    
    try:
        output_path, changes = fix_label_preprocessing(notebook_path)
        print(f"✅ Fixed notebook saved to: {output_path}")
        print(f"📝 Changes made:")
        for change in changes:
            print(f"  - {change}")
        
        if not changes:
            print("  - No changes needed")
            
    except Exception as e:
        print(f"❌ Error fixing notebook: {e}")
        sys.exit(1)

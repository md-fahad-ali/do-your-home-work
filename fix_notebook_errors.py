#!/usr/bin/env python3
"""
Script to fix errors in the DistilBERT notebook
"""

import json
import sys

def fix_notebook_errors(notebook_path):
    """Fix common errors in the notebook"""
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Track changes made
    changes_made = []
    
    # Iterate through cells and fix errors
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix 1: torch-audio -> torchaudio
            if 'torch-audio' in source:
                new_source = source.replace('torch-audio', 'torchaudio')
                cell['source'] = new_source.split('\n')
                # Remove the last empty element if it exists
                if cell['source'] and cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                # Add newlines back to each line except the last
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]] if cell['source'] else []
                changes_made.append("Fixed torch-audio -> torchaudio")
            
            # Fix 2: evaluation_strategy -> eval_strategy
            if 'evaluation_strategy=' in source:
                new_source = source.replace('evaluation_strategy=', 'eval_strategy=')
                cell['source'] = new_source.split('\n')
                # Remove the last empty element if it exists
                if cell['source'] and cell['source'][-1] == '':
                    cell['source'] = cell['source'][:-1]
                # Add newlines back to each line except the last
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]] if cell['source'] else []
                changes_made.append("Fixed evaluation_strategy -> eval_strategy")
    
    # Write the fixed notebook
    output_path = notebook_path.replace('.ipynb', '_FIXED.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    return output_path, changes_made

if __name__ == "__main__":
    notebook_path = "/home/fahad/Desktop/blp/subtask_1A_DistilBERT_Optimized_Fixed.ipynb"
    
    try:
        output_path, changes = fix_notebook_errors(notebook_path)
        print(f"✅ Fixed notebook saved to: {output_path}")
        print(f"📝 Changes made:")
        for change in changes:
            print(f"  - {change}")
        
        if not changes:
            print("  - No changes needed")
            
    except Exception as e:
        print(f"❌ Error fixing notebook: {e}")
        sys.exit(1)

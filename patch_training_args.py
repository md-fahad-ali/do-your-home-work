#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

TARGETS = {
    'eval_strategy_fix': re.compile(r"\beval_strategy\s*="),
    'training_args_start': re.compile(r"training_args\s*=\s*TrainingArguments\s*\("),
    'focal_trainer_start': re.compile(r"FocalWeightedTrainer\s*\("),
}

def patch_training_args_block(block: str) -> str:
    # Rename eval_strategy -> evaluation_strategy
    block = re.sub(r"\beval_strategy\s*=", "evaluation_strategy=", block)
    # Ensure warmup_ratio is 0.1
    block = re.sub(r"warmup_ratio\s*=\s*[0-9]*\.?[0-9]+", "warmup_ratio=0.1", block)
    # Ensure per_device_eval_batch_size=16 present (match commas and whitespace robustly)
    if not re.search(r"per_device_eval_batch_size\s*=", block):
        # insert after per_device_train_batch_size if present, else near start
        m = re.search(r"per_device_train_batch_size\s*=.*?(,\s*\n)", block, flags=re.S)
        if m:
            insert_at = m.end()
            block = block[:insert_at] + "    per_device_eval_batch_size=16,\n" + block[insert_at:]
        else:
            # insert after opening parenthesis and newline
            block = re.sub(r"(TrainingArguments\s*\(\s*\n)", r"\1    per_device_eval_batch_size=16,\n", block)
    else:
        block = re.sub(r"per_device_eval_batch_size\s*=\s*\d+", "per_device_eval_batch_size=16", block)
    # Ensure save_total_limit=2 present
    if not re.search(r"save_total_limit\s*=", block):
        # insert near save_strategy/save_steps area or before closing
        m = re.search(r"save_steps\s*=.*?(,\s*\n)", block, flags=re.S)
        if m:
            insert_at = m.end()
            block = block[:insert_at] + "    save_total_limit=2,\n" + block[insert_at:]
        else:
            block = re.sub(r"\)\s*$", "    save_total_limit=2,\n)", block)
    # Optional: group_by_length and dataloader_num_workers
    if not re.search(r"group_by_length\s*=", block):
        block = block.replace("TrainingArguments(", "TrainingArguments(\n    group_by_length=True,", 1)
    if not re.search(r"dataloader_num_workers\s*=", block):
        block = block.replace("TrainingArguments(", "TrainingArguments(\n    dataloader_num_workers=2,", 1)
    return block


def patch_trainer_block(block: str) -> str:
    # Ensure callbacks with EarlyStoppingCallback present
    if "callbacks=" not in block:
        # insert before closing parenthesis
        block = re.sub(r"\)\s*$", "    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],\n)", block)
    else:
        # keep existing callbacks, but ensure EarlyStoppingCallback is mentioned
        if "EarlyStoppingCallback" not in block:
            block = re.sub(r"callbacks\s*=\s*\[", "callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ", block)
    return block


def ensure_early_stopping_import(cell_source: list) -> list:
    # Find the transformers import tuple and add EarlyStoppingCallback if missing
    joined = "".join(cell_source)
    if "from transformers import (" in joined and "EarlyStoppingCallback" not in joined:
        new_source = []
        for line in cell_source:
            new_source.append(line)
            if line.strip().startswith("set_seed,"):
                # Insert right after set_seed,
                new_source.append("    EarlyStoppingCallback,\n")
        return new_source
    return cell_source


def process_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding='utf-8'))
    changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source', [])
        if not src:
            continue
        text = "".join(src)
        # Patch import cell
        if "from transformers import (" in text:
            new_src = ensure_early_stopping_import(src)
            if new_src != src:
                cell['source'] = new_src
                changed = True
        # Patch TrainingArguments block(s)
        if 'TrainingArguments' in text:
            # Handle possibly multiple blocks in same cell
            def repl(m):
                return patch_training_args_block(m.group(0))
            new_text = re.sub(r"training_args\s*=\s*TrainingArguments\s*\([\s\S]*?\)\s*", repl, text)
            if new_text != text:
                cell['source'] = [new_text]
                changed = True
        # Patch FocalWeightedTrainer block
        if 'FocalWeightedTrainer(' in text:
            new_text = re.sub(r"FocalWeightedTrainer\s*\([\s\S]*?\)\s*", lambda m: patch_trainer_block(m.group(0)), text)
            if new_text != text:
                cell['source'] = [new_text]
                changed = True
    if changed:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        # Write updated notebook
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return changed


def main():
    if len(sys.argv) != 2:
        print("Usage: patch_training_args.py <path-to-notebook.ipynb>")
        sys.exit(1)
    nb_path = Path(sys.argv[1])
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        sys.exit(2)
    changed = process_notebook(nb_path)
    if changed:
        print("✅ Notebook patched. A backup was saved with .bak extension.")
    else:
        print("No changes were necessary. Notebook already configured.")

if __name__ == "__main__":
    main()

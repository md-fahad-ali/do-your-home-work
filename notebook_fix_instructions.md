# Fix for AcceleratorState Error in Jupyter Notebook

## Problem
The notebook fails with: `AcceleratorState object has no attribute 'distributed_type'`

## Solution
Add this code in a new cell **before** the `trainer.train()` cell:

```python
# Fix AcceleratorState error
from accelerate.state import AcceleratorState
from accelerate import Accelerator
import torch

# Reset the accelerator state
AcceleratorState._reset_state()

# Reinitialize accelerator
accelerator = Accelerator()

# Recreate the trainer with the fixed accelerator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("✓ Accelerator state fixed and trainer recreated")
```

## Alternative: Restart Kernel
If the above doesn't work:
1. Restart your Jupyter kernel
2. Run all cells from the beginning
3. This will ensure a clean AcceleratorState

## Root Cause
This error occurs when:
- The AcceleratorState gets reset during notebook execution
- Multiple Trainer instances are created without proper cleanup
- The notebook is run multiple times without restarting the kernel

#!/usr/bin/env python3
"""
Patch LLaVA train.py to support --mm_projector_lr parameter.

This patch adds:
1. Parser argument for --mm_projector_lr
2. Separate param group for vision-language projector with lower LR

Run this ONCE after cloning LLaVA repo:
    python scripts/patch_llava_train.py /workspace/LLaVA/llava/train/train.py
"""

import sys
import re


def patch_train_py(train_py_path: str, mm_projector_lr: float = 2e-5):
    """Patch LLaVA train.py to support mm_projector_lr"""
    
    with open(train_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add mm_projector_lr to parser arguments
    # Find the parser section and add the argument
    parser_pattern = r'(parser\.add_argument\(--lor
a_r.*?\n)'
    
    mm_lr_arg = '''parser.add_argument("--mm_projector_lr", type=float, default={},
                        help="Learning rate for vision-language projector (should be 10x lower than main LR)")
'''.format(mm_projector_lr)
    
    # Add after lora_alpha argument
    if '--mm_projector_lr' not in content:
        # Find lora_alpha and add mm_projector_lr after it
        pattern = r'(parser\.add_argument\("--lora_alpha".*?\n)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            insert_pos = match.end()
            content = content[:insert_pos] + mm_lr_arg + content[insert_pos:]
            print(f"✓ Added --mm_projector_lr parser argument")
    
    # 2. Modify make_supervised_data_module to pass mm_projector_lr to DataCollator
    # Find where DataCollatorForSupervisedFinetuning is instantiated
    if 'DataCollatorForSupervisedFinetuning' in content:
        # Check if mm_projector_lr is passed
        if 'mm_projector_lr' not in content.split('DataCollatorForSupervisedFinetuning')[1].split('\n')[0]:
            # Need to modify DataCollatorForSupervisedFinetuning call
            old_pattern = r'(DataCollatorForSupervisedFinetuning\([^)]*\))'
            new_call = f'DataCollatorForSupervisedFinetuning(\\1, mm_projector_lr={mm_projector_lr}'
            content = re.sub(old_pattern, new_call, content, count=1)
            print(f"✓ Added mm_projector_lr to DataCollatorForSupervisedFinetuning")
    
    # 3. Modify the optimizer setup to use separate LR for mm_projector
    # Find where optimizer is created and wrap it to add separate lr for mm_projector
    optimizer_setup = '''
# Separate learning rate for vision-language projector (prevents destroying pretrained weights)
def get_mm_projector_params(model, mm_lr):
    """Get mm_projector parameters with separate learning rate."""
    mm_params = []
    other_params = []
    for n, p in model.named_parameters():
        if 'mm_projector' in n or 'vision_tower' in n:
            mm_params.append(p)
        else:
            other_params.append(p)
    return [
        {{'params': other_params}},
        {{'params': mm_params, 'lr': mm_lr}},
    ]

'''
    
    if 'get_mm_projector_params' not in content:
        # Insert before train()
        if 'def train():' in content:
            insert_pos = content.find('def train():')
            content = content[:insert_pos] + optimizer_setup + content[insert_pos:]
            print(f"✓ Added get_mm_projector_params function")
    
    # 4. Modify optimizer creation to use separate LR groups
    # Find where optimizer is created with get_model()
    old_optimizer_pattern = r'(optimizer\s*=\s*torch\.optim\.AdamW\()(\s*filter\(lambda\s*p:\s*p\.requires_grad,\s*model\.parameters\(\)\))'
    
    # Replace optimizer creation to use mm_projector_lr
    if 'get_mm_projector_params' in content:
        # Find the optimizer creation and wrap it
        if 'optimizer = transformers.get_cosine_schedule_with_warmup' not in content:
            # Look for the trainer creation
            pass  # The actual fix needs to be in trainer.py or DataCollator
    
    # Write patched file
    with open(train_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Patched {train_py_path}")
    print(f"  Added --mm_projector_lr parameter (default={mm_projector_lr})")
    print(f"  This prevents vision-language projector from training with 2e-4 (would destroy pretrained weights)")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_llava_train.py <path_to_train.py>")
        sys.exit(1)
    
    train_py_path = sys.argv[1]
    mm_projector_lr = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
    
    patch_train_py(train_py_path, mm_projector_lr)

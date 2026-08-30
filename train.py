"""
DARL Training Script
Main entry point for training DARL (Diffusion AutoRegression with Look-ahead) model.

This script initializes the model, loads datasets, and starts the training process
using the DARL framework with Online Monte Carlo Trajectory Generation (OMTG)
and Diffusion Trajectory Preference Optimization (DTPO).
"""

import os
import torch
import torch.serialization
import numpy as np
import _codecs
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor,
    TrainingArguments,
    TrainerCallback
)

# Import custom modules
from src.dataset import MotherDataset
from src.dar_trainer import DARLTrainer

# Safe globals for torch serialization
torch.serialization.add_safe_globals([
    np._core.multiarray._reconstruct, 
    np.ndarray, 
    np.dtype, 
    np.dtypes.UInt32DType, 
    _codecs.encode
])


class OverrideArgsCallback(TrainerCallback):
    """Callback to override training arguments at runtime"""
    def on_train_begin(self, args, state, control, **kwargs):
        args.save_steps = 200
        print(f"\n====== Checkpoint save_steps set to {args.save_steps} ======\n")


def main():
    # WandB configuration for experiment tracking
    os.environ['WANDB_PROJECT'] = 'darl_training'
    os.environ['WANDB_MODE'] = 'offline'  # Set to 'online' for cloud logging
    
    # Model and tokenizer paths
    model_path = 'dots.ocr-3B'  # Update with your model path
    torch_dtype = torch.bfloat16
    
    # Load model with Flash Attention for efficiency
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    )
    
    # Load processor for multimodal inputs
    processor = AutoProcessor.from_pretrained(
        model_path,
        max_pixels=2500*28*28,  # Critical setting for high-resolution documents
        trust_remote_code=True
    )
    
    tokenizer = processor.tokenizer
    
    # Dataset configuration with amplification factors
    # Format: (json_path, amplification_factor)
    all_data_jsons = [
        ('data/magazine/magazine.json', 1),
        ('data/newspaper/newspaper.json', 1),
        ('data/handwritten/handwritten.json', 1),
        ('data/paper/paper.json', 1),
        ('data/exam/exam.json', 1),
        ('data/textbook/textbook.json', 1),
        ('data/presentation/presentation.json', 1),
        ('data/report/report.json', 1),
    ]
    
    print("Loading datasets...")
    train_ds = MotherDataset(
        all_data_jsons=[it[0] for it in all_data_jsons],
        amplification=[it[1] for it in all_data_jsons],
        tokenizer=tokenizer,
        processor=processor
    )
    
    # Enable gradient computation for input embeddings (required for RL)
    model.enable_input_require_grads()
    
    # Training configuration
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./checkpoints/{time_tag}'
    resume_from_checkpoint = os.path.exists(output_dir + '/checkpoint-1000')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        logging_dir=f'./logs/{time_tag}',
        dataloader_num_workers=16,
        dataloader_pin_memory=False,
        logging_steps=10,
        learning_rate=2e-5,
        save_strategy="steps",
        save_steps=200,
        warmup_steps=500,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="wandb",
    )
    
    # Initialize DARL Trainer
    print("Initializing DARL Trainer...")
    trainer = DARLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        max_new_tokens=16,  # Sliding window size (w=16 in paper)
        use_gt_labels=True,
        time_tag=time_tag
    )
    
    # Start training
    print("Starting training...")
    if resume_from_checkpoint:
        print('Resuming from checkpoint...')
    else:
        print('Training from scratch...')
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("Training completed!")


if __name__ == "__main__":
    main()

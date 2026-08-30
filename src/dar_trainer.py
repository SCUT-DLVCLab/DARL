"""
DARL Trainer Implementation
Implements the DARL training framework combining:
1. Diffusion-based parallel generation with sliding window
2. Online Monte Carlo Trajectory Generation (OMTG)
3. Diffusion Trajectory Preference Optimization (DTPO)
"""

import os
import json
import torch
import gc
from datetime import datetime
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .trajectory_generator import make_dar_trajectory_online_full
from .reward_calculator import cal_all_rewards

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DARLTrainer(Trainer):
    """
    DARL Trainer implementing hybrid diffusion-autoregressive training.
    
    Combines supervised learning (DAR loss) with reinforcement learning (DTPO loss)
    for efficient document parsing with speculative decoding.
    """
    
    def __init__(self, model=None, args=None, train_dataset=None,
                 processing_class=None, 
                 max_new_tokens=16,
                 use_gt_labels=True,
                 time_tag='0'):
        super().__init__(model=model, args=args, train_dataset=train_dataset)
        self.train_step_cnt = 0
        self.max_new_tokens = max_new_tokens  # Sliding window size (w in paper)
        self.use_gt_labels = use_gt_labels
        self.processing_class = processing_class
        self.time_tag = time_tag
        self.total_num_samples = 4  # Number of trajectory samples for DTPO
        
        # Loss tracking
        self.ar_losses = []
        self.dar_losses = []
        self.rl_losses = []

    def training_step(self, model, inputs, num_items_in_batch):
        """Main training step combining DAR and DTPO losses"""
        self.train_step_cnt += 1
        return self.consistency_training_step(model, inputs)

    def consistency_training_step(self, model, inputs):
        """
        Combined training step with:
        1. DAR Loss: Supervised diffusion loss for trajectory denoising
        2. DTPO Loss: Reinforcement learning loss based on LCP rewards
        """
        # Loss weights (can be tuned)
        weight_ar = 1.0   # Supervised DAR loss weight
        weight_rl = 0.1   # DTPO reinforcement learning loss weight
        
        # Clear gradients once at the beginning
        self.optimizer.zero_grad()
        
        # ==========================================
        # Part 1: DAR Loss (Supervised Learning)
        # ==========================================
        max_new_tokens = self.max_new_tokens      
        dummy_labels = inputs['input_ids']
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']
        masked_tgt = inputs['labels']
        
        # Generate diffusion trajectories using OMTG
        dar_dummy_inputs, dar_tgts, past_gens, online_infos = make_dar_trajectory_online_full(
            model, pixel_values, image_grid_thw, dummy_labels, masked_tgt, 
            self.processing_class, self.max_new_tokens
        )

        # Pad inputs to ensure consistent window size
        END_PAD_TOKEN_ID = self.processing_class.eos_token_id
        fake_dummy_label_tail = torch.tensor(
            [END_PAD_TOKEN_ID] * (self.max_new_tokens + 2),
            device=masked_tgt.device
        )
        dummy_labels = torch.cat([dummy_labels[0], fake_dummy_label_tail]).unsqueeze(0)
        masked_tgt = torch.cat([masked_tgt[0], fake_dummy_label_tail]).unsqueeze(0)

        # Forward pass for DAR loss
        with self.accelerator.accumulate(model):
            dar_model_output = model(
                input_ids=dar_dummy_inputs[0].unsqueeze(0), 
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw, 
                labels=None
            )
            logits = dar_model_output.logits.float()

            # Compute cross-entropy loss
            loss_fct = CrossEntropyLoss(
                label_smoothing=0.00, 
                ignore_index=-100, 
                reduction='mean'
            )
            dar_labels = dar_tgts[0].unsqueeze(0)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
            shift_labels = dar_labels[:, 1:].contiguous().view(-1).to(logits.device)

            dar_loss = loss_fct(shift_logits, shift_labels)
            
            # Backward with weight
            final_dar_loss = dar_loss * weight_ar
            self.accelerator.backward(final_dar_loss)
            
            loss_dar_item = dar_loss.item()
            
            # Free memory
            del dar_model_output, logits, shift_logits, shift_labels, dar_loss, final_dar_loss
            torch.cuda.empty_cache()
        
        print(f'{os.environ.get("LOCAL_RANK")}: DAR loss backwarded')
        
        # ==========================================
        # Part 2: DTPO Loss (Reinforcement Learning)
        # ==========================================
        loss_rl_global = 0.0
        step_stats = {'reward_mean': 0.0}

        # Check if valid trajectories were generated
        is_valid_rl = True
        if past_gens == [[]] or online_infos[0][2] == 0:
            is_valid_rl = False
            print(f'{os.environ.get("LOCAL_RANK")}: Warning - zero past_gens encountered')

        with self.accelerator.accumulate(model):
            if is_valid_rl:
                # Calculate rewards (LCP + second-order verification)
                with torch.no_grad():
                    rewards_per_func = cal_all_rewards(
                        model, pixel_values, image_grid_thw, 
                        dummy_labels, past_gens, online_infos
                    )
                    rewards = rewards_per_func.sum(dim=1)
                    
                    # GRPO (Group Relative Policy Optimization) advantage calculation
                    mean_grouped_rewards = rewards.view(-1, self.total_num_samples).mean(dim=1)
                    std_grouped_rewards = rewards.view(-1, self.total_num_samples).std(dim=1)
                    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                        self.total_num_samples, dim=0
                    )
                    std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                        self.total_num_samples, dim=0
                    )
                    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
                    
                    step_stats['reward_mean'] = rewards.mean().item()

                # Process each trajectory sample
                SP_INIT_TOKEN_ID = 151642  # Special init token for NAR generation
            
                for i in range(self.total_num_samples):
                    past_gen = past_gens[0][i]
                    n_pad, start_off, n_pickup_from_past, n_new_init = online_infos[0]
                    stable_past = dummy_labels[0][:n_pad + start_off]

                    # Construct NAR input (stable context + init tokens)
                    past_init = torch.tensor(
                        [SP_INIT_TOKEN_ID] * n_pickup_from_past, 
                        device=dummy_labels.device, 
                        dtype=dummy_labels.dtype
                    )
                    this_completion = torch.cat((stable_past, past_init)).unsqueeze(0)
                    
                    adv = advantages[i]

                    # Forward pass
                    outputs = model(
                        this_completion, 
                        pixel_values=pixel_values, 
                        image_grid_thw=image_grid_thw
                    )
                    print(f'{os.environ.get("LOCAL_RANK")}: RL forward {i+1}/{self.total_num_samples}')
                    
                    # Calculate NAR log probabilities
                    nar_logits = outputs.logits[:, -n_pickup_from_past:, :]
                    log_probs = nar_logits.log_softmax(dim=-1)
                    token_log_probs = torch.gather(
                        log_probs, dim=-1, 
                        index=past_gen.unsqueeze(0).unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # Policy gradient loss with advantage weighting
                    per_token_loss = -torch.exp(token_log_probs - token_log_probs.detach()) * adv
                    single_sample_loss = per_token_loss.mean()

                    # Normalize and weight
                    weighted_loss = (single_sample_loss / self.total_num_samples) * weight_rl
                    
                    self.accelerator.backward(weighted_loss)
                    print(f'{os.environ.get("LOCAL_RANK")}: RL backward {i+1}/{self.total_num_samples}')
                    
                    loss_rl_global += weighted_loss.item()

                    # Free memory
                    del outputs, nar_logits, log_probs, token_log_probs
                    del per_token_loss, single_sample_loss, weighted_loss
                    del past_init, this_completion
                    torch.cuda.empty_cache()
            else:
                # Dummy forward for gradient synchronization
                for i in range(self.total_num_samples):
                    dummy_out = model(
                        input_ids=dar_dummy_inputs[0].unsqueeze(0), 
                        pixel_values=pixel_values, 
                        image_grid_thw=image_grid_thw
                    )
                    dummy_loss = dummy_out.logits.sum() * 0.0 
                    self.accelerator.backward(dummy_loss)
                    del dummy_out, dummy_loss
                    torch.cuda.empty_cache()

        # ==========================================
        # Part 3: Optimizer Step
        # ==========================================
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        
        self.optimizer.step()
        print(f'{os.environ.get("LOCAL_RANK")}: Optimizer step completed')

        # Logging
        if self.accelerator.is_main_process:
            print(f"Loss DAR: {loss_dar_item:.4f}, "
                  f"Loss RL: {loss_rl_global:.4f}, "
                  f"Reward: {step_stats['reward_mean']:.4f}")
            
            if self.train_step_cnt % self.args.logging_steps == 0:
                with open(f'{self.args.logging_dir}/log_{self.time_tag}.txt', 'a') as f:
                    log_entry = {
                        'time': datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
                        'step': self.train_step_cnt, 
                        "loss_dar": loss_dar_item, 
                        "loss_rl": loss_rl_global,
                        "reward_mean": step_stats['reward_mean'],
                        "lr": self.optimizer.param_groups[0]['lr']
                    }
                    f.write(json.dumps(log_entry) + '\n')
        
        return torch.tensor(loss_dar_item + loss_rl_global, device=self.accelerator.device)

    def log(self, logs, foo=None):
        """Custom logging to filter dummy loss values"""
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']
        super().log(logs)

    def get_train_dataloader(self):
        """Create custom DataLoader with shuffling"""
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": True,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

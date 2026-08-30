"""
Online Monte Carlo Trajectory Generation (OMTG)
Generates diffusion trajectories for DARL training with sliding window mechanism.

This module implements the core trajectory generation algorithm that:
1. Randomly selects a starting position in the sequence
2. Generates multiple trajectory candidates via temperature sampling
3. Constructs training pairs for diffusion denoising
"""

import torch
import random
import os
import gc

torch.set_printoptions(threshold=5000)

IGNORE_TOKEN_ID = -100


def make_dar_trajectory_online_full(model, pixel_values, image_grid_thw, 
                                    dummy_labels, masked_tgts, tokenizer, max_new_tokens):
    """
    Generate DAR trajectories for a batch of inputs.
    
    Args:
        model: The VLLM model
        pixel_values: Image tensor
        image_grid_thw: Image grid dimensions
        dummy_labels: Input token IDs
        masked_tgts: Target labels with -100 masking
        tokenizer: Tokenizer instance
        max_new_tokens: Sliding window size (w in paper)
        
    Returns:
        Tuple of (dar_inputs, dar_targets, past_generations, online_infos)
    """
    bz = dummy_labels.shape[0]
    dar_dummy_inputs = []
    dar_tgts = []
    past_gens = []
    online_infos = []

    for i in range(bz):
        a_dar_dummy_input, a_dar_tgt, past_gen, online_info = make_a_dar_trajectory_online_full(
            model, pixel_values, image_grid_thw, dummy_labels[i], 
            masked_tgts[i], tokenizer, max_new_tokens
        )
        dar_dummy_inputs.append(a_dar_dummy_input)
        dar_tgts.append(a_dar_tgt)
        past_gens.append(past_gen)
        online_infos.append(online_info)

    return dar_dummy_inputs, dar_tgts, past_gens, online_infos


def get_past_from_model_uncertain_full(model, pixel_values, image_grid_thw, 
                                       past_in, n_pickup_from_past, 
                                       total_num_samples=1, temperature=1.2):
    """
    Generate multiple trajectory candidates via temperature sampling.
    
    This implements the stochastic diffusion initialization in OMTG,
    creating diverse trajectories for preference optimization.
    
    Args:
        model: The VLLM model
        pixel_values: Image tensor
        image_grid_thw: Image grid dimensions
        past_in: Input tokens up to current position
        n_pickup_from_past: Number of tokens to generate
        total_num_samples: Number of trajectory candidates (default=4)
        temperature: Sampling temperature for diversity
        
    Returns:
        Tuple of (logits, sampled_trajectories)
    """
    with torch.inference_mode():
        outputs_logits = model(
            input_ids=past_in.unsqueeze(0), 
            pixel_values=pixel_values, 
            image_grid_thw=image_grid_thw, 
            labels=None
        ).logits[:, -n_pickup_from_past:, :]
    
    if n_pickup_from_past == 0:
        return outputs_logits, [
            torch.tensor([], device=past_in.device, dtype=past_in.dtype) 
            for _ in range(total_num_samples)
        ]
    
    # Temperature-scaled sampling for diversity
    probs = torch.softmax(outputs_logits / temperature, dim=-1)
    all_trajectories = [outputs_logits.argmax(-1)[0]]  # Start with greedy

    # Generate diverse trajectories with deduplication
    max_retries = 20
    retry_count = 0

    while len(all_trajectories) < total_num_samples:
        model_infered = torch.multinomial(
            probs[0], num_samples=1, replacement=True
        ).squeeze(-1)
        
        # Add if not duplicate or max retries reached
        if (not is_dup(all_trajectories, model_infered)) or (retry_count > max_retries):
            all_trajectories.append(model_infered)

        retry_count += 1

    all_trajectories = torch.stack(all_trajectories, dim=0)
    return outputs_logits, all_trajectories


def make_a_dar_trajectory_online_full(model, pixel_values, image_grid_thw, 
                                      dummy_label, masked_tgt, tokenizer, max_new_tokens):
    """
    Generate a single DAR trajectory with sliding window.
    
    This implements the core OMTG algorithm:
    1. Random window position selection
    2. Split window into 'pickup from past' and 'new init' parts
    3. Generate multiple trajectory candidates for the 'pickup' part
    4. Construct training input with special init tokens
    
    Args:
        model: The VLLM model
        pixel_values: Image tensor
        image_grid_thw: Image grid dimensions
        dummy_label: Input token IDs for single sample
        masked_tgt: Target labels with -100 masking
        tokenizer: Tokenizer instance
        max_new_tokens: Window size w
        
    Returns:
        Tuple of (truncated_input, truncated_target, past_generations, online_info)
    """
    MAX_TRUC_LENGTH = 6000  # Maximum sequence length before truncation
    torch.cuda.empty_cache()
    
    END_PAD_TOKEN_ID = tokenizer.eos_token_id
    IMG_PAD_TOKEN_ID = tokenizer.encode('<|imgpad|>')[0]
    SP_INIT_TOKEN_ID = 151642  # Special initialization token for NAR generation

    # Calculate sequence statistics
    n_ign = (masked_tgt == IGNORE_TOKEN_ID).sum().item()  # Instruction tokens
    n_tgt = masked_tgt.shape[-1] - n_ign  # Target tokens
    n_img = (dummy_label == IMG_PAD_TOKEN_ID).sum().item()  # Image tokens
    
    total_ava_num = (masked_tgt != IGNORE_TOKEN_ID).sum().item()
    n_pad = len(masked_tgt) - total_ava_num

    # Pad to ensure minimum window size
    fake_dummy_label_tail = torch.tensor(
        [END_PAD_TOKEN_ID] * max_new_tokens, 
        device=masked_tgt.device
    )
    dummy_label = torch.cat([dummy_label, fake_dummy_label_tail])
    masked_tgt = torch.cat([masked_tgt, fake_dummy_label_tail])

    # Random window position selection
    start_off = random.randint(0, total_ava_num)
    if dummy_label.shape[0] > MAX_TRUC_LENGTH:
        print(f'Truncation triggered! MAX_TRUC_LENGTH-n_ign={MAX_TRUC_LENGTH - n_ign}')
        start_off = random.randint(0, MAX_TRUC_LENGTH - n_ign)

    # Split window: pickup from past + new init
    # This implements the hybrid AR-NAR mechanism
    n_new_init = random.randint(1, max_new_tokens)  # At least 1 for AR
    n_pickup_from_past = max_new_tokens - n_new_init  # Can be 0 if all new

    assert n_pickup_from_past + n_new_init == max_new_tokens, \
        f'{n_pickup_from_past}+{n_new_init}!={max_new_tokens}'
    
    print(f'Window at {start_off}: {n_pickup_from_past} pickup + {n_new_init} new = {max_new_tokens}')
    
    # Construct input with special init tokens for NAR part
    past_ok = dummy_label[:n_pad + start_off].clone()
    past_init = torch.tensor(
        [SP_INIT_TOKEN_ID] * n_pickup_from_past,
        device=masked_tgt.device,
        dtype=dummy_label.dtype
    )
    past_in = torch.cat([past_ok, past_init])
    
    # Generate trajectory candidates via OMTG
    model_out, past_gen = get_past_from_model_uncertain_full(
        model, pixel_values, image_grid_thw, past_in, 
        n_pickup_from_past, total_num_samples=4
    )
    
    # Construct final training input
    new_init = torch.tensor(
        [SP_INIT_TOKEN_ID] * n_new_init, 
        device=masked_tgt.device
    )
    trunced_in = torch.cat([past_ok, past_gen[0], new_init])
    trunced_tgt = masked_tgt[:n_pad + start_off + n_pickup_from_past + n_new_init]

    return trunced_in, trunced_tgt, past_gen, (n_pad, start_off, n_pickup_from_past, n_new_init)


def is_dup(gened_trace, new_infer):
    """
    Check if a generated trajectory is duplicate.
    
    Args:
        gened_trace: List of existing trajectories
        new_infer: New trajectory to check
        
    Returns:
        True if duplicate exists
    """
    for t in gened_trace:
        if (t == new_infer).all():
            return True
    return False

"""
Reward Calculator for DTPO (Diffusion Trajectory Preference Optimization)

Implements two reward functions for trajectory ranking:
1. First-order reward: Longest Common Prefix (LCP) matching
2. Second-order reward: Verification via next-step autoregressive prediction

These rewards guide the preference optimization in DARL training.
"""

import torch


def first_order_reward(teacher_forcing_label, past_gen, online_info):
    """
    Calculate first-order reward based on Longest Common Prefix (LCP).
    
    This reward measures how many tokens at the beginning of a trajectory
    match the ground truth, directly aligning with the sliding verification
    mechanism during speculative inference.
    
    Args:
        teacher_forcing_label: Ground truth labels
        past_gen: List of generated trajectories
        online_info: Tuple of (n_pad, start_off, n_pickup_from_past, n_new_init)
        
    Returns:
        Tensor of shape [n_gen, 1] with normalized LCP scores
    """
    n_pad, start_off, n_pickup_from_past, n_new_init = online_info
    
    # Extract ground truth for the pickup window
    all_right_trace = teacher_forcing_label[
        n_pad + start_off + 1 : n_pad + start_off + n_pickup_from_past + 1
    ]
    
    n_gen = len(past_gen)
    this_reward = torch.zeros((n_gen, 1), dtype=torch.float32, device=teacher_forcing_label.device)
    
    for i in range(n_gen):
        this_trace = past_gen[i]
        
        # Calculate LCP: cumulative product of matches from left to right
        # This rewards longer correct prefixes more than isolated correct tokens
        this_score = torch.eq(this_trace, all_right_trace).int().cumprod(dim=0).sum() / all_right_trace.shape[0]
        this_reward[i] = this_score
    
    return this_reward


def second_order_reward(model, pixel_values, image_grid_thw, 
                       teacher_forcing_label, past_gen, online_info):
    """
    Calculate second-order reward via autoregressive verification.
    
    This reward measures whether a trajectory leads to correct predictions
    in the next autoregressive step, simulating the acceptance behavior
    during speculative decoding.
    
    Args:
        model: The VLLM model for forward prediction
        pixel_values: Image tensor
        image_grid_thw: Image grid dimensions
        teacher_forcing_label: Ground truth labels
        past_gen: List of generated trajectories
        online_info: Tuple of (n_pad, start_off, n_pickup_from_past, n_new_init)
        
    Returns:
        Tensor of shape [n_gen, 1] with normalized verification scores
    """
    SP_INIT_TOKEN_ID = 151642  # Special init token
    
    n_pad, start_off, n_pickup_from_past, n_new_init = online_info
    n_gen = len(past_gen)
    
    this_reward = torch.zeros((n_gen, 1), dtype=torch.float32, device=teacher_forcing_label.device)
    
    # Stable context that doesn't change
    stable_past = teacher_forcing_label[:n_pad + start_off + 1]
    
    # New init tokens for the next AR step
    new_init = torch.tensor(
        [SP_INIT_TOKEN_ID] * n_new_init, 
        device=teacher_forcing_label.device
    )
    
    # Ground truth for verification
    all_right_trace2 = teacher_forcing_label[
        n_pad + start_off + 2 : n_pad + start_off + n_pickup_from_past + n_new_init + 2
    ]
    
    for i in range(n_gen):
        # Construct input: stable context + generated trajectory + new init
        this_input = torch.cat([stable_past, past_gen[i], new_init]).unsqueeze(0)

        # Forward pass to get next-step predictions
        with torch.inference_mode():
            outputs = model(
                input_ids=this_input, 
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw, 
                labels=None
            )
        
        # Extract predictions for the verification window
        this_path = outputs.logits.argmax(-1)
        this_trace2 = this_path[0][-n_pickup_from_past - n_new_init:]
        
        # Calculate LCP for verification
        this_score = torch.eq(this_trace2, all_right_trace2).int().cumprod(dim=0).sum() / all_right_trace2.shape[0]
        this_reward[i] = this_score
    
    return this_reward


def cal_all_rewards(model, pixel_values, image_grid_thw, 
                   teacher_forcing_labels, past_gens, online_infos):
    """
    Calculate combined rewards for all trajectories.
    
    This function combines first-order (LCP) and second-order (verification)
    rewards to provide a comprehensive ranking of trajectory quality.
    
    Args:
        model: The VLLM model
        pixel_values: Image tensor
        image_grid_thw: Image grid dimensions
        teacher_forcing_labels: Ground truth labels
        past_gens: List of generated trajectories per sample
        online_infos: List of online_info tuples
        
    Returns:
        Tensor of shape [n_trajectories, 2] with [LCP_reward, verification_reward]
    """
    # Handle empty trajectory case
    if len(past_gens[0]) == 0:
        print(f'Warning: Empty past_gens detected in reward calculation!')
        return torch.zeros((4, 2), device=teacher_forcing_labels.device)
    
    # Calculate both reward types
    reward1 = first_order_reward(teacher_forcing_labels[0], past_gens[0], online_infos[0])
    reward2 = second_order_reward(
        model, pixel_values, image_grid_thw, 
        teacher_forcing_labels[0], past_gens[0], online_infos[0]
    )
    
    # Combine rewards into single tensor
    rewards_per_func = torch.cat([reward1, reward2], dim=1)
    
    return rewards_per_func
